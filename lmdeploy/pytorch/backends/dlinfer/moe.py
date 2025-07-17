# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from sglang.srt.layers.moe.ep_moe.kernels import (post_reorder_triton_kernel, pre_reorder_triton_kernel,
                                                  run_moe_ep_preproess, silu_and_mul_triton_kernel)
from sglang.srt.layers.moe.ep_moe.layer import GroupedGemmRunner

from lmdeploy.pytorch.distributed import get_ep_world_rank
from lmdeploy.pytorch.kernels.dlinfer import moe_gating_topk_softmax

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """Dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """Dlinfer fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size
        # torch.distributed.get
        self.ep_size, self.ep_rank = get_ep_world_rank()
        # self.tp_size, self.tp_rank = get_tp_world_rank()
        # self.tp_size, self.tp_rank = get_world_rank()
        # if self.tp_rank == 0:
        print('self.ep_size, self.ep_rank:', self.ep_size, self.ep_rank, self.num_experts, flush=True)
        # if self.tp_rank == 1:
        # print("self.tp_size, self.tp_rank:", self.tp_size, self.tp_rank, self.num_experts, flush=True)
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.w2_input_scale = None
        self.use_block_quant = False

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def support_ep(self):
        """Support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        expert_per_rank = (self.num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, self.num_experts)
        return list(range(first_expert, last_expert))

    # from sglang/srt/layers/moe/ep_moe/layer.py class EPMoE forward
    def tmp_forward(self, hidden_states: torch.Tensor, gate_up_weights: torch.Tensor, down_weights: torch.Tensor,
                    topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        # print("test tmp_forward!", flush=True)
        # if self.grouped_gemm_runner is None:
        self.grouped_gemm_runner = GroupedGemmRunner(hidden_states.device, )

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(topk_ids, self.num_experts)

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0], )](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            None,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id:self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            gateup_input.shape[0],
            gate_up_weights.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=gate_up_weights,
            c=gateup_output,
            batch_size=self.num_experts_per_partition,
            top_k=self.top_k,
            weight_column_major=True,
            gateup_stage=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            block_shape=None,
        )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )

        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )

        silu_and_mul_triton_kernel[(gateup_output.shape[0], )](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            self.w2_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            BLOCK_SIZE=512,
        )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            down_weights.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=down_weights,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            top_k=self.top_k,
            weight_column_major=True,
            gateup_stage=False,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            scale_a=self.w2_input_scale,
            block_shape=None,
        )

        # PostReorder
        output = torch.empty_like(hidden_states)
        post_reorder_triton_kernel[(hidden_states.size(0), )](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )
        return output

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""

        q_seq = hidden_states.shape[0]
        topk_weights = topk_weights.reshape(q_seq, -1).contiguous()
        topk_ids = topk_ids.reshape(q_seq, -1).contiguous()
        # if q_seq != 1:
        #     print(f"self.top_k: {self.top_k}.", flush=True)
        #     print(f"hidden_states.shape: {hidden_states.shape}.", flush=True)
        #     print(f"gate_up_weights.shape: {gate_up_weights.shape}.", flush=True)
        #     print(f"down_weights.shape: {down_weights.shape}.", flush=True)
        #     # print(f"topk_ids.shape: {topk_ids.shape}.", flush=True)
        #     print(f"topk_weights.shape: {topk_weights.shape}.", flush=True)

        # from lmdeploy.pytorch.kernels.dlinfer import fused_moe
        # return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights,
        #                  topk_ids, self.top_k, self.num_experts, self.ep_size, self.renormalize, expert_list)

        # import pdb; pdb.set_trace()
        # if True:
        # # if expert_list is None:
        #     # print("!!!!!!!!!!!!", flush=True)
        #     from lmdeploy.pytorch.kernels.dlinfer import fused_moe
        #     # from lmdeploy.pytorch.backends.cuda.moe import fused_moe
        # return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights,
        #                  topk_ids, self.top_k, 1, 1, self.renormalize)
        # # print("test tmp_forward!", flush=True)
        out = self.tmp_forward(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids)
        # # print("test tmp_forward end!", flush=True)
        # if self.tp_size > 1:
        #     dist.all_reduce(out, group='tp')

        # dist.all_reduce(out, group='tp')
        if self.ep_size > 1:
            print('running 1', flush=True)
            # dist.all_reduce(out, group='dp')
            # dist.all_reduce(out, group='tp')
            # dist.all_reduce(out, group='ep')
            print('running 2', flush=True)
            # print("running 3", flush=True)
        return out

        # ep
        # expert_offset = 0
        # num_experts = None
        # if expert_list is not None and len(expert_list) != self.num_experts:
        #     expert_offset = expert_list[0]
        #     num_experts = self.num_experts
        # from lmdeploy.pytorch.backends.cuda.moe import fused_moe
        # out = fused_moe(hidden_states,
        #                 gate_up_weights,
        #                 down_weights,
        #                 topk_weights=topk_weights,
        #                 topk_ids=topk_ids,
        #                 topk=self.top_k,
        #                 expert_offset=expert_offset,
        #                 num_experts=num_experts,
        #                 renormalize=self.renormalize)
        # return out


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """Dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False, ep_size: int = 1):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize, ep_size=ep_size)
