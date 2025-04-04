// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cublasMMWrapper.h"
#include <istream>
#include <ostream>

namespace turbomind {

template<typename T>
class LlamaLinear {
public:
    enum Type
    {
        kGemm,
        kFusedSiluFfn,
        kFusedAdd
    };

    struct Pitched {
        const T* ptr;
        int      pitch;
        Pitched(const T* ptr, int pitch = 0): ptr{ptr}, pitch{pitch} {}
    };

    LlamaLinear(cublasMMWrapper* cublas_wrapper, cudaStream_t stream);

    void forward(T*                         output_data,
                 Pitched                    input_data,
                 int                        batch_size,
                 const LlamaDenseWeight<T>& weight,
                 Type                       type      = kGemm,
                 T*                         lora_buff = nullptr,
                 int*                       lora_mask = nullptr);

    void forward_moe(T*                         output_data,
                     Pitched                    input_data,
                     const int*                 indexes,
                     const int*                 offsets,
                     int                        batch_size,
                     const LlamaDenseWeight<T>& weight,
                     Type                       type,
                     gemm::Context*             context);

    void set_measure(bool measure);

    [[maybe_unused]] int Export(std::ostream& os);

    [[maybe_unused]] int Import(std::istream& is);

    std::vector<int> GetTuningSeq() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace turbomind
