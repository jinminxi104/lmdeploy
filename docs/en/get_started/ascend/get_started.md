# Get Started with Huawei Ascend

We currently support running lmdeploy on **Atlas 800T A3, Atlas 800T A2 and Atlas 300I Duo**.
The usage of lmdeploy on a Huawei Ascend device is almost the same as its usage on CUDA with PytorchEngine in lmdeploy.
Please read the original [Get Started](../get_started.md) guide before reading this tutorial.

Here is the [supported model list](../../supported_models/supported_models.md#PyTorchEngine-on-Other-Platforms).

> \[!IMPORTANT\]
> We have uploaded a docker image with KUNPENG CPU to aliyun.
> Please try to pull the image by following command:
>
> Atlas 800T A3:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a3-latest`
>
> (Atlas 800T A3 currently supports only the Qwen-series with eager mode.)
>
> Atlas 800T A2:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest`
>
> 300I Duo:
>
> `docker pull crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:300i-duo-latest`
>
> (Atlas 300I Duo currently works only with graph mode.)
>
> To build the environment yourself, refer to the Dockerfiles [here](../../../../docker).

## Offline batch inference

### LLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline
from lmdeploy import PytorchEngineConfig
pipe = pipeline("internlm/internlm2_5-7b-chat",
        backend_config=PytorchEngineConfig(tp=1, device_type="ascend"))
question = ["Shanghai is", "Please introduce China", "How are you?"]
response = pipe(question)
print(response)
```

### VLM inference

Set `device_type="ascend"` in the `PytorchEngineConfig`:

```python
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
pipe = pipeline('OpenGVLab/InternVL2-2B',
        backend_config=PytorchEngineConfig(tp=1, device_type='ascend'))
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)
```

## Online serving

### Serve a LLM model

Add `--device ascend` in the serve command.

```bash
lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat
```

Run the following commands to launch docker container for lmdeploy LLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

### Serve a VLM model

Add `--device ascend` in the serve command

```bash
lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B
```

Run the following commands to launch docker container for lmdeploy VLM serving:

```bash
docker run -it --net=host crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy serve api_server --backend pytorch --device ascend OpenGVLab/InternVL2-2B"

### Streaming Output

Ascend backend supports streaming output. After starting the service, you can test streaming output with the following commands:

#### Using curl command

```bash
curl -N -X POST http://localhost:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "internlm2_5-7b-chat", "messages": [{"role": "user", "content": "Tell me about yourself"}], "stream": true }'
```

#### Using Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url="http://0.0.0.0:23333/v1"
)

model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "Tell me about yourself"}
    ],
    stream=True  # Enable streaming output
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## Inference with Command line Interface

Add `--device ascend` in the serve command.

```bash
lmdeploy chat internlm/internlm2_5-7b-chat --backend pytorch --device ascend
```

Run the following commands to launch lmdeploy chatting after starting container:

```bash
docker run -it crpi-4crprmm5baj1v8iv.cn-hangzhou.personal.cr.aliyuncs.com/lmdeploy_dlinfer/ascend:a2-latest \
    bash -i -c "lmdeploy chat --backend pytorch --device ascend internlm/internlm2_5-7b-chat"
```

## Quantization

### w4a16 AWQ

Run the following commands to quantize weights on Atlas 800T A2.

```bash
lmdeploy lite auto_awq $HF_MODEL --work-dir $WORK_DIR --device npu
```

Please check [supported_models](../../supported_models/supported_models.md) before use this feature.

### w8a8 SMOOTH_QUANT

Run the following commands to quantize weights on Atlas 800T A2.

```bash
lmdeploy lite smooth_quant $HF_MODEL --work-dir $WORK_DIR --device npu
```

Please check [supported_models](../../supported_models/supported_models.md) before use this feature.

### int8 KV-cache Quantization

Ascend backend has supported offline int8 KV-cache Quantization on eager mode.

Please refer this [doc](https://github.com/DeepLink-org/dlinfer/blob/main/docs/quant/ascend_kv_quant.md) for details.

## Limitations on 300I Duo

1. only support dtype=float16.
2. only support graph mode, please do not add --eager-mode.
