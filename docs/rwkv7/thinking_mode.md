# RWKV7 vLLM 思考模式说明

## 结论

当前模型：

```text
/hikscale/models/XiaoKe/rwkv-0-hf
```

在 vLLM 中**默认开启思考**。

根因不是 vLLM 的 reasoning parser 自动开启，也不是 SP token 适配逻辑造成，而是模型目录中的 `generation_config.json` 写入了：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

vLLM 在未显式传入 `--generation-config` 时默认使用：

```bash
--generation-config auto
```

`auto` 会读取模型目录内的 `generation_config.json`，因此会把 `enable_thinking=true` 注入 chat template。

---

## Chat template 的思考分支

模型模板：

```text
/hikscale/models/XiaoKe/rwkv-0-hf/chat_template.jinja
```

转换器模板源：

```text
/mnt/data/Codes/RWKV-7-World-HF-Converter-special-first/chat_template.jinja
```

Assistant generation prompt 的关键逻辑：

```jinja2
{%- if add_generation_prompt -%}
    {{- '<|im_start|>Assistant:' }}
    {%- if enable_thinking is defined and enable_thinking is true -%}
        {{- ' <think>\n' }}
    {%- elif no_add_thinking is defined and no_add_thinking is true -%}
        {{- ' ' }}
    {%- else -%}
        {{- ' <think>\n\n</think>\n\n' }}
    {%- endif -%}
{%- endif -%}
```

正常 Chat Completions 请求中，vLLM 会设置：

```text
add_generation_prompt = true
```

具体行为：

| 参数 | 实际 Assistant prompt | 行为 |
| --- | --- | --- |
| `enable_thinking=true` | `<|im_start|>Assistant: <think>\n` | 开启思考。模型从未闭合的 `<think>` 后继续生成 reasoning。 |
| `enable_thinking=false` | `<|im_start|>Assistant: <think>\n\n</think>\n\n` | 直答引导。预填一个空且闭合的 think 块。 |
| 不传思考参数，且 generation config 未设置默认值 | `<|im_start|>Assistant: <think>\n\n</think>\n\n` | 与 `enable_thinking=false` 相同。 |
| `no_add_thinking=true` | `<|im_start|>Assistant: ` | 不预填 think 标签；模型仍可能自行输出 `<think>`，不是可靠的关闭方式。 |

---

## 当前服务为什么默认思考

当前模型的 `generation_config.json` 中有：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

因此没有传 `--generation-config` 的 vLLM 服务等价于使用：

```bash
--generation-config auto
```

并最终把 Assistant prompt 渲染为：

```text
<|im_start|>Assistant: <think>
```

由于 `<think>` 未闭合，模型会先生成 reasoning，之后输出 `</think>` 与最终回答。

`--reasoning-parser rwkv` 的职责只是识别模型实际生成的 `<think>...</think>`，并将其中内容返回为 `reasoning` / `reasoning_content`；它不会自行开启思考。

---

## 请求级控制

### 开启思考

```json
{
  "model": "xiaoke-5",
  "messages": [
    {
      "role": "user",
      "content": "解释一下 180 除以 4 的计算过程。"
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

### 关闭思考

推荐显式传：

```json
{
  "model": "xiaoke-5",
  "messages": [
    {
      "role": "user",
      "content": "你是谁"
    }
  ],
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

实测当前服务的固定采样请求：

| 请求方式 | 是否输出 reasoning | completion tokens |
| --- | ---: | ---: |
| 默认请求 | 是 | 约 228 |
| `enable_thinking=false` | 否 | 约 59 |
| `no_add_thinking=true` | 仍可能输出 | 约 280 |

因此：

```text
关闭思考应使用 enable_thinking=false。
不要将 no_add_thinking=true 视为可靠的关闭思考方式。
```

---

## 服务全局默认不思考

若希望服务默认直接回答，同时保留模型 `generation_config.json` 中的温度、top-p、repetition penalty 等默认采样参数，启动时加入：

```bash
--override-generation-config '{"chat_template_kwargs":{"enable_thinking":false}}'
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
RWKV7_USE_FUSED_MIX6=1 \
RWKV7_USE_FUSED_KK_PRE=1 \
RWKV7_USE_FUSED_LNX_RKVRES_XG=1 \
RWKV7_USE_ALT_RECURRENT_KERNEL=1 \
RWKV7_USE_FUSED_CMIX=1 \
RWKV7_USE_DIRECT_LINEAR=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
/mnt/data/anaconda3/envs/vllm-sp/bin/vllm serve \
  /hikscale/models/XiaoKe/rwkv-0-hf \
  --served-model-name xiaoke-5 \
  --reasoning-parser rwkv \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8030 \
  --max-model-len 1M \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --override-generation-config '{"chat_template_kwargs":{"enable_thinking":false}}'
```

此时：

```text
服务默认：不思考
请求传 enable_thinking=true：开启思考
```

不建议仅使用：

```bash
--generation-config vllm
```

因为它会忽略模型 `generation_config.json` 中的采样默认值，例如：

```text
temperature = 0.55
top_p = 0.6
repetition_penalty = 1.2
```

---

## Converter 层面的长期默认值

转换器文件：

```text
/mnt/data/Codes/RWKV-7-World-HF-Converter-special-first/converter.py
```

当前会写入：

```python
"chat_template_kwargs": {
    "enable_thinking": True,
}
```

如果希望模型转换后默认不思考，可改为：

```python
"chat_template_kwargs": {
    "enable_thinking": False,
}
```

重新转换模型后，未显式指定思考参数的 vLLM 服务会默认渲染空且闭合的 think 块：

```text
<|im_start|>Assistant: <think>

</think>

```

即默认直接输出最终回答。
