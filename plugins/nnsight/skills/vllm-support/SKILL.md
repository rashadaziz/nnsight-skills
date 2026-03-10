---
name: vllm-support
description: Use NNsight with vLLM for fast tracing, async streaming, and activation interventions on vLLM-backed text generation models. Use when workflows need tensor parallelism, one-prompt-per-invoke batching, or generation-step interventions.
---

# vLLM Support

Use this skill when you need NNsight tracing/intervention on top of vLLM inference.

## Required Versions

Install these versions before using this skill:

```bash
pip install -U "nnsight>=0.6" "vllm==0.15.1" "triton==3.5.1"
```

## When to Use vLLM

- Use `VLLM` for faster inference, higher-throughput generation, and async streaming patterns.
- Current support is for text-generation models.
- vLLM traces do not support gradient/backward workflows. Use `LanguageModel` for gradient-based methods.
- Expect some behavior differences versus Transformers backend outputs, even with similar interventions.

## Quick Start

```python
from nnsight.modeling.vllm import VLLM

vllm = VLLM(
    "meta-llama/Llama-3.1-8B-Instruct",
    dispatch=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
)

with vllm.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=5):
    logits = vllm.logits.output.save()
```

## Async Streaming Pattern

Use `mode="async"` and consume outputs from `tracer.backend()`:

```python
import asyncio
from nnsight.modeling.vllm import VLLM

vllm = VLLM("meta-llama/Llama-3.1-8B-Instruct", dispatch=True, mode="async")

async def run(prompt: str):
    with vllm.trace(prompt, temperature=0.7, max_tokens=128) as tracer:
        pass

    full_text = ""
    async for output in tracer.backend():
        if output.outputs:
            new_text = output.outputs[0].text
            delta = new_text[len(full_text):]
            print(delta, end="", flush=True)
            full_text = new_text
    print()

asyncio.run(run("Tell me about Paris."))
```

## Batching Rule: One Prompt Per Invoke

With vLLM, each invoke maps to one request. Do not pass a list of prompts to one invoke.

```python
prompts = [
    "The Eiffel Tower is in the city of",
    "Madison Square Garden is in the city of",
    "The Colosseum is in the city of",
]

with vllm.trace(temperature=0.0, top_p=1.0) as tracer:
    predictions = list().save()  # shared trace-scope state

    for prompt in prompts:
        with tracer.invoke(prompt):
            token_id = vllm.logits.output.argmax(dim=-1)
            predictions.append(vllm.tokenizer.decode(token_id))
```

Key points:
- One prompt per `tracer.invoke(...)`.
- Use a loop of invokes inside one trace for batching.
- Trace-scope variables are shared across invokes.
- Call `.save()` on trace-scope containers you need after the trace exits.

## Safe Interventions Under vLLM

vLLM runs in inference mode. Clone activations before mutation.

```python
neurons = [394, 5490, 8929]
mlp = vllm.model.layers[16].mlp.down_proj

with vllm.trace("The truth is the", max_tokens=1):
    mlp.input = mlp.input.clone()
    mlp.input[-1, neurons] = 10
    out = vllm.output.save()
```

Guidelines:
- Clone before write (`x = x.clone()`), then assign back.
- Add one intervention at a time when debugging.
- Validate effect against an unmodified vLLM baseline first.

## Generation-Step Interventions

Use `tracer.all()` for every generated token step, or `tracer.iter[a:b]` for a subset.

```python
mlp = vllm.model.layers[16].mlp.down_proj

with vllm.trace("The Eiffel Tower is in the city of", max_tokens=10) as tracer:
    hidden_states = list().save()

    with tracer.iter[2:5]:
        mlp.input = mlp.input.clone()
        mlp.input[-1] = 0
        hidden_states.append(mlp.input)
```

## Multi-Feature Steering Patterns

For many simultaneous interventions (for example, multiple SAE features), use grouped-and-sorted steering plus clone-before-write updates.

```python
from collections import defaultdict

POSITION_PRIORITY = {"A": 0, "M": 1, "R": 2}

def build_sorted_steerings(active_steerings):
    """Group by (layer, position, expansion) and sort by forward-pass order."""
    grouped = defaultdict(list)
    for _, info in active_steerings.items():
        key = (info["layer"], info["position"], info["expansion"])
        grouped[key].append((info["index"], info["scale"]))
    return sorted(
        [(layer, pos, exp, mods) for (layer, pos, exp), mods in grouped.items()],
        key=lambda x: (x[0], POSITION_PRIORITY[x[1]]),
    )

def apply_sorted_steerings(vllm, sorted_steerings, steer_fn, state=None):
    """Apply grouped steerings safely under vLLM inference mode."""
    for layer, position, expansion, mods in sorted_steerings:
        if position == "R":
            layer_out = vllm.model.layers[layer].output
            hidden = layer_out[0].clone()
            vllm.model.layers[layer].output = (
                steer_fn(hidden, layer, position, expansion, mods, state=state),
                *layer_out[1:],
            )
        elif position == "M":
            out = vllm.model.layers[layer].mlp.output.clone()
            vllm.model.layers[layer].mlp.output = steer_fn(
                out, layer, position, expansion, mods, state=state
            )
        elif position == "A":
            out = vllm.model.layers[layer].self_attn.output.clone()
            vllm.model.layers[layer].self_attn.output = steer_fn(
                out, layer, position, expansion, mods, state=state
            )
```

```python
# Example steering specs (plain scalars only, safe across process boundaries)
active_steerings = {
    "shakespeare": {"layer": 28, "position": "R", "expansion": "8x", "index": 8401, "scale": 10.0},
    "romance": {"layer": 22, "position": "R", "expansion": "8x", "index": 9321, "scale": 8.0},
}
sorted_steerings = build_sorted_steerings(active_steerings)

with vllm.trace("Tell me about Paris", temperature=0.0, max_tokens=80) as tracer:
    state = {"max_steps": 10}  # optional cutoff to reduce long-run degeneration
    for _ in tracer.iter[:]:
        apply_sorted_steerings(
            vllm,
            sorted_steerings,
            steer_fn=steer_activations,  # your worker-safe steering function
            state=state,
        )
```

Key points:
- Sort by `(layer, position)` to follow forward execution order.
- Group same hook locations to avoid repeated reads/writes.
- Clone tensors before mutation.
- Keep payloads simple (indices/scales), and lazy-load heavy objects inside `steer_fn`.

## Troubleshooting

- Version mismatch errors: re-check exact pins (`nnsight>=0.6`, `vllm==0.15.1`, `triton==3.5.1`).
- Async loop returns nothing: confirm `mode="async"` and that you iterate `tracer.backend()`.
- In-place mutation errors: clone tensors before modifying.
- Batching confusion: one prompt per invoke; batch by looping invokes.
- Gradient calls fail: expected for vLLM traces; switch to `LanguageModel` if gradients are required.
