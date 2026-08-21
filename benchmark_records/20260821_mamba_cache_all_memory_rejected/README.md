# Rejected at initialization: `--mamba-cache-mode all`

The stable RWKV7 server intentionally uses:

```text
--mamba-cache-mode align
--max-model-len 1M
--gpu-memory-utilization 0.92
```

`all` was evaluated as a possible high-concurrency state-lifecycle alternative.
It did not reach CUDA Graph capture or an HTTP benchmark: after loading the
model, vLLM reported 15.47 GiB available for cache state but required
**3965.0 GiB** to serve even one request at `max_model_len=1,048,576`. Its
reported maximum feasible sequence length was only **4,080**.

This is a configuration/resource incompatibility rather than a model-output
precision failure. Reducing `max_model_len` merely to make `all` start would
not be comparable with the required stable 1M-token service. The startup
command was restored to `align`; no source path was changed.
