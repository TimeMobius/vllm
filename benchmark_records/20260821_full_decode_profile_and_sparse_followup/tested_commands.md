# Commands and gates

## Stable service launch

```bash
/tmp/start_rwkv7_fullgraph_t1_direct_cache.sh
```

## Profile-only launch

The profiler run used the same command with these additional flags:

```bash
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /tmp/rwkv7_profile_20260821 \
--profiler-config.torch_profiler_with_stack false \
--profiler-config.ignore_frontend true \
--profiler-config.warmup_iterations 2 \
--profiler-config.active_iterations 8 \
--profiler-config.max_iterations 12
```

Profile controls and workloads:

```bash
curl -X POST http://127.0.0.1:8030/start_profile
/mnt/data/anaconda3/envs/vllm-sp/bin/python /tmp/rwkv7_c1_http_bench.py
curl -X POST http://127.0.0.1:8030/stop_profile

curl -X POST http://127.0.0.1:8030/start_profile
/mnt/data/anaconda3/envs/vllm-sp/bin/python /tmp/rwkv7_c128_http_bench.py
curl -X POST http://127.0.0.1:8030/stop_profile
```

## Restore and validation

```bash
/tmp/start_rwkv7_fullgraph_t1_direct_cache.sh
curl -fsS http://127.0.0.1:8030/health
/mnt/data/anaconda3/envs/vllm-sp/bin/python \
  /tmp/rwkv7_service_accuracy.py http://127.0.0.1:8030
/mnt/data/anaconda3/envs/vllm-sp/bin/python /tmp/rwkv7_c1_http_bench.py
```

Observed final restore result: `/health` HTTP 200; the generated service trace
was byte-identical to the strict baseline; C=1 post-restore was 28.4517 TPS.
No repository runtime code changed in this diagnostic stage, so no new unit-test
coverage claim is made.
