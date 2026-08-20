# RWKV7 Performance Records

This directory stores immutable, reproducible performance and accuracy
artifacts for RWKV7 optimizations. Do not replace a previous run: create a new
run directory or comparison JSON for every before/after experiment.

Generate the self-contained performance dashboard with:

```bash
/mnt/data/anaconda3/envs/vllm-sp/bin/python tools/rwkv7_perf_dashboard.py
```

Open `benchmark_records/dashboard.html` in a browser. The chart shows only
records from `*_comparison.json`; each record must identify its GPU, workload,
parent algorithm, raw results, and an accuracy comparison artifact.
