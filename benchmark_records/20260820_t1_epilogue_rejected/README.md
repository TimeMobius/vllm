# Rejected T=1 recurrent + epilogue fusion

The candidate fused the T=1 recurrent state update, recurrent output reduction,
layer norm, rkv residual correction, and gate into one Triton kernel. Isolated
operator timings improved strongly (for example, B=8: 0.05480 ms -> 0.03174 ms), but the service precision gate failed.

- Service baseline: 137.972 TPS
- Candidate service: 140.176 TPS (+1.60%)
- Cache-on serial comparison: 48/256 token IDs mismatched across 8 prompts
- Candidate repeated serial run: 0 mismatches against itself

The source candidate was reverted. Raw operator timings, service throughput,
and token IDs remain in this directory for future work on numerically exact
epilogue fusion.
