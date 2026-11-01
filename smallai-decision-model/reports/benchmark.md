# Benchmark

- Parameters: 5,374
- FLOPs per decision (analytic, multiply-adds): 14,976
- Latency over 10,000 single decisions: mean 0.246 ms | p50 0.245 ms | p95 0.251 ms
- Vectorised throughput: 253,100 decisions/s (batch 256)
- Peak memory, single decision: 10.4 KiB

## LLM cost proxy (analytical, no API in the loop)

- gpt-4o-mini (published pricing): ~500 tokens/decision
- ~$111.00 per 1M decisions, p50 latency 900 ms vs ≈$0 API cost at 0.245 ms p50 for this model.
