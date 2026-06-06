# Phantora performance database

- **GPU:** NVIDIA L40S
- **Schema version:** 1
- **compute entries:** 594
- **sequence entries:** 0
- **memcpy entries:** 113
- **flash_attn entries:** 0

Recorded with `--record-perf-db <dir>`. Replay with `--perf-db <dir>` (no GPU required). Each `*.csv` is one timing table (values in nanoseconds); the `compute.csv` `key` column is the exact `TorchCallInfo` as JSON.
