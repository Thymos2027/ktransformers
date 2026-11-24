# Experiment run names

- New concrete run directories must include a Beijing time (Asia/Shanghai, UTC+08:00) timestamp down to seconds, formatted `YYYYMMDD-HHMMSS`, plus a descriptive experiment or retry label. Example: `ep2-expert64-20261124-200221-graph-resume`.
- Use the same resolved run ID for the local `runs/` directory, remote deployment directory, container names and launch metadata. Resolve the timestamp once at launch, not independently for each artifact.
- Top-level report directories may keep their existing date-and-topic format.
- Preserve existing run names and raw evidence references. Do not rename a live run directory while its controller is writing to it.
