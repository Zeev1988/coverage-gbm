---
name: cost-guard
description: Use when a RunPod pod is running, when launching one, when the 15-minute cost-guard loop fires, or when the user asks about GPU spend or idle pods.
---

Check `CronList` for an existing cost-guard loop. If there isn't one, start it with `/loop 15m /cost-guard`. Never start a second one. Then run one pass following `docs/guides/COST-GUARD.md`.

- Idle means every GPU is below 10%. Find out why before you decide anything.
- Kill only with `uv run tools/rp.py kill <id> --save`, and only when the pod's last log line is an `idle-1` from between 25 and 60 minutes ago and there's still no legitimate reason. If the copy fails, or you're unsure, notify the user and ask.
- Never kill pods that are `ours=no` or have `-keep` in their name.
- Append every verdict to `cost-guard.log`. Stop the loop when none of our pods are running.
