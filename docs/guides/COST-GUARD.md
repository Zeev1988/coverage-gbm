## Cost Guard

No GPU pod should sit idle and billing without someone noticing. The guard runs every 15 minutes while pods are up. It kills a pod that's idle for no reason, but **only after copying its outputs off**, because pods have `volumeInGb: 0` and terminating one deletes everything on it.

**Conventions it relies on** (see `CLAUDE.md`):
- Pod names start with `RUNPOD_POD_PREFIX-`. `rp.py launch` adds the prefix.
- Jobs write small outputs (logs, metrics) to `/root/out`. Checkpoints go to the HF Hub.
- A pod with `-keep` in its name is never killed.

### One pass
1. Run `uv run tools/rp.py audit`.
2. Pods with no runtime or GPU data yet (booting, exited): log `ok`, reason `booting`. Don't investigate them yet.
3. For each pod with `ours=no`: log `not-ours` and never touch it. For each pod with `-keep` in its name: log `keep` and report it.
4. A pod is **idle** when every GPU is below 10% utilization. If it's not idle, log `ok`.
5. **If it's idle, find out why.** Get the command with `uv run tools/rp.py ssh <id>`, then check:
   - `ps aux --sort=-%cpu | head` and `nvidia-smi`
   - `ls -lt /root/out | head` and `tail -50` of the newest log
   - If the job logs to W&B: is the run still logging? Use the `wandb` Python API, or the run URL in the logs.
6. **Decide:**
   - **Legitimate reason, keep it:** download or install in progress, model loading, expected CPU-bound work (high CPU), or a W&B run still logging. Log `ok` with the reason.
   - **No reason, and the pod's last log line is not an `idle-1` from between 25 and 60 minutes ago:** log `idle-1`. That covers no previous line, a line that's too recent (a manual pass just ran), and a stale one (from an old session).
   - **No reason, and the pod's last log line is an `idle-1` from between 25 and 60 minutes ago:** run `uv run tools/rp.py kill <id> --save`.
     - Exit 0 means it was killed: log `killed`.
     - Exit 1 means the copy failed and the pod is still running: log `ask`.
   - **Unsure:** unknown processes, SSH fails, or `/root/out` is missing or empty (the outputs may be somewhere else, and they'd be lost). Log `ask`. Don't kill.
7. **Log** one tab-separated line per pod to `cost-guard.log` at the repo root:
   `ISO-timestamp  pod_id  name  gpu_util  verdict  reason`
   The verdict is one of `ok|keep|idle-1|killed|ask|not-ours`. Step 6 reads the last line for that pod, including its timestamp.
8. **Notify** with a Claude Code push notification on every `killed` or `ask`. Say the pod, cost per hour, and why.
9. **Stop the loop** when the audit shows no running pods with `ours=yes`.

### Never
- Kill with anything other than `kill <id> --save`.
- Kill a pod that's `not-ours` or has `-keep` in its name.
- Kill after a failed copy. Ask instead.

`rp.py kill` also refuses pods that aren't ours or have `-keep` in their name, so a wrong id fails safe.
