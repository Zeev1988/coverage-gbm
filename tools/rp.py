"""RunPod helper. Reads .env at the repo root (see .env.example). Run with `uv run tools/rp.py ...`.
  avail <gpuTypeId> <count>
  launch <name> <gpuTypeId> <count> <disk_gb> [secure|community] [--image IMG]   # name gets RUNPOD_POD_PREFIX-
  status <pod_id>
  audit                          # all pods: cost, uptime, GPU/CPU util, ours=yes|no
  ssh <pod_id>                   # print the ssh command
  save <pod_id> [dest]           # rsync /root/out/ -> dest (default runs/<pod_id>/); exit 1 on failure
  kill <pod_id> [--save] [--force]  # only our pods, never -keep (unless --force); --save: only if save succeeded
  watchdog <pod_id> <hours>      # sleep, then kill --save, retrying every 15m (run with nohup)
  selftest
ponytail: GraphQL via stdlib urllib; no SDK.
"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Image has no sshd: install it, authorize our key, then idle.
BOOT = ("bash -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server rsync >/dev/null; "
        "mkdir -p /root/.ssh /run/sshd /root/out && echo \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && "
        "chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity'")


# Pods are short-lived and RunPod reuses ip:port across pods, so known_hosts would only cause false mismatches.
SSH_OPTS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"


def load_env(root: Path) -> dict:
    env = {}
    p = root / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("\"'")
    return {**env, **os.environ}  # real environment wins


ENV = load_env(ROOT)


def pod_prefix(env: dict, root: Path) -> str:
    return env.get("RUNPOD_POD_PREFIX") or root.name


def prefixed(name: str, prefix: str) -> str:
    return name if name.startswith(f"{prefix}-") else f"{prefix}-{name}"


def is_ours(pod_name: str, prefix: str) -> bool:
    return pod_name.startswith(f"{prefix}-")


def ssh_target(pod: dict):
    for x in (pod.get("runtime") or {}).get("ports") or []:
        if x["privatePort"] == 22 and x["isIpPublic"]:
            return x["ip"], x["publicPort"]
    return None


def pubkey_path() -> Path:
    return Path(ENV.get("RUNPOD_PUBKEY") or Path.home() / ".ssh/runpod_id.pub").expanduser()


def privkey_path() -> Path:
    p = pubkey_path()
    return p.with_suffix("") if p.suffix == ".pub" else p


def gql(query, variables=None):
    key = ENV.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("missing RUNPOD_API_KEY: copy .env.example to .env")
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "rp-helper"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:800]}")
    if out.get("errors"):
        sys.exit(json.dumps(out["errors"], indent=1))
    return out["data"]


STATUS_Q = """query($id:String!){ pod(input:{podId:$id}){ id name desiredStatus costPerHr
  runtime{ uptimeInSeconds ports{ ip isIpPublic privatePort publicPort } } } }"""


def get_pod(pod_id: str) -> dict:
    return gql(STATUS_Q, {"id": pod_id})["pod"] or {}


def save(pod_id: str, dest: Path | None = None) -> bool:
    t = ssh_target(get_pod(pod_id))
    if not t:
        print(f"save {pod_id}: no public ssh port", file=sys.stderr)
        return False
    dest = Path(dest or ROOT / "runs" / pod_id)
    dest.mkdir(parents=True, exist_ok=True)
    ssh = f"ssh -p {t[1]} -i '{privkey_path()}' {SSH_OPTS}"
    r = subprocess.run(["rsync", "-az", "-e", ssh, f"root@{t[0]}:/root/out/", f"{dest}/"])
    if r.returncode:
        print(f"save {pod_id}: rsync exit {r.returncode}", file=sys.stderr)
    return r.returncode == 0


def kill(pod_id: str, save_first: bool = False, force: bool = False) -> bool:
    name = get_pod(pod_id).get("name") or ""
    if not force and (not is_ours(name, pod_prefix(ENV, ROOT)) or "-keep" in name):
        print(f"NOT killing {pod_id} ({name or 'not found'}): not ours or -keep; use --force", file=sys.stderr)
        return False
    if save_first and not save(pod_id):
        print(f"NOT killing {pod_id}: save failed", file=sys.stderr)
        return False
    gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pod_id})
    print("terminated", pod_id)
    return True


def watchdog(pod_id: str, hours: float, retry_s: int = 900) -> bool:
    """Sleep, then copy-then-kill, retrying every retry_s until it works or the pod is gone."""
    time.sleep(hours * 3600)
    while get_pod(pod_id) and not kill(pod_id, save_first=True):
        time.sleep(retry_s)
    return True


def flag(args: list, name: str, value: bool = True):
    """Pop --name [value] from args."""
    if name not in args:
        return None
    i = args.index(name)
    out = args[i + 1] if value else True
    del args[i:i + (2 if value else 1)]
    return out


def main(cmd, *a):
    a = list(a)
    prefix = pod_prefix(ENV, ROOT)
    if cmd == "avail":
        g, n = a[0], int(a[1])
        q = """query($id:String,$n:Int!){ gpuTypes(input:{id:$id}){ id
          c: lowestPrice(input:{gpuCount:$n, secureCloud:false}){ uninterruptablePrice stockStatus }
          s: lowestPrice(input:{gpuCount:$n, secureCloud:true}){ uninterruptablePrice stockStatus } } }"""
        print(json.dumps(gql(q, {"id": g, "n": n}), indent=1))
    elif cmd == "launch":
        image = flag(a, "--image") or ENV.get("RUNPOD_IMAGE") or sys.exit("set RUNPOD_IMAGE in .env or pass --image")
        if not pubkey_path().exists():
            sys.exit(f"missing ssh pubkey {pubkey_path()}: set RUNPOD_PUBKEY")
        if not ENV.get("RUNPOD_POD_PREFIX"):
            print(f"warning: RUNPOD_POD_PREFIX unset, using '{prefix}'. Make it unique on the account.", file=sys.stderr)
        name, g, n, disk = prefixed(a[0], prefix), a[1], int(a[2]), int(a[3])
        cloud = (a[4] if len(a) > 4 else "secure").upper()
        q = """mutation($i:PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$i){ id name costPerHr machineId } }"""
        i = {"cloudType": cloud, "gpuCount": n, "gpuTypeId": g, "name": name, "imageName": image,
             "containerDiskInGb": disk, "volumeInGb": 0, "ports": "22/tcp", "dockerArgs": BOOT,
             "supportPublicIp": True, "env": [{"key": "PUBLIC_KEY", "value": pubkey_path().read_text().strip()}]}
        print(json.dumps(gql(q, {"i": i}), indent=1))
    elif cmd == "status":
        print(json.dumps(get_pod(a[0]), indent=1))
    elif cmd == "ssh":
        t = ssh_target(get_pod(a[0])) or sys.exit(f"{a[0]}: no public ssh port")
        print(f"ssh root@{t[0]} -p {t[1]} -i '{privkey_path()}' {SSH_OPTS}")
    elif cmd == "save":
        sys.exit(0 if save(a[0], a[1] if len(a) > 1 else None) else 1)
    elif cmd == "kill":
        force = bool(flag(a, "--force", value=False))
        sys.exit(0 if kill(a[0], save_first=bool(flag(a, "--save", value=False)), force=force) else 1)
    elif cmd == "audit":  # every pod on the account: cost, uptime, GPU/CPU utilization (from RunPod telemetry)
        q = """query{ myself{ clientBalance currentSpendPerHr pods{ id name desiredStatus costPerHr
          runtime{ uptimeInSeconds gpus{ id gpuUtilPercent memoryUtilPercent }
                   container{ cpuPercent memoryPercent }
                   ports{ ip isIpPublic privatePort publicPort } } } } }"""
        me = gql(q)["myself"]
        print(f"balance=${me['clientBalance']:.2f} spend=${me['currentSpendPerHr']:.2f}/h pods={len(me['pods'])} prefix={prefix}")
        for p in me["pods"]:
            rt = p["runtime"] or {}
            gpus = rt.get("gpus") or []
            t = ssh_target(p)
            print(f"- {p['id']} {p['name']} {p['desiredStatus']} ${p['costPerHr']}/h "
                  f"up={(rt.get('uptimeInSeconds') or 0) / 3600:.2f}h ssh={f'{t[0]}:{t[1]}' if t else '-'} "
                  f"gpu_util={[g['gpuUtilPercent'] for g in gpus]} gpu_mem={[g['memoryUtilPercent'] for g in gpus]} "
                  f"cpu={(rt.get('container') or {}).get('cpuPercent')}% ours={'yes' if is_ours(p['name'], prefix) else 'no'}")
    elif cmd == "watchdog":
        watchdog(a[0], float(a[1]))
    elif cmd == "selftest":
        selftest()
    else:
        sys.exit(__doc__)


def selftest():
    assert pod_prefix({}, Path("/x/icwm")) == "icwm"
    assert pod_prefix({"RUNPOD_POD_PREFIX": "s1"}, Path("/x/icwm")) == "s1"
    assert pod_prefix({"RUNPOD_POD_PREFIX": ""}, Path("/x/icwm")) == "icwm"
    assert prefixed("run", "s1") == "s1-run" and prefixed("s1-run", "s1") == "s1-run"
    assert is_ours("s1-run", "s1") and not is_ours("s12-run", "s1") and not is_ours("s1", "s1")
    pod = {"runtime": {"ports": [{"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 22, "publicPort": 4022}]}}
    assert ssh_target(pod) == ("1.2.3.4", 4022)
    assert ssh_target({"runtime": None}) is None
    assert ssh_target({"runtime": {"ports": None}}) is None
    # kill --save: a failed save must never terminate
    g = globals()
    calls = []
    pre = pod_prefix(ENV, ROOT)
    g["get_pod"] = lambda pid: {"name": f"{pre}-run"}
    g["save"] = lambda *a, **k: False
    g["gql"] = lambda *a, **k: calls.append(a)
    assert kill("p", save_first=True) is False and calls == []
    g["save"] = lambda *a, **k: True
    assert kill("p", save_first=True) is True and len(calls) == 1
    # kill refuses pods that aren't ours or are marked -keep, unless forced
    for name in ("other-run", f"{pre}-run-keep", ""):
        g["get_pod"] = lambda pid, n=name: {"name": n}
        assert kill("p", save_first=True) is False and len(calls) == 1, name
    assert kill("p", force=True) is True and len(calls) == 2
    # watchdog retries a failed copy-then-kill until it works or the pod is gone
    sleeps, tries = [], iter([False, False, True])
    time.sleep, g["kill"] = sleeps.append, lambda *a, **k: next(tries)
    g["get_pod"] = lambda pid: {"name": f"{pre}-run"}
    assert watchdog("p", 1) is True and sleeps == [3600, 900, 900]
    sleeps.clear(); g["get_pod"] = lambda pid: {}
    assert watchdog("p", 1) is True and sleeps == [3600]
    print("selftest ok")


if __name__ == "__main__":
    main(*sys.argv[1:]) if sys.argv[1:] else sys.exit(__doc__)
