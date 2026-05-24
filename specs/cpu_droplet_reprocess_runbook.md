# CPU Droplet Reprocess Runbook

How to run `tools/reprocess_hough.py` against a coin-id list on a separate
CPU-only droplet (DO or RunPod CPU) to free up the primary box.

This is the CPU-only twin of the GPU runs documented in
`/root/trivalaya-pipeline/docs/runpod_first_run.md`. The vision pipeline
has no torch/cuda dependency — Spaces I/O + OpenCV + boto3 are the only
moving parts, so a 16-32 vCPU instance is the cheapest path to throughput.

---

## 1. Provision the droplet

| spec | value |
|---|---|
| OS | Ubuntu 22.04 or 24.04 |
| CPU | 16 vCPU (cheap) or 32 vCPU (faster) |
| RAM | ≥4 GB (each worker uses ~300 MB) |
| Disk | 20 GB (no source caching — Spaces stream-through) |
| Region | SFO3 if possible (same as Spaces); any US region is fine |

DigitalOcean CLI:

```bash
doctl compute droplet create vision-cpu-burst \
  --image ubuntu-24-04-x64 \
  --size c-16   `# 16 vCPU 32GB; "c-32" for 32 vCPU` \
  --region sfo3 \
  --ssh-keys $(doctl compute ssh-key list --format ID --no-header | head -1) \
  --enable-monitoring \
  --wait
```

RunPod CPU pod (cheaper): pick "CPU Pod" template, 16+ vCPUs, 8 GB RAM,
20 GB volume. SSH details show up after deploy.

---

## 2. Bootstrap the droplet

SSH in as root, then:

```bash
apt-get update -qq
apt-get install -y python3 python3-venv python3-pip git libgl1 libglib2.0-0
# libgl1 + libglib2.0-0 are required by OpenCV; minimal install otherwise
```

---

## 3. Clone the repos

The reprocess script lives in `trivalaya-vision`; the DB credentials and
auction-data table live with `trivalaya-pipeline` (we only need `.env`).

```bash
cd /root
git clone <your-fork>/trivalaya-vision.git
git clone <your-fork>/trivalaya-pipeline.git

cd trivalaya-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q opencv-python-headless==4.10.0.84 numpy boto3 mysql-connector-python pillow
```

> Use `opencv-python-headless`, not `opencv-python` — droplets don't have a display
> server and the full package will fail to import.

---

## 4. Wire up secrets

Two env files are required. The reprocess script reads both implicitly.

`~/.bashrc` (or `~/.env-vision` sourced before each run):

```bash
export SPACES_BUCKET=trivalaya-data
export SPACES_REGION=sfo3
export SPACES_ENDPOINT=https://sfo3.digitaloceanspaces.com
export AWS_ACCESS_KEY_ID=<from primary droplet env>
export AWS_SECRET_ACCESS_KEY=<from primary droplet env>
```

`/root/trivalaya-pipeline/.env` (copy from primary):

```bash
TRIVALAYA_DB_HOST=<primary droplet IP — must be reachable from the burst droplet>
TRIVALAYA_DB_USER=auction_user
TRIVALAYA_DB_PASSWORD=<from primary>
TRIVALAYA_DB_NAME=auction_data
```

**Important**: the primary droplet's MySQL is bound to `127.0.0.1` by default.
To let the burst droplet read from it:
1. On the primary, edit `/etc/mysql/mysql.conf.d/mysqld.cnf` and set
   `bind-address = 0.0.0.0` (or the primary's private IP if both are in the
   same DO VPC).
2. `GRANT SELECT ON auction_data.* TO 'auction_user'@'<burst droplet IP>';`
3. `FLUSH PRIVILEGES;`
4. Open port 3306 in the firewall for the burst droplet's IP.
5. Restart MySQL.

Alternative: SSH-tunnel from burst droplet to primary:
`ssh -fNL 3306:127.0.0.1:3306 root@<primary IP>` and keep `TRIVALAYA_DB_HOST=127.0.0.1`.

---

## 5. Prepare the coin-id list

Two options:

**(a) Generate the list on the primary, scp over:**

```bash
# on primary:
scp /tmp/v1v2/transparent_redo.txt root@<burst IP>:/tmp/coins.txt
# or for a smaller scope, e.g. /tmp/v1v2/scan866_eaten_coins.txt
```

**(b) Generate on the burst droplet (needs DB access):**

```bash
# on burst:
python3 - <<'EOF'
import os, mysql.connector
for line in open("/root/trivalaya-pipeline/.env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k,v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip("'\""))
conn = mysql.connector.connect(
    host=os.environ["TRIVALAYA_DB_HOST"],
    user=os.environ["TRIVALAYA_DB_USER"],
    password=os.environ["TRIVALAYA_DB_PASSWORD"],
    database=os.environ["TRIVALAYA_DB_NAME"])
cur = conn.cursor(dictionary=True)
# Example: all coins from a specific sale
cur.execute("""
    SELECT c.id FROM coins c
    JOIN auction_data a ON a.id = c.auction_record_id
    WHERE a.auction_house='cng' AND a.sale_id='614'
""")
ids = [r["id"] for r in cur.fetchall()]
open("/tmp/coins.txt","w").write(",".join(str(x) for x in ids))
print(f"wrote {len(ids)} coin_ids")
EOF
```

---

## 6. Run the batch

```bash
cd /root/trivalaya-vision
source .venv/bin/activate
source ~/.env-vision    # if you used a separate file for SPACES creds

# Smoke test on 5 coins first
head -c 200 /tmp/coins.txt | tr ',' '\n' | head -5 | tr '\n' ',' > /tmp/coins_smoke.txt
python3 tools/reprocess_hough.py \
  --coin-ids "$(cat /tmp/coins_smoke.txt)" \
  --no-hough-filter \
  --workers 4 \
  --log /tmp/smoke_log.jsonl

# Inspect smoke log; if OK, launch the full batch in tmux/nohup
tmux new -d -s repro "python3 tools/reprocess_hough.py \
  --coin-ids-file /tmp/coins.txt \
  --no-hough-filter \
  --resume \
  --workers 16 \
  --log /tmp/repro_log.jsonl"
```

Worker count guidance:
- `--workers N` where N = number of vCPUs. The pipeline is CPU-bound; oversubscribing past the physical core count gives no speedup and hurts memory.
- Each worker holds ~300 MB resident. With 16 workers expect ~5 GB RAM.

---

## 7. Monitor

```bash
# attach to the running tmux session
tmux attach -t repro

# Or poll from outside:
python3 - <<'EOF'
import json, os, time
from collections import Counter
recs = [json.loads(l) for l in open("/tmp/repro_log.jsonl")]
c = Counter(r["status"] for r in recs)
mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime("/tmp/repro_log.jsonl")))
print(f"{len(recs)} processed | {dict(c)} | last entry: {mtime}")
EOF
```

The script writes one JSON line per coin to `--log`. `--resume` reads
this log on restart and skips coin_ids whose status is `ok`, so it's
safe to kill + relaunch.

---

## 8. After completion

Sanity-check a sample by re-downloading the new transparents from Spaces:

```bash
# Pick 10 random OK coins and verify alpha coverage
python3 - <<'EOF'
import json, os, random
import boto3, numpy as np
from PIL import Image
from io import BytesIO
s3 = boto3.client("s3",
    region_name=os.environ["SPACES_REGION"],
    endpoint_url=os.environ["SPACES_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
recs = [json.loads(l) for l in open("/tmp/repro_log.jsonl") if json.loads(l)["status"]=="ok"]
sample = random.sample(recs, min(10, len(recs)))
for r in sample:
    for key in r.get("uploaded", []):
        if not key.endswith("_transparent.png"): continue
        obj = s3.get_object(Bucket=os.environ["SPACES_BUCKET"], Key=key)
        alpha = np.asarray(Image.open(BytesIO(obj["Body"].read())).convert("RGBA").split()[3])
        pct = (alpha>200).mean()*100
        print(f"coin {r['coin_id']:>7}  alpha={pct:.1f}%  {key}")
EOF
```

Expect alpha 60-78% for clean coins.

---

## 9. Teardown

```bash
# Save the log back to primary for the audit trail
scp /tmp/repro_log.jsonl root@<primary IP>:/tmp/v1v2/

# Destroy droplet
doctl compute droplet delete vision-cpu-burst
# or in RunPod dashboard: terminate pod
```

If you used SSH-tunneled MySQL, also `pkill -f "ssh.*3306:"` on the
burst droplet to clean up.

---

## Throughput reference

From benchmarks on the primary 4-core droplet (May 2026):

| state | rate | source |
|---|---|---|
| Pre-Pattern-B fix | 11 coins/min | early non-GREEN remainder batch |
| Post-Pattern-B (rim recovery on every low-circ) | 3.5 coins/min | 322 batch BEFORE |
| Post-perf-opts (geo-conf gate + area-ratio gate) | 6.25 coins/min | 322 batch AFTER, 1.79× speedup |
| 19K mixed batch with perf opts | 4.77 coins/min (283/hour) | sustained avg |

Extrapolating to bigger CPU instances (linear with cores):

| vCPUs | est. coins/hour | 1000-coin auction | 19K batch |
|---|---|---|---|
| 4 (primary) | 280 | 3.5 h | 67 h |
| 16 | 1,100 | 1 h | 17 h |
| 32 | 2,200 | 0.5 h | 9 h |

I/O to Spaces is not the bottleneck up to ~30 workers — pipeline CPU
work (Hough rim recovery + L1 segmentation) dominates per-coin time.

---

## Code references

- `tools/reprocess_hough.py` — main entrypoint (`fetch_jobs`, `reprocess_one`,
  worker-pool driver, --resume logic, --coin-ids-file, --no-hough-filter).
- `src/pipeline_manager.py::analyze_image` — full vision pipeline call.
- `src/layer1_geometry.py` — Layer 1 segmentation; the rim-recovery
  trigger gate (commit `f52e963`) lives here at line ~195.
- `src/rim_logic.py::recover_rim` — geo + Hough rim recovery
  (commits `f62fa1b`, `c87b17f`, `d284286`, `ce4e0bc` for the
  `TRIVALAYA_PERF_LOG` timing).
- `src/two_coin_resolver.py::_vectorized_hough` — fixed two-coin
  splitter (commit `f0a77d0`).
