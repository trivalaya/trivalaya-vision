#!/usr/bin/env bash
# Serialize the M1 A/B arms: wait out the running OFF arm, then run ON.
# One heavy process at a time -- the box is 4 vCPU and hosts the live
# visual_search service. Ticket: specs/background_estimator_repair.md.
set -u
WT=/home/claudeuser/vision-wt-m1
PY=/home/claudeuser/trivalaya-pipeline/.venv/bin/python
OUT="$WT/specs/results/m1_ab"

# 1. Wait for the OFF arm (already running) to land.
while pgrep -f "bg_estimator_m1_ab.py --gate off" >/dev/null; do sleep 20; done

if ! grep -q "^done ->" "$OUT/ab_off.log"; then
  echo "M1_AB_FAIL off arm exited without completing; see ab_off.log"
  exit 1
fi
echo "M1_AB_OFF_DONE $(grep -c . "$OUT/ab_off.jsonl") rows"

# 2. ON arm, same shape, same worker count.
cd "$WT" || exit 1
"$PY" -u tools/bg_estimator_m1_ab.py --gate on --all \
      --out "$OUT/ab_on.jsonl" --workers 3 > "$OUT/ab_on.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || ! grep -q "^done ->" "$OUT/ab_on.log"; then
  echo "M1_AB_FAIL on arm rc=$rc; see ab_on.log"
  exit 1
fi
echo "M1_AB_COMPLETE off=$(grep -c . "$OUT/ab_off.jsonl") on=$(grep -c . "$OUT/ab_on.jsonl") rows"
