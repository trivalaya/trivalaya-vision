#!/bin/bash
# Probe the §7.1 six (+ the two non-healing no-op fixtures + 2 clean controls)
# through the REAL service, and pull each one's shadow-log record.
# Usage: live_probe_six.sh <out.json>
set -u
OUT="$1"
F=/home/claudeuser/trivalaya-pipeline/visual_search/tests/appv2_regression
LOG=/home/claudeuser/trivalaya-pipeline/logs/identify_shadow.jsonl

FIXTURES=(
  01_geta_caesar_denarius
  23_athenian_owl_new_style
  86_pergamon_cistophoric
  255_cyprus_kition_herakles
  214_macedon_demetrios_poliorketes
  235_hk_mithradates_vi
  121_late_solidus_standing
  122_late_solidus_captive_trophy
  02_domitian_augustus_minerva
  100_sicily_katane_quadriga
)

echo "[" > "$OUT"
first=1
for fx in "${FIXTURES[@]}"; do
  d="$F/$fx"
  obv=$(ls "$d"/obv.* 2>/dev/null | head -1)
  rev=$(ls "$d"/rev.* 2>/dev/null | head -1)
  if [ -z "$obv" ] || [ -z "$rev" ]; then echo "  SKIP $fx (no pair)" >&2; continue; fi
  before=$(wc -l < "$LOG")
  curl -sf -X POST http://127.0.0.1:8081/identify \
       -F "obv=@$obv" -F "rev=@$rev" -o /dev/null || { echo "  FAIL $fx" >&2; continue; }
  # the record this request just appended
  rec=$(tail -1 "$LOG")
  [ $first -eq 0 ] && echo "," >> "$OUT"
  first=0
  python3 -c "
import json,sys
rec=json.loads(sys.stdin.read())
s1=rec.get('stage1') or {}
out={'fixture':'$fx','img':rec.get('img'),'topk':(s1.get('topk') or [])[:3],
     'margin':s1.get('top1_top2_margin')}
print(json.dumps(out,indent=1),end='')
" <<< "$rec" >> "$OUT"
  echo "  probed $fx" >&2
done
echo "" >> "$OUT"
echo "]" >> "$OUT"
echo "-> $OUT" >&2
