#!/usr/bin/env python3
"""Adjudicate the M1 flip sweep against §7.1-as-stamped (three-part expectation).

  1. The SIX heal        — masked sha MUST change on their crossing sides;
                           top-1 (material + stable_key) MUST NOT move.
                           Cosine may move either direction.
  2. The THREE hold still — 121 obv+rev, 122 rev: byte-identical, top-1 and
                           cosine identical. M1 does not reach them.
  3. Everything else byte-still — identical masked sha both sides, identical
                           top-1, identical cosine.

Exit non-zero on any REAL FAILURE.
Usage: m1_flip_adjudicate.py <off.json> <on.json>
"""
from __future__ import annotations

import json
import sys

SIX = {
    "01_geta_caesar_denarius": ["obv", "rev"],
    "23_athenian_owl_new_style": ["obv", "rev"],
    "86_pergamon_cistophoric": ["obv", "rev"],
    "255_cyprus_kition_herakles": ["obv", "rev"],
    "214_macedon_demetrios_poliorketes": ["rev"],
    "235_hk_mithradates_vi": ["obv"],
}
HOLD = {
    "121_late_solidus_standing": ["obv", "rev"],
    "122_late_solidus_captive_trophy": ["rev"],
}


def load(p):
    d = json.load(open(p))
    return d, {r["fixture"]: r for r in d["rows"]}


def main():
    off_d, off = load(sys.argv[1])
    on_d, on = load(sys.argv[2])
    print(f"OFF arm={off_d['arm']}  lane={off_d.get('l1_lane')}")
    print(f"ON  arm={on_d['arm']}  lane={on_d.get('l1_lane')}")
    assert off_d["arm"] == "off" and on_d["arm"] == "on", "arms mislabeled"
    if off_d.get("l1_lane") != on_d.get("l1_lane"):
        print("!! L1 lane differs between arms — A/B is not controlled")
        return 2

    failures, findings, expected = [], [], []
    common = sorted(set(off) & set(on))
    only_off, only_on = sorted(set(off) - set(on)), sorted(set(on) - set(off))
    for f in only_off + only_on:
        failures.append(f"fixture set differs between arms: {f}")

    for fx in common:
        a, b = off[fx], on[fx]
        moved_sides = [s for s in ("obv", "rev")
                       if a["sides"][s]["sha_masked"] != b["sides"][s]["sha_masked"]]
        t_a, t_b = a["top3"][0], b["top3"][0]
        card_moved = (t_a["material"], t_a["stable_key"]) != (t_b["material"], t_b["stable_key"])
        cos_moved = t_a["cos"] != t_b["cos"]

        if fx in SIX:
            want = SIX[fx]
            if sorted(moved_sides) != sorted(want):
                findings.append(
                    f"{fx}: expected sides {want} to heal, sha moved on {moved_sides or 'NOTHING'}"
                    " — clause 4, flip did not take where measured")
            if card_moved:
                failures.append(
                    f"{fx}: TOP-1 MOVED {t_a['material']}|{t_a['stable_key']} -> "
                    f"{t_b['material']}|{t_b['stable_key']} — clause 3, REAL FAILURE")
            else:
                for s in want:
                    ma, mb = a["sides"][s], b["sides"][s]
                    expected.append(
                        f"{fx} [{s}] noop {ma['mask_noop']}->{mb['mask_noop']} "
                        f"area {ma['mask_area_fraction']}->{mb['mask_area_fraction']}")
                expected.append(f"{fx} cos {t_a['cos']} -> {t_b['cos']} (card held)")
        elif fx in HOLD:
            if moved_sides or card_moved or cos_moved:
                failures.append(
                    f"{fx}: HOLD-STILL fixture moved (sides={moved_sides} card={card_moved} "
                    f"cos={t_a['cos']}->{t_b['cos']}) — stamp clause 2, REAL FAILURE")
            for s in HOLD[fx]:
                mb = b["sides"][s]
                if not mb["mask_noop"]:
                    findings.append(
                        f"{fx} [{s}]: expected to REMAIN a no-op, now mask_noop=False "
                        f"(area {mb['mask_area_fraction']}) — M1 reached further than measured")
        else:
            if moved_sides or card_moved or cos_moved:
                failures.append(
                    f"{fx}: DRIFT OUTSIDE THE SIX (sides={moved_sides} card={card_moved} "
                    f"cos={t_a['cos']}->{t_b['cos']}) — clause 2, REAL FAILURE")

    # grading-level regression: the standing bar's own numbers must not move
    def grade_counts(d):
        g = [r for r in d["rows"] if r["grade"]["state"] == "graded" and r["grade"]["checks"]]
        return len(g), sum(1 for r in g if not r["grade"]["fails"]), sum(1 for r in g if r["grade"]["fails"])

    ga, gb = grade_counts(off_d), grade_counts(on_d)
    print(f"\nexpected.yaml grading  OFF graded/clean/mismatch = {ga}")
    print(f"                       ON  graded/clean/mismatch = {gb}"
          f"   {'IDENTICAL' if ga == gb else '*** MOVED ***'}")
    if ga != gb:
        failures.append(f"expected.yaml grading moved {ga} -> {gb}")

    noop_off = sum(1 for r in off_d["rows"] for s in ("obv", "rev") if r["sides"][s]["mask_noop"])
    noop_on = sum(1 for r in on_d["rows"] for s in ("obv", "rev") if r["sides"][s]["mask_noop"])
    print(f"no-op sides            OFF {noop_off}  ->  ON {noop_on}"
          f"   (healed {noop_off - noop_on}, residual {noop_on})")

    print(f"\n--- EXPECTED CHANGE ({len(expected)} lines, the six healing) ---")
    for e in expected:
        print(f"  {e}")
    print(f"\n--- FINDINGS ({len(findings)}) ---")
    for f in findings:
        print(f"  {f}")
    print(f"\n--- REAL FAILURES ({len(failures)}) ---")
    for f in failures:
        print(f"  {f}")

    n_still = sum(1 for fx in common if fx not in SIX and fx not in HOLD)
    print(f"\nfixtures required byte-still: {n_still}   verdict: "
          f"{'PASS' if not failures else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
