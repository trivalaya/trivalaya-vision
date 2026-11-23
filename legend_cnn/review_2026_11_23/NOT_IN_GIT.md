# Files of the legend-ribbon experiment that are not in git (2026-11-23)

They exist only on the machine that holds this checkout. Sizes in bytes. Listed by `git ls-files -o -i --exclude-standard legend_cnn`.

| Where | Files | Bytes | What |
|---|---|---|---|
| `checkpoints/` | 3 | 18,870,267 | model weights of three training runs |
| `data/ (ribbon gallery images)` | 11 | 35,994,912 | ribbon gallery images made while writing the unwrap |
| `runs/run_20260410_001901_baseline/galleries/` | 5 | 5,733,164 | review galleries (.png) |
| `runs/run_20260410_001901_baseline/previews/` | 30 | 569,813 | ribbon previews (.png) |
| `runs/run_20260410_002211_baseline/galleries/` | 5 | 5,815,675 | review galleries (.png) |
| `runs/run_20260410_002707_baseline/galleries/` | 6 | 8,233,956 | review galleries (.png) |
| `runs/run_20260410_002707_baseline/previews/` | 300 | 5,636,701 | ribbon previews (.png) |
| `runs/run_20260410_004248_baseline/galleries/` | 5 | 7,497,596 | review galleries (.png) |
| `runs/run_20260410_005128_cm_moments/galleries/` | 6 | 8,858,079 | review galleries (.png) |
| `runs/run_20260410_005146_cm_min_enclosing_circle/galleries/` | 6 | 8,560,757 | review galleries (.png) |
| `runs/run_20260410_005221_sweep_cen_bbox_inn068_out088/galleries/` | 6 | 8,867,317 | review galleries (.png) |
| `runs/run_20260410_005239_sweep_cen_bbox_inn068_out090/galleries/` | 6 | 9,229,330 | review galleries (.png) |
| `runs/run_20260410_005255_sweep_cen_bbox_inn068_out092/galleries/` | 6 | 9,420,066 | review galleries (.png) |
| `runs/run_20260410_005311_sweep_cen_bbox_inn072_out088/galleries/` | 6 | 8,641,900 | review galleries (.png) |
| `runs/run_20260410_005328_sweep_cen_bbox_inn072_out090/galleries/` | 6 | 8,255,286 | review galleries (.png) |
| `runs/run_20260410_005346_sweep_cen_bbox_inn072_out092/galleries/` | 6 | 8,812,804 | review galleries (.png) |
| `runs/run_20260410_005401_sweep_cen_bbox_inn076_out088/galleries/` | 6 | 8,340,757 | review galleries (.png) |
| `runs/run_20260410_005416_sweep_cen_bbox_inn076_out090/galleries/` | 6 | 8,399,986 | review galleries (.png) |
| `runs/run_20260410_005431_sweep_cen_bbox_inn076_out092/galleries/` | 6 | 8,589,148 | review galleries (.png) |
| `runs/run_20260410_010158_offset_150/galleries/` | 6 | 9,331,480 | review galleries (.png) |
| `runs/run_20260410_010218_offset_175/galleries/` | 6 | 8,959,216 | review galleries (.png) |
| `runs/run_20260410_010238_offset_200/galleries/` | 6 | 8,587,379 | review galleries (.png) |
| `runs/run_20260410_010258_offset_225/galleries/` | 6 | 8,469,724 | review galleries (.png) |
| `runs/run_20260410_010318_offset_200_moments/galleries/` | 6 | 8,725,488 | review galleries (.png) |
| `runs/run_20260410_013529_promoted_v1/galleries/` | 5 | 7,004,089 | review galleries (.png) |
| `runs/run_20260410_013529_promoted_v1/ribbons/` | 6232 | 817,638,400 | ribbon arrays (.npy), rebuilt by `unwrap.py` from the coin list; cut with the fault REPORT.md describes |
| `runs/run_20260410_161758_promoted_v1_h128/galleries/` | 5 | 7,713,389 | review galleries (.png) |
| `runs/run_20260410_161829_promoted_v1_h128/galleries/` | 5 | 7,062,327 | review galleries (.png) |

Total: 6708 files, 1,067,819,006 bytes.

## Checksums of the model weights (sha256)

- `checkpoints/best.pt`: `f68958d36dd4b460afd9fbbfb0ac3fd947b39c7d600bfe4cc81dc2c2c787de12` (6,290,089 bytes)
- `checkpoints/best_run2_lr3e4_bs64.pt`: `995c9fe4fb8758c4da9623eef0c69545984b6cceadd9a560be984bf9019034c1` (6,290,089 bytes)
- `checkpoints/best_run3_filtered.pt`: `8143f0790879bb42d3e17465664a15f6b76294ffec45dbbe93bff5ef76e83a84` (6,290,089 bytes)

`best.pt` and the two named files are the checkpoints of the April training runs; none was evaluated on the test set.
