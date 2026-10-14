# deVision v0.3 adoption decision

Recorded: 2026-10-14 22:49:39 +0800

The project owner explicitly chose r15-vision to replace v12-ext and authorized publication as model v0.3. This is an owner-approved trade-off. The prewritten acceptance rule was not passed: the original `candidate=false` is preserved in [accept_vs_v0.2.json](accept_vs_v0.2.json). The counting condition passed; four old-capability protection conditions did not pass; no prewritten NLL warning was triggered.

| Set | v0.2 | v0.3 | Difference, points [95% CI] |
|---|---:|---:|---:|
| bench_countbench | 18.9% | 35.6% | +16.7 [+12.8, +21.0] |
| test_relation | 71.1% | 75.5% | +4.4 [+2.8, +5.9] |
| test_relation_flip | 66.9% | 69.8% | +2.9 [+1.0, +4.8] |
| bench_lv_scienceqa | 76.6% | 78.9% | +2.3 [+1.1, +3.5] |
| test_vsr | 67.9% | 66.2% | -1.8 [-3.9, +0.3] |
| bench_lv_aokvqa | 62.6% | 61.2% | -1.3 [-2.9, +0.3] |
| test_vqa_yesno | 71.0% | 70.4% | -0.6 [-2.2, +1.1] |
| test_position_flip | 85.0% | 84.7% | -0.3 [-2.2, +1.4] |

Intervals use paired resampling by picture. These evaluation sets have informed multiple experiments; the intervals do not correct for adaptive selection or multiple comparisons. A confidence interval containing zero does not establish equivalence.

Accepted limitations:

- VSR positive recall fell from 72.6% to 67.3% (post-hoc diagnostic); A-OKVQA lost 15 net correct answers, with worse NLL and ECE.
- CountBench accuracy is still only 35.6%. Reversing answer order changes the selected answer on 39.5% of questions, versus 16.9% for v0.2; reversed-order accuracy also improved, so the gain is not confined to the original ordering.
- What'sUp NLL increased by 0.0186 [0.0051, 0.0314], below the prewritten 0.02 warning threshold but still a deterioration in probability quality.
- Diagram reasoning, small text and general spatial binding remain limited. There is no completed frozen-vision control isolating the effect of unfreezing.

The final checkpoint is `runs/r15-vision/stage2`, step 10,235, seed 1. It continues exp19 at step 2,000 with the same optimizer/schedule state; see [seeded_from.json](seeded_from.json) and [training.yaml](training.yaml). Calibration is unchanged from the completed run; publication does not train or refit temperatures.
