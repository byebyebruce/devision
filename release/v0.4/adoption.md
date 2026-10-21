# v0.4 adoption

On 2026-10-22 the project owner decided to publish `r21-soft` as v0.4 ("发布这一版"), choosing it over the alternative candidate (r20 with temperatures refitted on the balanced dev set), although the prewritten acceptance rule 4 (calibration) is not met. The rules were frozen before training in `docs/experiments/2026-10-21-v04-eval-freeze.md`; the full verdict is in [accept_vs_v0.3.md](accept_vs_v0.3.md).

| Rule (vs v0.3, paired by question, pictures resampled) | Result |
|---|---|
| 1. no judged set significantly worse (accuracy) | met (0 of 23) |
| 2. at least 15 of 23 sets improve | met (22 of 23) |
| 3. at least 4 of 5 weak abilities improve with lower bound > 0 | met (ScienceQA, VSR, counting, V7W; A-OKVQA +2.1 with an interval containing 0) |
| 4. no set's NLL rises by > 0.02 with lower bound > 0 | **not met**: size +0.156, MME +0.037, COCO object presence +0.026 |

The rule-4 failures come mostly from under-confidence: size answers average 0.64 confidence at 0.87 accuracy, presence 0.88 at 0.955 (the presence / size training labels were softened to 0.9 / 0.1 to fix POPE over-confidence, which worked: POPE NLL 0.363 vs v0.3's 0.372). MME is slightly over-confident. One temperature per question type and option count cannot fit all of these at once.

Test sets informed several decisions in this release cycle (a stop at step 22,000, the science continuation, the label softening and the choice between two candidates). This is one training seed.
