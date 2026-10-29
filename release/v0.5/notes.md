<!-- DRAFT 2026-10-29 14:30: numbers from runs/r23-v05/eval (release/v0.5/accept_vs_v0.4.md). r23 passed plan rules 1-2; rule 3 (calibration) not fully met - awaiting the owner's decision, not published. -->
## training

This release continues v0.4 in one pass on one RTX 5070 (bf16), with SigLIP2's top four encoder layers and final LayerNorm fine-tuned at 5e-6 alongside the projector, decision head and ModernBERT LoRA (merged into the released weights). The input remains 256×256 with 64 visual tokens; the architecture is unchanged.

**205,166 questions, one pass from v0.4**, aimed at everyday uses v0.4 could not handle:
- phone screenshots (Android in the Wild: general and shopping tasks, with the screen state as the question) and web pages (Mind2Web: which marked element to act on), ScreenQA;
- first-person everyday scenes (EPIC-KITCHENS VISOR: what is being held, touched or done);
- charts (ChartQA, distractors from the same chart), diagrams and science (AI2D, TQA, ScienceQA, IconQA, CLEVR);
- replay of v0.4's natural-photo abilities (VQAv2, GQA, presence, size, position, relations, counting, Visual7W, VSR, A-OKVQA), whose targets were set to 0.5 × gold + 0.5 × v0.4's own answer so that the old abilities stay where they were.

Every picture of every evaluation set was excluded by id and image hash.

Temperatures were fitted at the end of training on a 512-question development set (mostly charts), by question type and option-count bucket (yes/no questions were too few there and keep temperature 1.0):

{{temperature_table}}

Against v0.4 on the same questions (95% intervals from resampling pictures; full table in accept_vs_v0.4.md):

| set | v0.4 | v0.5 | difference |
|---|---|---|---|
| phone shopping screens (AitW, state questions) | 21.4% | 71.4% | +50.0 [+45.7, +54.1] |
| phone / web screens (AitW, Mind2Web) | 38.6% | 76.6% | +38.0 [+35.0, +40.8] |
| phone screen state | 25.3% | 52.6% | +27.3 [+24.2, +30.5] |
| first-person everyday scenes (EPIC-KITCHENS) | 34.7% | 71.3% | +36.6 [+35.0, +38.3] |
| charts (ChartQA) | 30.7% | 58.0% | +27.3 [+24.7, +29.9] |
| What'sUp | 46.7% | 49.6% | +2.9 [+0.6, +5.2] |
| POPE | 86.5% | 86.2% | −0.3 [−1.0, +0.2] |

No other evaluation set (MMBench, MME, CV-Bench, NaturalBench, RealWorldQA, CountBench, EgoThink, CoGenT, the laya-vision sets, our presence / position / relation / size / VSR / Visual7W / VQA tests) is significantly worse.

## limitations

- **Scope.** English, one image, yes/no and choice questions only; `score` is not implemented. At 256×256, small text (receipts, forms, dense web pages) is hard to read.
- **POPE.** About 0.3 points below v0.4 on POPE (86.2% vs 86.5%, interval [−1.0, +0.2]) (object-presence hallucination); the owner accepted this small regression for the new abilities.
- **Web pages.** Our held-out phone and web screens improved strongly, but choosing the element to act on in dense desktop web pages stays near chance.
- **Counting and science** did not improve over v0.4 (CountBench 50.3% vs 50.1%, COCO counting 34.7% vs 34.5%, ScienceQA 83.2% vs 82.1%, none significant).
- **Calibration.** NLL and ECE improve on 23 of 33 sets (strongly on the new uses), but the model is more over-confident than v0.4 on harder general benchmarks: NLL +0.015 on CV-Bench, +0.026 on NaturalBench, +0.027 on RealWorldQA (all significant), ECE also up on MMBench and ScienceQA. Refitting the temperatures on the broader development mix used for v0.4 does not remove this.
- **Test-set reuse.** Test sets informed several decisions in this cycle. This is one training seed.
