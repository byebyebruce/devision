## training

This release continues v0.4 in one pass on one RTX 5070 (bf16), with SigLIP2's top four encoder layers and final LayerNorm fine-tuned at 5e-6 alongside the projector, decision head and ModernBERT LoRA (merged into the released weights). The input remains 256×256 with 64 visual tokens; the architecture is unchanged.

**268,583 questions, one pass from v0.4**, aimed at everyday uses v0.4 could not handle:
- phone screenshots (Android in the Wild: general and shopping tasks, with the screen state as the question) and web pages (Mind2Web: which marked element to act on), ScreenQA;
- first-person everyday scenes (EPIC-KITCHENS VISOR: what is being held, touched or done) and driving scenes (DriveLM, front camera);
- charts (ChartQA, distractors from the same chart), paper figures (ArxivQA), diagrams and science (AI2D, TQA, ScienceQA, IconQA, CLEVR);
- receipts and forms (CORD, FUNSD: which value a field holds), image–caption matching with hard negatives (COCO captions), meme moderation (Hateful Memes train), relative depth (COCO indoor photos labelled by a monocular depth model, kept only where the depth gap is large);
- replay of v0.4's natural-photo abilities (VQAv2, GQA, presence, size, position, relations, counting, Visual7W, VSR, A-OKVQA), whose targets were set to 0.5 × gold + 0.5 × v0.4's own answer so that the old abilities stay where they were.

Every picture of every evaluation set was excluded by id and image hash, including the pictures of the external benchmarks we report against (145 training rows were removed by this filter).

Temperatures were fitted at the end of training on a 1,362-question development set (charts, yes/no, presence, VQA and GQA choice), by question type and option-count bucket:

{{temperature_table}}

Against v0.4 on the same questions (95% intervals from resampling pictures; full table in accept_vs_v0.4.md):

| set | v0.4 | v0.5 | difference |
|---|---|---|---|
| phone shopping screens (AitW, state questions) | 21.4% | 68.5% | +47.0 [+42.8, +51.2] |
| phone / web screens (AitW, Mind2Web) | 38.6% | 78.2% | +39.7 [+36.8, +42.7] |
| phone screen state | 25.3% | 52.4% | +27.1 [+24.0, +30.1] |
| first-person everyday scenes (EPIC-KITCHENS) | 34.7% | 71.7% | +37.0 [+35.4, +38.7] |
| charts (ChartQA) | 30.7% | 58.9% | +28.2 [+25.7, +30.9] |
| CV-Bench | 55.1% | 60.8% | +5.7 [+4.0, +7.3] |
| What'sUp | 46.7% | 51.8% | +5.1 [+2.7, +7.6] |
| POPE | 86.5% | 86.1% | −0.5 [−1.1, +0.1] |

On held-out pictures of the new training sources: paper figures 32.0% → 51.5%, receipts / forms 39.3% → 53.4%, driving scenes 43.5% → 75.7%, caption matching 85.9% → 94.4%, memes 50.8% → 60.9%, relative depth 50.4% → 55.4% (all intervals above 0).

No other evaluation set (MMBench, MME, NaturalBench, RealWorldQA, CountBench, EgoThink, CoGenT, the laya-vision sets, our presence / position / relation / size / VSR / Visual7W / VQA tests) is significantly worse.

## limitations

- **Scope.** English, one image, yes/no and choice questions only; `score` is not implemented. At 256×256, small text (dense documents, infographics, desktop web pages) is hard to read.
- **POPE.** About 0.5 points below v0.4 on POPE (86.1% vs 86.5%, interval [−1.1, +0.1]) (object-presence hallucination); the owner accepted this small regression for the new abilities.
- **Web pages.** Our held-out phone and web screens improved strongly, but choosing the marked element to act on in dense desktop web pages stays near chance.
- **Counting and science** did not improve over v0.4 (CountBench 51.5% vs 50.1%, COCO counting 35.0% vs 34.5%, ScienceQA 82.1% vs 82.1%, none significant).
- **Calibration.** NLL is lower than or equal to v0.4's on 27 of 30 sets (strongly on the new uses), but the model is more over-confident on NaturalBench (NLL +0.029, significant); ECE is also higher on RealWorldQA, MMBench, NaturalBench and a few others (by 0.003–0.029). Refitting the temperatures on the development sets alone narrows but does not remove this.
- **Labels.** The relative-depth questions use a depth model's predictions, not measured depth; the memes source has its own research licence. Several sources are non-commercial; see the dataset list.
- **Test-set reuse.** Test sets informed several decisions in this cycle. This is one training seed.

- **Release decision and original goals.** The owner approved r24 as v0.5 on 2026-10-30 with the calibration trade-offs above. This release does not establish official ImageJevBench v0.3.0 admission, sub-1B leadership, or improvement across its five official domains. The five project-use evaluations above are a different protocol; no official rank is claimed.
