<!-- DRAFT 2026-10-29: numbers marked TODO are filled from runs/r23-v05/eval after the r23 judgement; nothing here is a result yet. -->
## training

This release continues v0.4 in one pass on one RTX 5070 (bf16), with SigLIP2's top four encoder layers and final LayerNorm fine-tuned at 5e-6 alongside the projector, decision head and ModernBERT LoRA (merged into the released weights). The input remains 256×256 with 64 visual tokens; the architecture is unchanged.

**205,166 questions, one pass from v0.4**, aimed at everyday uses v0.4 could not handle:
- phone screenshots (Android in the Wild: general and shopping tasks, with the screen state as the question) and web pages (Mind2Web: which marked element to act on), ScreenQA;
- first-person everyday scenes (EPIC-KITCHENS VISOR: what is being held, touched or done);
- charts (ChartQA, distractors from the same chart), diagrams and science (AI2D, TQA, ScienceQA, IconQA, CLEVR);
- replay of v0.4's natural-photo abilities (VQAv2, GQA, presence, size, position, relations, counting, Visual7W, VSR, A-OKVQA), whose targets were set to 0.5 × gold + 0.5 × v0.4's own answer so that the old abilities stay where they were.

Every picture of every evaluation set was excluded by id and image hash.

Temperatures were fitted on a development mix with balanced sources, by question type and option-count bucket:

{{temperature_table}}

TODO: comparison table against v0.4 (accept_vs_v0.4.md).

## limitations

- **Scope.** English, one image, yes/no and choice questions only; `score` is not implemented. At 256×256, small text (receipts, forms, dense web pages) is hard to read.
- **POPE.** About TODO points below v0.4 on POPE (object-presence hallucination); the owner accepted this small regression for the new abilities.
- **Web pages.** Mind2Web element choice improved but remains far from reliable (TODO).
- **Counting and science** did not improve over v0.4 (TODO numbers).
- **Calibration.** TODO.
- **Test-set reuse.** Test sets informed several decisions in this cycle. This is one training seed.
