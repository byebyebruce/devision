## training

This release continues v0.3 in three steps on one RTX 5070 (bf16), with SigLIP2's top four encoder layers and final LayerNorm fine-tuned at 5e-6 alongside the projector, decision head and ModernBERT LoRA (merged into the released weights). The input remains 256×256 with 64 visual tokens; the architecture is unchanged.

1. **742,238 questions, one pass from v0.3.** VQAv2 (annotators' answers as soft targets) and GQA (balanced train, all shards, open questions turned into choices), about 190,000 each, plus targeted data for the abilities v0.3 lacked:
   - all of Visual7W telling (52,390 after filtering);
   - caption-choice spatial relations (on / under / in front of / behind / left / right) from Visual Genome scene graphs, with every distractor checked against the boxes;
   - counting on COCO and TallyQA, always with the eleven options 0–10;
   - CLEVR, IconQA;
   - programmatically drawn magnet and solution diagrams in ScienceQA's style;
   - replay of v0.3's other abilities.

   Every picture of every evaluation set was excluded by id and image hash. Choice questions v0.3 still answered confidently with the wrong picture were dropped.
2. **A short science-heavy continuation (123,084 questions):** all ScienceQA / AI2D / TQA training questions ×3, IconQA, the magnet diagrams and 40,000 replay. This also repaired an exclusion bug that had dropped about half of the ScienceQA training questions in step 1.
3. **A short continuation (104,331 questions) with presence and size labels softened from 0 / 1 to 0.9 / 0.1,** to stop over-confident presence answers (POPE).

Temperatures were fitted on a development mix with balanced sources, by question type and option-count bucket. Missing buckets fall back to the question-type temperature:

{{temperature_table}}

The owner adopted this model although one prewritten acceptance rule (calibration) is not met. See the [adoption decision](evaluation/adoption.md), the [full comparison with v0.3](evaluation/accept_vs_v0.3.md) and the [training configurations](evaluation/training.yaml).

Against v0.3 on the same questions (accuracy points, 95% intervals), 22 of 23 judged sets improve and none is significantly worse:

| Set | Change |
|---|---|
| mirrored relations | +21.2 |
| What'sUp | +18.0 |
| CountBench | +14.5 |
| Visual7W | +7.2 |
| VSR | +7.0 |
| GQA | +6.1 |
| CLEVR-CoGenT B | +4.0 |
| ScienceQA | +3.1 [+1.6, +4.8] |
| MMBench-en dev | +2.3 |
| POPE | +1.7 |

## limitations

- **Scope.** English, one image, yes/no and choice questions only; `score` is not implemented. At 256×256, small text and fine diagram details are hard to read.
- **Counting.** It improved (CountBench 35.6% → 50.1%, COCO counts 30.6% → 34.5%), but this is still not an accurate counter, especially above five.
- **Calibration.** Probabilities are slightly under-confident on easy presence and size questions (size: mean confidence 0.64 at 0.87 accuracy) and slightly over-confident on MME. POPE calibration is on par with v0.3. Validate thresholds on your own examples.
- **External benchmarks.** MME (−0.6), NaturalBench (+0.3), CV-Bench (+0.7) and A-OKVQA (+2.1) did not change significantly.
- **Diagrams.** Diagram reading that depends on small printed numbers (particle speeds, magnet distances) remains weak. The drawn solution diagrams did not transfer to ScienceQA's beakers.
- **Test-set reuse.** Test sets informed several decisions in this cycle. This is one training seed, not an independent confirmation.
