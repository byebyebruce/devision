## training

This release continues v0.2 with 81,875 questions of counting, object presence, mirrored spatial questions, VSR and broad replay, for one 10,235-step schedule (seed 1). The first 2,000 steps were run as exp19; r15 resumes that exact training state. SigLIP2's top four encoder layers and final LayerNorm were fine-tuned at 5e-6 alongside the projector, decision head and ModernBERT LoRA. LoRA is merged into the released weights. The input remains 256×256 with 64 visual tokens; S2 question conditioning is **not** enabled.

Temperatures were fitted on the designated calibration set, by question type and option-count bucket. Missing buckets fall back to the question-type temperature:

{{temperature_table}}

The owner adopted v0.3 as a trade-off: counting, relations and ScienceQA improved, while four old-capability protection conditions were not met. The original automated verdict remains `candidate=false`. See the [adoption decision and v0.2 comparison](evaluation/adoption.md), [original acceptance report](evaluation/accept_vs_v0.2.json) and [training configuration](evaluation/training.yaml).

## limitations

- English, one image, yes/no and choice questions only. `score` is not implemented. At 256×256, small text and fine diagram details are hard to read.
- CountBench improved by 16.7 points over v0.2, but absolute accuracy is only 35.6%. Answer-order sensitivity increased (39.5% change their answer when options are reversed, versus 16.9%); this is not a general-purpose accurate counter.
- VSR accuracy is 1.8 points lower and A-OKVQA 1.3 points lower than v0.2; their overall intervals contain zero, but non-inferiority was not established. VSR positive recall fell by 5.3 points. A-OKVQA and What'sUp have worse probability quality than v0.2. No prewritten NLL warning was triggered; that does not mean all probabilities improved.
- Diagram, chart and map comparisons and question-to-region binding remain weak. On natural-science ScienceQA questions that require the picture, accuracy is 49.5% versus Laya Vision's 70.0%.
- Calibration reflects the calibration distribution, not every deployment domain. Validate thresholds on your own examples.
- Test sets have repeatedly informed experiment and release decisions. This is one training seed, not a fresh independent confirmation. Gains cannot be attributed solely to unfreezing the vision encoder.
