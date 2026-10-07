## training
deVision is trained in two stages. Stage 1 aligns the projector on COCO captions (image-conditioned masked words). Stage 2 trains the decision on yes / no and multiple-choice questions with Laya's RLCD objective (projector, decision head and a LoRA on ModernBERT; the LoRA is merged for release). The training lineage runs over eleven rounds (COCO-derived existence / position / size questions, VQAv2, GQA, A-OKVQA, ScienceQA, AI2D, TQA, VSR, Visual7W telling, mirrored left/right pairs, counting). Round `v11-v9b-recovery` continued from the previous best for one pass over 9,988 distinct questions (3,000 counting, 4,500 replay of earlier abilities, 2,488 left/right), 1,249 steps on Apple MPS in FP32, learning rates `5e-5` (projector, head) and `1e-4` (LoRA), 125 warm-up steps. Full configurations and results: the [training log](https://github.com/byebyebruce/devision/blob/main/docs/training-log.md).

## limitations
- **Scientific diagrams remain difficult.** On the 323 ScienceQA natural-science questions that need the picture, this release is about 23 points behind Laya Vision (see the comparison table). Comparing two named regions of a diagram (two magnet poles, two series of a chart) is the main open weakness.
- **Spatial reasoning is incomplete.** A left/right question is answered right on both a picture and its mirror for about 61% of position pairs and 39% of relative-position choice pairs, but only 10% of relative-position yes / no pairs.
- **Counting** on pictures never used in training (the fresh 600-question test) is no better than the previous release.
- **Small text and fine detail** are limited by the 256 × 256 input; this is not an OCR model.
- **English, one image, `noul` and `choice` only.** Other languages, several images and `score` questions are not supported.
- **Probabilities can be miscalibrated out of domain.** A low ECE on these benchmarks does not guarantee reliable confidence on your data.
