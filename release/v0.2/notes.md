## training
Training: the projector is first aligned on COCO captions, then the decision is trained with Laya's RLCD objective on yes/no and multiple-choice questions (projector, decision head and a LoRA on ModernBERT), round after round. This release continued for one pass over 353,830 questions, 315,342 of them from thirteen public datasets (Objects365, TallyQA, CLEVR, CLEVR-Math, Super-CLEVR, FigureQA, MapQA, IconQA, SNLI-VE, Vision-Flan, VisOnlyQA, SpatialSense, PixMo-Count) and the rest replaying earlier data (VQAv2, GQA, A-OKVQA, ScienceQA, VSR, Visual7W, COCO-derived questions). Temperatures are fitted per question type and option count. Details: the [training log](https://github.com/byebyebruce/devision/blob/master/docs/training-log.md).

## limitations
- **Comparing two places named in the question** (two magnet poles, two chart series, two map regions) stays near chance; on the ScienceQA questions that need the picture it is about 24 points behind Laya Vision.
- **Relative-position yes/no questions** are weak: right on both a picture and its mirror for only 19% of pairs (65% for relative-position choice questions).
- **Presence leans towards "yes"**: it says an absent object is there more often than the previous release (POPE adversarial 0.791).
- English only, one image, `noul` and `choice` only; 256 × 256 input, so no small text.
- Calibration was fitted on this project's data and may not hold on yours.
