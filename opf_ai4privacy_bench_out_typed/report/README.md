# OPF ai4privacy Benchmark Report

This report visualizes the OPF benchmark run in `../summary.json`.

## What Was Tested

- Dataset: `ai4privacy/pii-masking-400k`
- Split/language: `validation` / `de`
- Examples: `5000`
- Input text: `source_text` with spans from `privacy_mask`
- Evaluation mode: `typed`
- Label mapping: `opf-model-card`
- Checkpoint: `/Users/enrue/.opf/privacy_filter`
- Device: `mps`

The ai4privacy source labels were mapped to OPF v2-style labels before typed evaluation. No source labels were dropped in this run (`dropped_source` is empty).

## Important Disclaimer

This is not a perfect reproduction of the OPF model-card benchmark. The model card references PII-Masking-300k and includes dataset corrections/adjudication; this run uses ai4privacy/pii-masking-400k validation/de as available locally. Results can be affected by:

- imperfect mapping from fine-grained ai4privacy labels to broad OPF labels;
- span-boundary differences, for example address parts or first/last names being split differently;
- false positives in OPF that may actually be real PII missing from ai4privacy gold labels;
- false positives or synthetic artifacts in ai4privacy labels;
- using the first 5000 German validation examples rather than a randomized or stratified sample;
- runtime numbers being dominated by eval score stitching on MPS, not pure model inference.

## Generated Charts

- `01_overall_detection_metrics.png`: overall precision/recall/F1 for token-level detection and stricter span-level matching.
- `02_per_class_span_metrics.png`: per-OPF-label span precision/recall/F1.
- `03_mapped_label_distribution.png`: target OPF label distribution after mapping.
- `04_source_label_distribution.png`: original ai4privacy source label counts.
- `05_runtime_breakdown.png`: wall-time breakdown of the OPF eval pipeline.
- `06_precision_recall_by_class.png`: per-class precision vs recall scatterplot.
