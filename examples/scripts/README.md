# Harness Scripts

Canonical script entrypoints (in `examples/scripts/finetuning/`):

- `finetune_secret_demo.sh`
  - Baseline vs finetuned `secret` behavior on fixed held-out data.
- `finetune_custom_label_demo.sh`
  - Single custom-label-space adaptation (`custom_secret`).

All demos use tiny toy splits (1 example per split) and higher default epoch
counts to make qualitative before/after behavior obvious.

Common options:

- `--checkpoint <BASE_CHECKPOINT_DIR>` (required)
- `--workdir <ARTIFACT_DIR>` (optional, default is a timestamped `/tmp/...` path)
- `--output-checkpoint-dir <CHECKPOINT_DIR>` (optional, defaults to `<workdir>/finetuned_checkpoint`)
- `--preview-examples <N>` (optional)

## Benchmarking

Canonical script entrypoints (in `examples/scripts/benchmarking/`):

- `benchmark_ai4privacy_opf.py`
  - Loads `ai4privacy/pii-masking-400k` by default.
  - Filters German examples with `--lang-filter de` by default.
  - Converts `source_text` and `privacy_mask` into OPF eval JSONL.
  - Runs `opf eval` and writes metrics/timings/summary.

Install benchmark dependencies first:

```bash
python3 -m pip install -e ".[benchmark]"
```

Example:

```bash
python3 examples/scripts/benchmarking/benchmark_ai4privacy_opf.py \
  --out-dir ./opf_ai4privacy_bench_out \
  --max-eval-examples 5000
```

To evaluate with OPF-compatible labels, use the PII-Masking mapping described in
the OPF model card and switch to typed evaluation:

```bash
python3 examples/scripts/benchmarking/benchmark_ai4privacy_opf.py \
  --label-map opf-model-card \
  --eval-mode typed \
  --out-dir ./opf_ai4privacy_bench_out_typed \
  --max-eval-examples 5000
```

By default, OPF uses `OPF_CHECKPOINT` or `~/.opf/privacy_filter`. Pass
`--checkpoint /real/checkpoint/dir` only when you want to override that.

The default `--source-mode privacy-mask` uses natural text and character
offsets. `--source-mode mbert-token-join` exists only to reproduce older
wordpiece-based HF benchmark inputs and is not recommended for OPF quality
measurement.

If Hugging Face authentication is required, set `HF_TOKEN` or pass
`--hf-token <TOKEN>`. For local parquet exports, use `--validation-parquet`.
