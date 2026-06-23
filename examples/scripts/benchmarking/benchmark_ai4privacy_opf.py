#!/usr/bin/env python3
"""Benchmark OPF on the ai4privacy PII masking dataset.

By default, the harness reads natural ``source_text`` plus ``privacy_mask``
character offsets. A legacy mBERT-token-join mode is available only for
comparisons against earlier HF token-classifier experiments that used the same
wordpiece text representation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("opf-ai4privacy-bench")


OPF_MODEL_CARD_LABEL_MAP: dict[str, str] = {
    # PII-Masking-300k mappings from the OPF model card, plus common
    # ai4privacy/pii-masking-400k spelling variants observed in practice.
    "ACCOUNT": "account_number",
    "ACCOUNTNUM": "account_number",
    "BANKNUM": "account_number",
    "BIC": "account_number",
    "CREDITCARD": "account_number",
    "CREDITCARDNUMBER": "account_number",
    "CRYPTOADDRESS": "account_number",
    "DOCNUM": "account_number",
    "DRIVERLICENSE": "account_number",
    "DRIVERLICENSENUM": "account_number",
    "IBAN": "account_number",
    "IDCARD": "account_number",
    "IDCARDNUM": "account_number",
    "PASSPORT": "account_number",
    "SOCIALNUMBER": "account_number",
    "SOCIALNUM": "account_number",
    "TAXNUM": "account_number",
    "BANKMUNICIP": "private_address",
    "BANKPOSTCODE": "private_address",
    "BANKSTREET": "private_address",
    "BUILDING": "private_address",
    "BUILDINGNUM": "private_address",
    "CITY": "private_address",
    "GEOCOORD": "private_address",
    "POSTCODE": "private_address",
    "SECADDRESS": "private_address",
    "STREET": "private_address",
    "ZIPCODE": "private_address",
    "CARDEXPIRY": "private_date",
    "DATE": "private_date",
    "DATEOFBIRTH": "private_date",
    "DOB": "private_date",
    "EMAIL": "private_email",
    "GIVENNAME": "private_person",
    "GIVENNAME1": "private_person",
    "GIVENNAME2": "private_person",
    "LASTNAME1": "private_person",
    "LASTNAME2": "private_person",
    "LASTNAME3": "private_person",
    "SURNAME": "private_person",
    "TITLE": "private_person",
    "USERNAME": "private_person",
    "TEL": "private_phone",
    "TELEPHONENUM": "private_phone",
    "IP": "private_url",
    "OTP": "secret",
    "PASS": "secret",
    "PASSWORD": "secret",
    "PIN": "secret",
}


def human(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}k"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}B"


def normalize_label(label: str) -> str:
    return str(label).strip().upper().replace(" ", "_")


def to_bio(label: str) -> str:
    normalized = normalize_label(label)
    if normalized == "O":
        return "O"
    if normalized.startswith("B-") or normalized.startswith("I-"):
        return normalized
    return f"B-{normalized}"


def tokens_and_labels_to_spans(
    tokens: Sequence[object],
    labels: Sequence[object],
) -> tuple[str, list[dict[str, object]]]:
    """Convert mBERT token labels into OPF-compatible character spans."""
    if len(tokens) != len(labels):
        raise ValueError(
            f"tokens/classes length mismatch ({len(tokens)} != {len(labels)})"
        )

    token_texts = [str(token) for token in tokens]
    parts: list[str] = []
    token_starts: list[int] = []
    pos = 0
    for idx, token in enumerate(token_texts):
        if idx > 0:
            parts.append(" ")
            pos += 1
        token_starts.append(pos)
        parts.append(token)
        pos += len(token)
    text = "".join(parts)

    spans: list[dict[str, object]] = []
    current_label: str | None = None
    current_start_token: int | None = None
    current_end_token: int | None = None

    def flush() -> None:
        nonlocal current_label, current_start_token, current_end_token
        if (
            current_label is None
            or current_start_token is None
            or current_end_token is None
        ):
            return
        start = token_starts[current_start_token]
        end = token_starts[current_end_token] + len(token_texts[current_end_token])
        spans.append(
            {
                "category": current_label,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
        current_label = None
        current_start_token = None
        current_end_token = None

    for idx, raw_label in enumerate(labels):
        bio_label = to_bio(str(raw_label))
        if bio_label == "O":
            flush()
            continue
        prefix, entity_label = bio_label.split("-", 1)
        if prefix == "B":
            flush()
            current_label = entity_label
            current_start_token = idx
            current_end_token = idx
            continue
        if current_label == entity_label and current_end_token is not None:
            current_end_token = idx
            continue

        # Malformed BIO sequence: preserve the entity instead of dropping it.
        flush()
        current_label = entity_label
        current_start_token = idx
        current_end_token = idx

    flush()
    return text, spans


def resolve_label_map(name: str) -> Mapping[str, str] | None:
    if name == "none":
        return None
    if name == "opf-model-card":
        return OPF_MODEL_CARD_LABEL_MAP
    raise ValueError(f"Unsupported label map: {name!r}")


def apply_label_map(
    labels: Sequence[Mapping[str, object]],
    label_map: Mapping[str, str] | None,
    *,
    merge_gap_chars: int = 1,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Map source labels to OPF labels, dropping unmapped labels."""
    stats = {"mapped": 0, "dropped": 0}
    mapped_labels: list[dict[str, object]] = []
    for item in labels:
        source_category = str(item["category"])
        target_category = source_category
        if label_map is not None:
            mapped = label_map.get(normalize_label(source_category))
            if mapped is None:
                stats["dropped"] += 1
                continue
            target_category = mapped
            stats["mapped"] += 1
        mapped_labels.append(
            {
                "category": target_category,
                "start": item["start"],
                "end": item["end"],
            }
        )
    return merge_adjacent_same_label_spans(mapped_labels, gap_chars=merge_gap_chars), stats


def merge_adjacent_same_label_spans(
    labels: Sequence[Mapping[str, object]],
    *,
    gap_chars: int,
) -> list[dict[str, object]]:
    """Merge nearby spans that collapse to the same OPF category."""
    if not labels:
        return []
    sorted_labels = sorted(
        labels,
        key=lambda item: (str(item["category"]), int(item["start"]), int(item["end"])),
    )
    merged: list[dict[str, object]] = []
    for item in sorted_labels:
        category = str(item["category"])
        start = int(item["start"])
        end = int(item["end"])
        if end <= start:
            continue
        if (
            merged
            and str(merged[-1]["category"]) == category
            and start <= int(merged[-1]["end"]) + gap_chars
        ):
            merged[-1]["end"] = max(int(merged[-1]["end"]), end)
            continue
        merged.append({"category": category, "start": start, "end": end})
    merged.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["category"])))
    return merged


def require_sequence(row: Mapping[str, Any], field: str, *, idx: int) -> Sequence[object]:
    value = row.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"row {idx} field {field!r} must be a sequence")
    return value


def spans_from_privacy_mask(
    row: Mapping[str, Any],
    idx: int,
    *,
    text_field: str,
    privacy_mask_field: str,
) -> tuple[str, list[dict[str, object]]]:
    text_raw = row.get(text_field)
    if not isinstance(text_raw, str):
        raise ValueError(f"row {idx} field {text_field!r} must be a string")
    privacy_mask = require_sequence(row, privacy_mask_field, idx=idx)
    labels: list[dict[str, object]] = []
    for mask_idx, item in enumerate(privacy_mask):
        if not isinstance(item, Mapping):
            raise ValueError(f"row {idx} privacy_mask[{mask_idx}] must be an object")
        label = item.get("label")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(label, str) or not label:
            raise ValueError(f"row {idx} privacy_mask[{mask_idx}] missing label")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"row {idx} privacy_mask[{mask_idx}] start must be an int")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"row {idx} privacy_mask[{mask_idx}] end must be an int")
        if not (0 <= start < end <= len(text_raw)):
            raise ValueError(
                f"row {idx} privacy_mask[{mask_idx}] invalid span ({start}, {end})"
            )
        labels.append({"category": label, "start": start, "end": end})
    return text_raw, labels


def row_to_opf_record(
    row: Mapping[str, Any],
    idx: int,
    *,
    dataset_name: str,
    split_name: str,
    source_mode: str,
    text_field: str,
    privacy_mask_field: str,
    tokens_field: str,
    classes_field: str,
    language_field: str,
    label_map: Mapping[str, str] | None,
    label_stats: dict[str, Counter[str]] | None = None,
) -> dict[str, object]:
    if source_mode == "privacy-mask":
        text, labels = spans_from_privacy_mask(
            row,
            idx,
            text_field=text_field,
            privacy_mask_field=privacy_mask_field,
        )
        source_text_mode = "source_text_privacy_mask"
    elif source_mode == "mbert-token-join":
        tokens = require_sequence(row, tokens_field, idx=idx)
        classes = require_sequence(row, classes_field, idx=idx)
        text, labels = tokens_and_labels_to_spans(tokens, classes)
        source_text_mode = "joined_mbert_tokens"
    else:
        raise ValueError(f"Unsupported source mode: {source_mode!r}")
    mapped_labels, label_map_stats = apply_label_map(labels, label_map)
    if label_stats is not None:
        for item in labels:
            source_category = normalize_label(str(item["category"]))
            label_stats["source"][source_category] += 1
            if label_map is not None and source_category not in label_map:
                label_stats["dropped_source"][source_category] += 1
        for item in mapped_labels:
            label_stats["target"][str(item["category"])] += 1
    return {
        "text": text,
        "label": mapped_labels,
        "info": {
            "id": str(row.get("id", idx)),
            "language": row.get(language_field),
            "source_dataset": dataset_name,
            "source_split": split_name,
            "source_text_mode": source_text_mode,
            "source_label_count": len(labels),
            "mapped_label_count": len(mapped_labels),
            "label_map_stats": label_map_stats,
        },
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_source_dataset(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires the optional 'datasets' package. "
            'Install benchmark dependencies with: pip install -e ".[benchmark]"'
        ) from exc

    if args.train_parquet or args.validation_parquet:
        if not args.validation_parquet:
            raise ValueError("--validation-parquet is required when using local parquet")
        data_files: dict[str, str] = {"validation": args.validation_parquet}
        if args.train_parquet:
            data_files["train"] = args.train_parquet
        return load_dataset("parquet", data_files=data_files)

    load_kwargs: dict[str, object] = {}
    if args.hf_token:
        load_kwargs["token"] = args.hf_token
    return load_dataset(args.dataset, **load_kwargs)


def resolve_eval_split(dataset: Any, split_name: str):
    if split_name in dataset:
        return dataset[split_name], split_name
    if split_name == "validation" and "val" in dataset:
        return dataset["val"], "val"
    available = ", ".join(dataset.keys())
    raise ValueError(f"Dataset split {split_name!r} not found. Available: {available}")


def filter_and_limit_split(ds: Any, args: argparse.Namespace):
    if args.lang_filter:
        log.info("Filter %s == %s", args.language_field, args.lang_filter)
        ds = ds.filter(lambda row: row.get(args.language_field) == args.lang_filter)
    if args.max_eval_examples is not None and len(ds) > args.max_eval_examples:
        ds = ds.select(range(args.max_eval_examples))
    return ds


def build_opf_eval_command(
    *,
    dataset_path: Path,
    metrics_path: Path,
    timings_path: Path,
    predictions_path: Path | None,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "opf",
        "eval",
        str(dataset_path),
        "--eval-mode",
        args.eval_mode,
        "--span-metrics-space",
        args.span_metrics_space,
        "--device",
        args.device,
        "--window-batch-size",
        str(args.window_batch_size),
        "--preprocess-workers",
        str(args.preprocess_workers),
        "--metrics-out",
        str(metrics_path),
        "--timings-out",
        str(timings_path),
    ]
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.n_ctx is not None:
        command.extend(["--n-ctx", str(args.n_ctx)])
    if args.decode_mode:
        command.extend(["--decode-mode", args.decode_mode])
    if args.max_opf_examples is not None:
        command.extend(["--max-examples", str(args.max_opf_examples)])
    if args.progress_every is not None:
        command.extend(["--progress-every", str(args.progress_every)])
    if args.discard_overlapping_predicted_spans:
        command.append("--discard-overlapping-predicted-spans")
    if args.discard_overlapping_ground_truth_spans:
        command.append("--discard-overlapping-ground-truth-spans")
    if args.no_trim_whitespace:
        command.append("--no-trim-whitespace")
    if predictions_path is not None:
        command.extend(["--predictions-out", str(predictions_path)])
    return command


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def validate_checkpoint_arg(args: argparse.Namespace) -> None:
    """Fail early when the example placeholder is passed as a checkpoint."""
    if args.prepare_only or not args.checkpoint:
        return
    checkpoint_path = Path(args.checkpoint).expanduser()
    if str(checkpoint_path) == "/path/to/opf_checkpoint":
        raise ValueError(
            "--checkpoint /path/to/opf_checkpoint is a placeholder. "
            "Pass a real OPF checkpoint directory, set OPF_CHECKPOINT, or omit "
            "--checkpoint to use the default ~/.opf/privacy_filter location."
        )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}. "
            "Pass a real OPF checkpoint directory, set OPF_CHECKPOINT, or omit "
            "--checkpoint to use the default ~/.opf/privacy_filter location."
        )


def validate_eval_config(args: argparse.Namespace) -> None:
    """Reject benchmark configurations that are known to be misleading."""
    if args.eval_mode == "typed" and args.label_map == "none":
        raise ValueError(
            "--eval-mode typed requires --label-map opf-model-card for the "
            "ai4privacy dataset. Use --eval-mode untyped to ignore source labels."
        )


def write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    prepared_examples: int,
    metrics_path: Path,
    timings_path: Path,
    predictions_path: Path | None,
    opf_eval_seconds: float | None,
    label_stats: Mapping[str, Counter[str]],
) -> None:
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    timings = read_json(timings_path) if timings_path.exists() else {}
    metric_values = metrics.get("metrics", {})
    throughput = timings.get("throughput_tokens_per_second", {})
    counts = timings.get("counts", {})
    summary = {
        "schema_version": "opf.ai4privacy.benchmark.v1",
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "language": args.lang_filter,
            "prepared_examples": prepared_examples,
            "source_mode": args.source_mode,
            "text_field": args.text_field,
            "privacy_mask_field": args.privacy_mask_field,
            "tokens_field": args.tokens_field,
            "classes_field": args.classes_field,
            "label_counts": {
                "source": dict(sorted(label_stats["source"].items())),
                "target": dict(sorted(label_stats["target"].items())),
                "dropped_source": dict(
                    sorted(label_stats["dropped_source"].items())
                ),
            },
        },
        "opf": {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "eval_mode": args.eval_mode,
            "span_metrics_space": args.span_metrics_space,
            "decode_mode": args.decode_mode,
            "label_map": args.label_map,
        },
        "outputs": {
            "metrics": str(metrics_path),
            "timings": str(timings_path),
            "predictions": str(predictions_path) if predictions_path else None,
        },
        "counts": counts,
        "metrics": metric_values,
        "throughput_tokens_per_second": throughput,
        "opf_eval_wall_seconds": opf_eval_seconds,
    }
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ai4privacy data and benchmark OPF with opf eval."
    )
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-400k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--train-parquet", default=None)
    parser.add_argument("--validation-parquet", default=None)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--lang-filter", default="de")
    parser.add_argument(
        "--source-mode",
        choices=("privacy-mask", "mbert-token-join"),
        default="privacy-mask",
        help=(
            "Dataset-to-text conversion. 'privacy-mask' uses natural source_text "
            "and privacy_mask offsets; 'mbert-token-join' reproduces the legacy "
            "wordpiece-joined benchmark text."
        ),
    )
    parser.add_argument("--text-field", default="source_text")
    parser.add_argument("--privacy-mask-field", default="privacy_mask")
    parser.add_argument("--tokens-field", default="mbert_tokens")
    parser.add_argument("--classes-field", default="mbert_token_classes")
    parser.add_argument("--language-field", default="language")
    parser.add_argument(
        "--label-map",
        choices=("none", "opf-model-card"),
        default="none",
        help=(
            "Map source dataset labels before writing OPF JSONL. "
            "'opf-model-card' applies the PII-Masking mapping described in the "
            "OPF model card and drops unmapped labels."
        ),
    )
    parser.add_argument("--max-eval-examples", type=int, default=5000)
    parser.add_argument("--max-opf-examples", type=int, default=None)
    parser.add_argument("--out-dir", default="./opf_ai4privacy_bench_out")
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-ctx", type=int, default=None)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--preprocess-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--decode-mode", choices=("viterbi", "argmax"), default="viterbi")
    parser.add_argument("--eval-mode", choices=("typed", "untyped"), default="untyped")
    parser.add_argument("--span-metrics-space", choices=("char", "token"), default="char")
    parser.add_argument("--discard-overlapping-predicted-spans", action="store_true")
    parser.add_argument("--discard-overlapping-ground-truth-spans", action="store_true")
    parser.add_argument("--no-trim-whitespace", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_checkpoint_arg(args)
    validate_eval_config(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map = resolve_label_map(args.label_map)
    label_stats: dict[str, Counter[str]] = {
        "source": Counter(),
        "target": Counter(),
        "dropped_source": Counter(),
    }

    log.info(
        "Starting OPF ai4privacy benchmark | dataset=%s split=%s lang=%s "
        "source_mode=%s label_map=%s",
        args.dataset,
        args.split,
        args.lang_filter,
        args.source_mode,
        args.label_map,
    )
    dataset = load_source_dataset(args)
    eval_raw, actual_split = resolve_eval_split(dataset, args.split)
    eval_raw = filter_and_limit_split(eval_raw, args)
    log.info("Final eval split: %s examples", human(len(eval_raw)))

    prepared_path = out_dir / "opf_eval_dataset.jsonl"
    prepared_count = write_jsonl(
        prepared_path,
        (
            row_to_opf_record(
                row,
                idx,
                dataset_name=args.dataset,
                split_name=actual_split,
                source_mode=args.source_mode,
                text_field=args.text_field,
                privacy_mask_field=args.privacy_mask_field,
                tokens_field=args.tokens_field,
                classes_field=args.classes_field,
                language_field=args.language_field,
                label_map=label_map,
                label_stats=label_stats,
            )
            for idx, row in enumerate(eval_raw)
        ),
    )
    log.info("Wrote OPF eval dataset: %s (%s examples)", prepared_path, prepared_count)

    metrics_path = out_dir / "opf_metrics.json"
    timings_path = out_dir / "opf_timings.json"
    predictions_path = out_dir / "opf_predictions.jsonl" if args.write_predictions else None
    summary_path = out_dir / "summary.json"

    opf_eval_seconds: float | None = None
    if not args.prepare_only:
        command = build_opf_eval_command(
            dataset_path=prepared_path,
            metrics_path=metrics_path,
            timings_path=timings_path,
            predictions_path=predictions_path,
            args=args,
        )
        log.info("Run OPF eval: %s", " ".join(command))
        started = time.perf_counter()
        subprocess.run(command, check=True)
        opf_eval_seconds = time.perf_counter() - started
        log.info("OPF eval done in %.2fs", opf_eval_seconds)

    if metrics_path.exists() and timings_path.exists():
        write_summary(
            path=summary_path,
            args=args,
            prepared_examples=prepared_count,
            metrics_path=metrics_path,
            timings_path=timings_path,
            predictions_path=predictions_path,
            opf_eval_seconds=opf_eval_seconds,
            label_stats=label_stats,
        )
        log.info("Wrote benchmark summary: %s", summary_path)
    elif args.prepare_only:
        log.info("Prepare-only mode: skipping summary because OPF eval was not run")


if __name__ == "__main__":
    main()
