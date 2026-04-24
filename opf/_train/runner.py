from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
import tiktoken

from .args import parse_args
from .._api import resolve_checkpoint_path
from .._common.constants import SCHEMA_VERSION
from .._common.device import resolve_device
from .._common.label_space import (
    resolve_checkpoint_label_space,
    resolve_label_space_from_config,
)
from .._core.runtime import _load_checkpoint_config, _resolve_n_ctx
from .._core.sequence_labeling import (
    LabelInfo,
    TokenizedExample,
    Window,
    build_label_info,
    example_to_windows,
)
from .._eval.data import iter_json_records, iter_message_variant_records
from .._eval.preprocess import labels_from_entities, parse_record, token_char_ranges
from .._model.model import Transformer
from .._model.weights import save_named_tensors


@dataclass(frozen=True)
class LoopStats:
    """Aggregated loss/accuracy stats for one train or validation loop."""

    loss: float
    token_accuracy: float
    tokens: int
    batches: int
    optimizer_steps: int = 0


@dataclass(frozen=True)
class OutputHeadRemapStats:
    """Row-copy stats when rebuilding the output head for a target label space."""

    exact_rows_copied: int
    fallback_rows_copied: int

    @property
    def total_rows_copied(self) -> int:
        return self.exact_rows_copied + self.fallback_rows_copied


def _format_duration(seconds: float) -> str:
    """Format one duration in seconds for human-readable progress logs."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _collect_example_ids(windows: Sequence[Window]) -> set[str]:
    """Collect unique example ids referenced by one batch of windows."""
    seen: set[str] = set()
    for window in windows:
        if window.token_example_ids:
            for example_id in window.token_example_ids:
                if example_id is not None:
                    seen.add(example_id)
            continue
        if window.example_id is not None:
            seen.add(window.example_id)
    return seen


def _iter_dataset_records(
    dataset_path: str,
    *,
    dataset_variant: str,
) -> Iterable[Mapping[str, object]]:
    records: Iterable[Mapping[str, object]] = iter_json_records(dataset_path)
    if dataset_variant == "message":
        records = iter_message_variant_records(records)
    return records


def _prepare_tokenized_examples(
    *,
    dataset_path: str,
    dataset_variant: str,
    encoding: tiktoken.Encoding,
    label_info: LabelInfo,
    max_examples: int | None,
) -> list[TokenizedExample]:
    examples: list[TokenizedExample] = []
    for idx, record in enumerate(
        _iter_dataset_records(dataset_path, dataset_variant=dataset_variant)
    ):
        example_id, text, entities = parse_record(record, idx)
        tokens = list(encoding.encode(text, allowed_special="all"))
        if not tokens:
            continue
        char_starts, char_ends = token_char_ranges(tokens, encoding, text)
        labels = labels_from_entities(
            tokens=tokens,
            text=text,
            entities=entities,
            label_info=label_info,
            encoding=encoding,
            eval_mode="typed",
            token_char_ranges_hint=(char_starts, char_ends),
        )
        if len(tokens) != len(labels):
            raise ValueError(
                "Prepared example has mismatched token and label lengths "
                f"for example_id={example_id}"
            )
        examples.append(
            TokenizedExample(
                tokens=tuple(tokens),
                labels=tuple(labels),
                example_id=example_id,
                text=text,
            )
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


def _split_train_validation(
    examples: Sequence[TokenizedExample],
    *,
    validation_split: float,
    shuffle_seed: int,
) -> tuple[list[TokenizedExample], list[TokenizedExample]]:
    if not (0.0 <= validation_split < 1.0):
        raise ValueError("--validation-split must be in [0, 1).")
    if validation_split == 0.0:
        return list(examples), []
    if len(examples) < 2:
        return list(examples), []

    shuffled = list(examples)
    random.Random(shuffle_seed).shuffle(shuffled)
    validation_count = int(math.floor(len(shuffled) * validation_split))
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_examples = shuffled[:validation_count]
    train_examples = shuffled[validation_count:]
    return train_examples, validation_examples


def _build_windows(
    examples: Sequence[TokenizedExample],
    *,
    n_ctx: int,
) -> list[Window]:
    windows: list[Window] = []
    for example in examples:
        for window in example_to_windows(example, n_ctx):
            if window.tokens:
                windows.append(window)
    return windows


def _build_epoch_batches(
    windows: Sequence[Window],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[list[Window]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    shuffled = list(windows)
    rng.shuffle(shuffled)
    batches: list[list[Window]] = []
    for start in range(0, len(shuffled), batch_size):
        batches.append(shuffled[start : start + batch_size])
    return batches


def _batch_to_tensors(
    windows: Sequence[Window],
    *,
    device: torch.device,
    pad_token_id: int,
    pad_label_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not windows:
        raise ValueError("Cannot build tensors for an empty window batch")
    max_window_len = max(len(window.tokens) for window in windows)
    token_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[float]] = []
    for window in windows:
        pad_count = max_window_len - len(window.tokens)
        token_rows.append(list(window.tokens) + ([pad_token_id] * pad_count))
        label_rows.append(list(window.labels) + ([pad_label_id] * pad_count))
        mask_rows.append(list(window.mask) + ([0.0] * pad_count))
    tokens = torch.tensor(
        token_rows,
        device=device,
        dtype=torch.long,
    )
    labels = torch.tensor(
        label_rows,
        device=device,
        dtype=torch.long,
    )
    masks = torch.tensor(
        mask_rows,
        device=device,
        dtype=torch.float32,
    )
    return tokens, labels, masks


def _masked_token_loss_and_accuracy(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if logits.dim() != 3:
        raise ValueError(
            f"Expected logits with shape [batch, tokens, classes], got {tuple(logits.shape)}"
        )
    if logits.shape[:2] != labels.shape or labels.shape != masks.shape:
        raise ValueError(
            "Logits/labels/masks shape mismatch: "
            f"logits={tuple(logits.shape)} labels={tuple(labels.shape)} masks={tuple(masks.shape)}"
        )

    num_classes = int(logits.shape[-1])
    per_token_loss = F.cross_entropy(
        logits.float().reshape(-1, num_classes),
        labels.reshape(-1),
        reduction="none",
    )
    mask_flat = masks.reshape(-1)
    valid_tokens = int(mask_flat.sum().item())
    if valid_tokens <= 0:
        zero = per_token_loss.new_zeros(())
        return zero, 0, 0

    loss = (per_token_loss * mask_flat).sum() / mask_flat.sum()

    predictions = logits.argmax(dim=-1)
    mask_bool = masks.bool()
    correct = int(((predictions == labels) & mask_bool).sum().item())
    return loss, valid_tokens, correct


def _train_one_epoch(
    *,
    model: Transformer,
    windows: Sequence[Window],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    rng: random.Random,
    pad_token_id: int,
    pad_label_id: int,
    epoch_index: int,
    num_epochs: int,
    expected_examples: int,
    progress_interval_s: float,
) -> LoopStats:
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")

    epoch_batches = _build_epoch_batches(windows, batch_size=batch_size, rng=rng)
    if not epoch_batches:
        raise ValueError("No training batches were built; check dataset and n_ctx")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss_sum = 0.0
    total_tokens = 0
    total_correct = 0
    optimizer_steps = 0
    processed_windows = 0
    seen_example_ids: set[str] = set()
    total_batches = len(epoch_batches)
    epoch_start = time.perf_counter()
    next_progress_at = (
        epoch_start + progress_interval_s if progress_interval_s > 0.0 else float("inf")
    )

    for batch_idx, batch_windows in enumerate(epoch_batches, start=1):
        processed_windows += len(batch_windows)
        seen_example_ids.update(_collect_example_ids(batch_windows))
        tokens, labels, masks = _batch_to_tensors(
            batch_windows,
            device=device,
            pad_token_id=pad_token_id,
            pad_label_id=pad_label_id,
        )
        logits = model(tokens, attention_mask=masks.bool())
        loss, batch_tokens, batch_correct = _masked_token_loss_and_accuracy(
            logits=logits,
            labels=labels,
            masks=masks,
        )
        if batch_tokens == 0:
            continue

        total_loss_sum += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
        total_correct += batch_correct

        (loss / grad_accum_steps).backward()

        should_step = batch_idx % grad_accum_steps == 0 or batch_idx == len(
            epoch_batches
        )
        if not should_step:
            continue

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

        now = time.perf_counter()
        should_report_progress = now >= next_progress_at or batch_idx == total_batches
        if should_report_progress:
            elapsed = now - epoch_start
            batch_fraction = float(batch_idx) / float(total_batches)
            estimated_epoch_total = (
                elapsed / batch_fraction if batch_fraction > 0.0 else 0.0
            )
            eta_epoch_s = max(0.0, estimated_epoch_total - elapsed)
            remaining_epochs = max(0, num_epochs - epoch_index)
            eta_total_s = eta_epoch_s + (estimated_epoch_total * remaining_epochs)
            running_loss = total_loss_sum / total_tokens if total_tokens > 0 else 0.0
            running_accuracy = (
                float(total_correct) / float(total_tokens) if total_tokens > 0 else 0.0
            )
            print(
                "train progress: "
                f"epoch={epoch_index}/{num_epochs} "
                f"batch={batch_idx}/{total_batches} "
                f"windows={processed_windows}/{len(windows)} "
                f"examples_seen={len(seen_example_ids)}/{expected_examples} "
                f"tokens={total_tokens} "
                f"train_loss={running_loss:.6f} "
                f"train_token_accuracy={running_accuracy:.4f} "
                f"eta_epoch={_format_duration(eta_epoch_s)} "
                f"eta_total={_format_duration(eta_total_s)}",
                flush=True,
            )
            if progress_interval_s > 0.0:
                next_progress_at = now + progress_interval_s

    if total_tokens <= 0:
        raise ValueError("No valid training tokens were observed in this epoch")

    return LoopStats(
        loss=total_loss_sum / total_tokens,
        token_accuracy=float(total_correct) / float(total_tokens),
        tokens=total_tokens,
        batches=len(epoch_batches),
        optimizer_steps=optimizer_steps,
    )


def _evaluate_windows(
    *,
    model: Transformer,
    windows: Sequence[Window],
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
    pad_label_id: int,
) -> LoopStats:
    batches = _build_epoch_batches(
        windows,
        batch_size=batch_size,
        rng=random.Random(0),
    )
    if not batches:
        raise ValueError("No validation batches were built; check dataset and n_ctx")

    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.inference_mode():
        for batch_windows in batches:
            tokens, labels, masks = _batch_to_tensors(
                batch_windows,
                device=device,
                pad_token_id=pad_token_id,
                pad_label_id=pad_label_id,
            )
            logits = model(tokens, attention_mask=masks.bool())
            loss, batch_tokens, batch_correct = _masked_token_loss_and_accuracy(
                logits=logits,
                labels=labels,
                masks=masks,
            )
            if batch_tokens == 0:
                continue
            total_loss_sum += float(loss.item()) * batch_tokens
            total_tokens += batch_tokens
            total_correct += batch_correct

    if total_tokens <= 0:
        raise ValueError("No valid validation tokens were observed")

    return LoopStats(
        loss=total_loss_sum / total_tokens,
        token_accuracy=float(total_correct) / float(total_tokens),
        tokens=total_tokens,
        batches=len(batches),
    )


def _resolve_output_dtype(
    *,
    output_param_dtype_flag: str,
    base_config: Mapping[str, object],
) -> tuple[str, torch.dtype]:
    if output_param_dtype_flag == "inherit":
        configured = str(base_config.get("param_dtype", "bfloat16")).strip().lower()
        output_param_dtype_flag = (
            "bf16" if configured in {"bf16", "bfloat16"} else "fp32"
        )
    if output_param_dtype_flag == "bf16":
        return "bfloat16", torch.bfloat16
    if output_param_dtype_flag == "fp32":
        return "fp32", torch.float32
    raise ValueError(
        f"Unsupported --output-param-dtype value: {output_param_dtype_flag!r}"
    )


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"--output-dir is not a directory: {path}")
        has_entries = any(path.iterdir())
        if has_entries and not overwrite:
            raise FileExistsError(
                f"--output-dir already contains files: {path} "
                "(pass --overwrite-output to allow reuse)"
            )
    path.mkdir(parents=True, exist_ok=True)


def _load_custom_label_space(
    config_path: str | None,
) -> tuple[str, tuple[str, ...], tuple[str, ...], str] | None:
    """Load one optional custom label-space config payload from JSON."""
    if config_path is None:
        return None
    resolved_path = Path(config_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"--label-space-json must contain an object payload (got {type(payload)!r})"
        )
    category_version, span_class_names, ner_class_names = (
        resolve_label_space_from_config(
            payload,
            context=str(resolved_path),
        )
    )
    return (
        category_version,
        span_class_names,
        ner_class_names,
        str(resolved_path),
    )


def _rebuild_output_head_for_target_labels(
    model: Transformer,
    *,
    base_ner_class_names: Sequence[str],
    target_ner_class_names: Sequence[str],
    device: torch.device,
) -> OutputHeadRemapStats:
    """Rebuild and remap the output head so rows match the target label space."""
    old_head = model.unembedding
    old_weight = old_head.weight.detach().to(dtype=torch.float32).clone()
    new_head = torch.nn.Linear(
        old_head.in_features,
        len(target_ner_class_names),
        bias=False,
        device=device,
        dtype=torch.float32,
    )
    torch.nn.init.normal_(new_head.weight, mean=0.0, std=0.02)

    base_index_by_name = {name: idx for idx, name in enumerate(base_ner_class_names)}
    fallback_by_boundary: dict[str, int] = {}
    for boundary in ("B", "I", "E", "S"):
        preferred_name = f"{boundary}-secret"
        preferred_idx = base_index_by_name.get(preferred_name)
        if preferred_idx is not None and preferred_idx < old_weight.shape[0]:
            fallback_by_boundary[boundary] = preferred_idx
            continue
        for name, idx in base_index_by_name.items():
            if idx >= old_weight.shape[0]:
                continue
            if name.startswith(f"{boundary}-"):
                fallback_by_boundary[boundary] = idx
                break

    copied_exact = 0
    copied_fallback = 0
    with torch.no_grad():
        for new_idx, name in enumerate(target_ner_class_names):
            old_idx = base_index_by_name.get(name)
            if old_idx is not None and old_idx < old_weight.shape[0]:
                new_head.weight[new_idx].copy_(old_weight[old_idx])
                copied_exact += 1
                continue

            boundary = None
            if "-" in name:
                maybe_boundary, _base = name.split("-", 1)
                if maybe_boundary in {"B", "I", "E", "S"}:
                    boundary = maybe_boundary
            if boundary is None:
                continue
            fallback_idx = fallback_by_boundary.get(boundary)
            if fallback_idx is None:
                continue
            new_head.weight[new_idx].copy_(old_weight[fallback_idx])
            copied_fallback += 1

    model.unembedding = new_head
    return OutputHeadRemapStats(
        exact_rows_copied=copied_exact,
        fallback_rows_copied=copied_fallback,
    )


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None, prog=prog)

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        raise ValueError("--max-train-examples must be > 0 when provided")
    if args.max_validation_examples is not None and args.max_validation_examples <= 0:
        raise ValueError("--max-validation-examples must be > 0 when provided")

    progress_interval_s = 15.0
    progress_interval_raw = os.environ.get("OPF_TRAIN_PROGRESS_INTERVAL_S")
    if progress_interval_raw is not None:
        try:
            parsed_interval = float(progress_interval_raw)
        except ValueError:
            print(
                "warning: invalid OPF_TRAIN_PROGRESS_INTERVAL_S="
                f"{progress_interval_raw!r}; using default {progress_interval_s:.1f}s",
                flush=True,
            )
        else:
            if parsed_interval <= 0.0:
                progress_interval_s = 0.0
            else:
                progress_interval_s = parsed_interval

    checkpoint = resolve_checkpoint_path(args.checkpoint)
    device = resolve_device(args.device)

    # Default to Triton-backed MoE kernels on non-CPU devices unless callers
    # explicitly opt out. CPU uses torch ops by default so Triton stays optional.
    if device.type != "cpu":
        os.environ.setdefault("OPF_MOE_TRITON", "1")

    base_config = _load_checkpoint_config(checkpoint)
    n_ctx = _resolve_n_ctx(base_config, args.n_ctx, device)
    encoding_name = base_config.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Checkpoint config field encoding must be a non-empty string")
    encoding = tiktoken.get_encoding(encoding_name)
    pad_token_id = int(encoding.eot_token)
    active_encoding_name = encoding_name

    (
        checkpoint_category_version,
        _checkpoint_span_class_names,
        checkpoint_ner_class_names,
    ) = resolve_checkpoint_label_space(checkpoint)
    custom_label_space = _load_custom_label_space(args.label_space_json)
    if custom_label_space is None:
        resolved_category_version = checkpoint_category_version
        resolved_ner_class_names = checkpoint_ner_class_names
        label_space_source = "checkpoint"
        resolved_label_space_path: str | None = None
    else:
        (
            resolved_category_version,
            _custom_span_class_names,
            resolved_ner_class_names,
            resolved_label_space_path,
        ) = custom_label_space
        label_space_source = "label-space-json"
    label_info = build_label_info(resolved_ner_class_names)

    # Keep the base checkpoint label-space metadata for compatibility/debugging.
    base_ner_class_names = checkpoint_ner_class_names

    train_examples_all = _prepare_tokenized_examples(
        dataset_path=args.dataset,
        dataset_variant=args.dataset_variant,
        encoding=encoding,
        label_info=label_info,
        max_examples=args.max_train_examples,
    )
    if not train_examples_all:
        raise ValueError("No training examples were loaded from --dataset")

    validation_variant = args.validation_dataset_variant or args.dataset_variant
    if args.validation_dataset:
        train_examples = train_examples_all
        validation_examples = _prepare_tokenized_examples(
            dataset_path=args.validation_dataset,
            dataset_variant=validation_variant,
            encoding=encoding,
            label_info=label_info,
            max_examples=args.max_validation_examples,
        )
    else:
        train_examples, validation_examples = _split_train_validation(
            train_examples_all,
            validation_split=args.validation_split,
            shuffle_seed=args.shuffle_seed,
        )
        if args.max_validation_examples is not None:
            validation_examples = validation_examples[: args.max_validation_examples]
        if args.validation_split > 0.0 and not validation_examples:
            print(
                "warning: validation split requested but not enough examples were "
                "available; continuing without validation",
                flush=True,
            )

    train_windows = _build_windows(
        train_examples,
        n_ctx=n_ctx,
    )
    if not train_windows:
        raise ValueError("No train windows were produced; check dataset and --n-ctx")

    validation_windows = _build_windows(
        validation_examples,
        n_ctx=n_ctx,
    )

    model = Transformer.from_checkpoint(checkpoint, device=device)
    model = model.to(dtype=torch.float32)
    output_head_remap_stats = OutputHeadRemapStats(
        exact_rows_copied=0,
        fallback_rows_copied=0,
    )
    output_head_reinitialized = False
    if tuple(resolved_ner_class_names) != tuple(base_ner_class_names):
        output_head_remap_stats = _rebuild_output_head_for_target_labels(
            model,
            base_ner_class_names=base_ner_class_names,
            target_ner_class_names=resolved_ner_class_names,
            device=device,
        )
        output_head_reinitialized = True
        print(
            "info: rebuilt output head for target label space "
            f"({len(resolved_ner_class_names)} labels; "
            f"copied_rows={output_head_remap_stats.total_rows_copied}; "
            f"exact={output_head_remap_stats.exact_rows_copied}; "
            f"fallback={output_head_remap_stats.fallback_rows_copied})",
            flush=True,
        )

    trainable_params = list(model.parameters())

    print(
        "training plan: "
        f"epochs={args.epochs} "
        f"train_examples={len(train_examples)} "
        f"train_windows={len(train_windows)} "
        f"validation_examples={len(validation_examples)} "
        f"validation_windows={len(validation_windows)} "
        f"progress_interval_s={progress_interval_s:.1f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    epoch_rng = random.Random(args.shuffle_seed)
    epoch_summaries: list[dict[str, object]] = []
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metric_name = "validation_loss" if validation_windows else "train_loss"

    overall_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = _train_one_epoch(
            model=model,
            windows=train_windows,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            rng=epoch_rng,
            pad_token_id=pad_token_id,
            pad_label_id=label_info.background_token_label,
            epoch_index=epoch,
            num_epochs=args.epochs,
            expected_examples=len(train_examples),
            progress_interval_s=progress_interval_s,
        )

        validation_stats: LoopStats | None = None
        if validation_windows:
            validation_stats = _evaluate_windows(
                model=model,
                windows=validation_windows,
                device=device,
                batch_size=args.batch_size,
                pad_token_id=pad_token_id,
                pad_label_id=label_info.background_token_label,
            )

        tracked_metric = (
            validation_stats.loss if validation_stats is not None else train_stats.loss
        )
        if tracked_metric < best_metric:
            best_metric = tracked_metric
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        epoch_summary: dict[str, object] = {
            "epoch": epoch,
            "elapsed_s": time.perf_counter() - epoch_start,
            "train_loss": train_stats.loss,
            "train_token_accuracy": train_stats.token_accuracy,
            "train_tokens": train_stats.tokens,
            "train_batches": train_stats.batches,
            "optimizer_steps": train_stats.optimizer_steps,
        }
        if validation_stats is not None:
            epoch_summary.update(
                {
                    "validation_loss": validation_stats.loss,
                    "validation_token_accuracy": validation_stats.token_accuracy,
                    "validation_tokens": validation_stats.tokens,
                    "validation_batches": validation_stats.batches,
                }
            )
        epoch_summaries.append(epoch_summary)

        if validation_stats is None:
            print(
                f"epoch {epoch}/{args.epochs}: train_loss={train_stats.loss:.6f} "
                f"train_token_accuracy={train_stats.token_accuracy:.4f} "
                f"optimizer_steps={train_stats.optimizer_steps}",
                flush=True,
            )
        else:
            print(
                f"epoch {epoch}/{args.epochs}: train_loss={train_stats.loss:.6f} "
                f"val_loss={validation_stats.loss:.6f} "
                f"val_token_accuracy={validation_stats.token_accuracy:.4f} "
                f"optimizer_steps={train_stats.optimizer_steps}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Training finished without any tracked metrics")
    model.load_state_dict(best_state, strict=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    _ensure_output_dir(output_dir, overwrite=args.overwrite_output)

    serialized_param_dtype, output_dtype = _resolve_output_dtype(
        output_param_dtype_flag=args.output_param_dtype,
        base_config=base_config,
    )

    output_config = dict(base_config)
    output_config["param_dtype"] = serialized_param_dtype
    output_config["num_labels"] = len(resolved_ner_class_names)
    output_config["category_version"] = resolved_category_version
    output_config["span_class_names"] = list(label_info.span_class_names)
    output_config["ner_class_names"] = list(resolved_ner_class_names)
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    weights_path = output_dir / "model.safetensors"
    save_named_tensors(
        weights_path,
        {name: param for name, param in model.named_parameters()},
        dtype=output_dtype,
    )

    summary_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "base_checkpoint": checkpoint,
        "output_checkpoint_dir": str(output_dir),
        "checkpoint_category_version": checkpoint_category_version,
        "resolved_category_version": resolved_category_version,
        "label_space_source": label_space_source,
        "label_space_json_path": resolved_label_space_path,
        "num_output_labels": len(resolved_ner_class_names),
        "output_head_reinitialized": output_head_reinitialized,
        "output_head_rows_copied": output_head_remap_stats.total_rows_copied,
        "output_head_rows_copied_exact": output_head_remap_stats.exact_rows_copied,
        "output_head_rows_copied_fallback": output_head_remap_stats.fallback_rows_copied,
        "span_class_names": list(label_info.span_class_names),
        "encoding": active_encoding_name,
        "resolved_n_ctx": n_ctx,
        "device": str(device),
        "train_dataset": args.dataset,
        "train_dataset_variant": args.dataset_variant,
        "validation_dataset": args.validation_dataset,
        "validation_dataset_variant": validation_variant
        if args.validation_dataset
        else None,
        "validation_split": (
            None if args.validation_dataset else float(args.validation_split)
        ),
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(validation_examples),
        "num_train_windows": len(train_windows),
        "num_validation_windows": len(validation_windows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "best_epoch": best_epoch,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "serialized_param_dtype": serialized_param_dtype,
        "epoch_metrics": epoch_summaries,
        "elapsed_s": time.perf_counter() - overall_start,
        "artifacts": {
            "config_json": str(config_path),
            "model_safetensors": str(weights_path),
            "summary_json": str(output_dir / args.summary_name),
        },
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    usage_path = output_dir / "USAGE.txt"
    usage_path.write_text(
        "\n".join(
            [
                "Finetuned checkpoint generated by `opf train`.",
                "",
                "Run local inference:",
                f'  opf --checkpoint {output_dir} --device {device.type} "Alice was born on 1990-01-02."',
                "",
                "Run eval:",
                f"  opf eval /path/to/eval.jsonl --checkpoint {output_dir} --device {device.type}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("finetune complete", flush=True)
    print(f"output_checkpoint: {output_dir}", flush=True)
    print(
        f"best_epoch: {best_epoch} {best_metric_name}: {best_metric:.6f}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
