import argparse
import sys
from pathlib import Path

from .._common.constants import (
    DEFAULT_MODEL_ENV_VAR,
    DEFAULT_MODEL_PATH,
    OUTPUT_MODES,
)

CHECKPOINT_HELP = (
    "Override checkpoint directory. If omitted, OPF uses "
    f"{DEFAULT_MODEL_ENV_VAR} or downloads/reuses {DEFAULT_MODEL_PATH}."
)
N_CTX_HELP = "Override context window length"
DISCARD_OVERLAPPING_PREDICTED_SPANS_HELP = (
    "Discard overlapping predicted spans per label"
)
TRIM_WHITESPACE_HELP = "Trim whitespace from predicted character spans (default)"
NO_TRIM_WHITESPACE_HELP = "Do not trim whitespace from predicted character spans"
OUTPUT_MODE_HELP = (
    "typed: keep model categories.\n"
    "redacted: collapse all spans into one generic redacted label."
)


class CliHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Help formatter used by the OPF CLI."""

    pass


def resolve_prog(module_prog: str) -> str:
    """Resolve the displayed program name for argparse help output."""
    argv0 = Path(sys.argv[0]).name
    if argv0 in {"", "__main__.py", "args.py"}:
        return module_prog
    return argv0


def add_checkpoint_arg(parser: object) -> None:
    """Add the shared ``--checkpoint`` argument to a parser or argument group."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=CHECKPOINT_HELP,
    )


def add_device_arg(parser: object) -> None:
    """Add the shared ``--device`` argument."""
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device to run on. 'auto' (default) picks cuda if a GPU is "
            "available, otherwise cpu. Pass 'cuda' or 'cpu' explicitly to "
            "override or to get a loud error when the requested backend is "
            "unavailable."
        ),
    )


def add_n_ctx_arg(parser: object) -> None:
    """Add the shared context-window override argument."""
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=None,
        help=N_CTX_HELP,
    )


def add_decode_mode_arg(parser: object) -> None:
    """Add the shared decode-mode selection argument."""
    parser.add_argument(
        "--decode-mode",
        choices=("viterbi", "argmax"),
        default="viterbi",
        help="Decode token labels with constrained Viterbi or independent argmax.",
    )


def add_discard_overlapping_predicted_spans_arg(parser: object) -> None:
    """Add the shared overlap-discard flag for predicted spans."""
    parser.add_argument(
        "--discard-overlapping-predicted-spans",
        action="store_true",
        help=DISCARD_OVERLAPPING_PREDICTED_SPANS_HELP,
    )


def add_trim_whitespace_args(
    parser: object,
    group: object,
) -> None:
    """Add the shared whitespace-trimming flags."""
    group.add_argument(
        "--trim-whitespace",
        dest="trim_span_whitespace",
        action="store_true",
        help=TRIM_WHITESPACE_HELP,
    )
    group.add_argument(
        "--no-trim-whitespace",
        dest="trim_span_whitespace",
        action="store_false",
        help=NO_TRIM_WHITESPACE_HELP,
    )
    parser.set_defaults(trim_span_whitespace=True)


def add_output_mode_arg(parser: object) -> None:
    """Add the shared output-mode argument."""
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODES,
        default="typed",
        help=OUTPUT_MODE_HELP,
    )


def add_viterbi_args(parser: object) -> None:
    """Add the shared Viterbi calibration argument."""
    parser.add_argument(
        "--viterbi-calibration-path",
        type=str,
        default=None,
        help=(
            "Path to local JSON Viterbi calibration artifact. If omitted, "
            "auto-discovers <checkpoint>/viterbi_calibration.json; if absent, "
            "uses all-zero transition biases."
        ),
    )
