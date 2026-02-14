#!/usr/bin/env python3
"""
Sound of Silence — Split an audio file into multiple files based on detected silence.

Accepts settings from command-line arguments and/or a config file (configparser format).
Command-line values override config file values.
"""

import argparse
import configparser
import sys
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("default.conf")
CONFIG_SECTION = "silence"


def load_config(path: Path) -> configparser.ConfigParser:
    """Load and return a ConfigParser for the given path. No file is created if missing."""
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path, encoding="utf-8")
    return config


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with options that mirror config file keys."""
    parser = argparse.ArgumentParser(
        prog="silence",
        description="Split an audio file into multiple files based on detected silence.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file to split.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        metavar="FILE",
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for split audio files.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=None,
        metavar="DB",
        help="Volume threshold in dB below which is considered silence.",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum duration of silence to use as a split point (seconds).",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum duration of a non-silent segment to keep (seconds).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Padding in seconds to keep around silence boundaries.",
    )
    return parser


def get_settings(args: argparse.Namespace) -> dict:
    """
    Merge config file and command-line arguments into a single settings dict.
    CLI values override config values. Returns keys in a consistent format (e.g. snake_case).
    """
    config = load_config(args.config)
    settings = {}

    # ConfigParser stores values as strings; we map to expected types
    type_map = {
        "output_dir": str,
        "silence_threshold": float,
        "min_silence_duration": float,
        "min_segment_duration": float,
        "padding": float,
    }

    if config.has_section(CONFIG_SECTION):
        for key in config.options(CONFIG_SECTION):
            norm_key = key.lower().replace("-", "_")
            raw = config.get(CONFIG_SECTION, key).strip()
            if norm_key in type_map:
                conv = type_map[norm_key]
                try:
                    settings[norm_key] = conv(raw) if conv is not str else raw
                except ValueError:
                    pass  # skip invalid config values; CLI or defaults can fill in

    # Override with CLI where provided
    if args.output_dir is not None:
        settings["output_dir"] = str(args.output_dir)
    if args.silence_threshold is not None:
        settings["silence_threshold"] = args.silence_threshold
    if args.min_silence_duration is not None:
        settings["min_silence_duration"] = args.min_silence_duration
    if args.min_segment_duration is not None:
        settings["min_segment_duration"] = args.min_segment_duration
    if args.padding is not None:
        settings["padding"] = args.padding

    # Ensure input is always from CLI
    settings["input"] = args.input
    settings["config_path"] = args.config

    return settings


def main() -> int:
    """Entry point: parse args and config, then run (no split logic yet)."""
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings(args)

    # Placeholder: no split logic implemented yet
    print("Sound of Silence — settings loaded (no split logic yet)", file=sys.stderr)
    print(f"  Input: {settings.get('input')}", file=sys.stderr)
    for k, v in sorted(settings.items()):
        if k not in ("input", "config_path"):
            print(f"  {k}: {v}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
