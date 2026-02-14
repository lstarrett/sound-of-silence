#!/usr/bin/env python3
"""
Sound of Silence — Split an audio file into multiple files based on detected silence.

Uses pydub (split_on_silence). Accepts settings from command-line arguments
and/or a config file (configparser format). Command-line values override config.
"""

import argparse
import configparser
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence, split_on_silence


DEFAULT_CONFIG_PATH = Path("default.conf")
CONFIG_SECTION = "silence"


class ConfigError(Exception):
    """Raised when a config file contains an illegal or malformed value."""


# Defaults when not set by config or CLI
DEFAULTS = {
    "output_dir": "output",
    "silence_threshold": -40.0,
    "min_silence_duration": 0.5,
    "min_segment_length": 1.0,
    "max_segment_length": 600.0,
    "padding_start": 0.1,
    "padding_end": 0.1,
}


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
        help="Input audio file to split (e.g. .mp3, .wav, .m4a).",
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
        help="Volume threshold in dBFS below which is considered silence.",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum duration of silence to use as a split point (seconds).",
    )
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum length of each split segment (seconds); shorter segments are merged.",
    )
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Maximum length of each split segment (seconds); split at best silence within range.",
    )
    parser.add_argument(
        "--padding-start",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Padding in seconds added at the beginning of each split segment.",
    )
    parser.add_argument(
        "--padding-end",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Padding in seconds added at the end of each split segment.",
    )
    return parser


def get_settings(args: argparse.Namespace) -> dict:
    """
    Merge config file and command-line arguments into a single settings dict.
    CLI values override config values. Missing values use DEFAULTS.
    """
    config = load_config(args.config)
    settings = dict(DEFAULTS)

    type_map = {
        "output_dir": str,
        "silence_threshold": float,
        "min_silence_duration": float,
        "min_segment_length": float,
        "max_segment_length": float,
        "padding_start": float,
        "padding_end": float,
    }

    if config.has_section(CONFIG_SECTION):
        for key in config.options(CONFIG_SECTION):
            norm_key = key.lower().replace("-", "_")
            raw = config.get(CONFIG_SECTION, key).strip()
            if norm_key in type_map:
                conv = type_map[norm_key]
                try:
                    settings[norm_key] = conv(raw) if conv is not str else raw
                except (ValueError, TypeError) as e:
                    raise ConfigError(
                        f"Invalid value for '{key}' in config: {raw!r}. Expected a valid {conv.__name__}."
                    ) from e

    if args.output_dir is not None:
        settings["output_dir"] = str(args.output_dir)
    if args.silence_threshold is not None:
        settings["silence_threshold"] = args.silence_threshold
    if args.min_silence_duration is not None:
        settings["min_silence_duration"] = args.min_silence_duration
    if args.min_segment_length is not None:
        settings["min_segment_length"] = args.min_segment_length
    if args.max_segment_length is not None:
        settings["max_segment_length"] = args.max_segment_length
    if args.padding_start is not None:
        settings["padding_start"] = args.padding_start
    if args.padding_end is not None:
        settings["padding_end"] = args.padding_end

    settings["input"] = args.input
    settings["config_path"] = args.config
    return settings


def sec_to_ms(sec: float) -> int:
    """Convert seconds to milliseconds for pydub."""
    return int(round(sec * 1000))


def load_audio(path: Path) -> AudioSegment:
    """Load an audio file via pydub (format inferred from extension)."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return AudioSegment.from_file(path)


def merge_short_chunks(
    chunks: list[AudioSegment],
    min_len_ms: int,
) -> list[AudioSegment]:
    """Merge consecutive chunks that are shorter than min_len_ms."""
    if not chunks or min_len_ms <= 0:
        return chunks
    merged: list[AudioSegment] = []
    acc: AudioSegment | None = None
    for ch in chunks:
        if acc is None:
            acc = ch
            continue
        if len(acc) < min_len_ms:
            acc = acc + ch
        else:
            merged.append(acc)
            acc = ch
    if acc is not None:
        merged.append(acc)
    return merged


def split_long_chunk(
    chunk: AudioSegment,
    max_len_ms: int,
    min_silence_len_ms: int,
    silence_thresh: float,
) -> list[AudioSegment]:
    """Split a chunk at the best silence so no segment exceeds max_len_ms."""
    if len(chunk) <= max_len_ms or max_len_ms <= 0:
        return [chunk]
    silences = detect_silence(
        chunk,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    # Prefer splitting at a silence near target = max_len_ms from start
    target_ms = max_len_ms
    best_start = None
    best_dist = float("inf")
    for start_ms, end_ms in silences:
        # Split point is the middle of the silence
        split_at = (start_ms + end_ms) // 2
        if split_at < min_silence_len_ms:
            continue
        dist = abs(split_at - target_ms)
        if dist < best_dist:
            best_dist = dist
            best_start = split_at
    if best_start is None or best_start >= len(chunk):
        return [chunk]
    left = chunk[:best_start]
    right = chunk[best_start:]
    return (
        split_long_chunk(left, max_len_ms, min_silence_len_ms, silence_thresh)
        + split_long_chunk(right, max_len_ms, min_silence_len_ms, silence_thresh)
    )


def enforce_min_max_segments(
    chunks: list[AudioSegment],
    min_len_ms: int,
    max_len_ms: int,
    min_silence_len_ms: int,
    silence_thresh: float,
) -> list[AudioSegment]:
    """Merge chunks under min length, then split chunks over max length at best silence."""
    chunks = merge_short_chunks(chunks, min_len_ms)
    result: list[AudioSegment] = []
    for ch in chunks:
        result.extend(
            split_long_chunk(ch, max_len_ms, min_silence_len_ms, silence_thresh)
        )
    return result


def run_split(settings: dict) -> list[Path]:
    """Load audio, split on silence, apply min/max and padding, export. Returns output paths."""
    input_path = Path(settings["input"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    min_silence_ms = sec_to_ms(settings["min_silence_duration"])
    silence_thresh = settings["silence_threshold"]
    min_seg_ms = sec_to_ms(settings["min_segment_length"])
    max_seg_ms = sec_to_ms(settings["max_segment_length"])
    pad_start_ms = sec_to_ms(settings["padding_start"])
    pad_end_ms = sec_to_ms(settings["padding_end"])

    audio = load_audio(input_path)
    # Split on silence; keep_silence=0, we add our own padding per segment
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh,
        keep_silence=0,
        seek_step=10,
    )
    if not chunks:
        # No silence found — treat whole file as one segment
        chunks = [audio]

    chunks = enforce_min_max_segments(
        chunks, min_seg_ms, max_seg_ms, min_silence_ms, silence_thresh
    )

    suffix = input_path.suffix.lower() or ".mp3"
    out_paths: list[Path] = []
    for i, chunk in enumerate(chunks):
        padded = (
            AudioSegment.silent(duration=pad_start_ms)
            + chunk
            + AudioSegment.silent(duration=pad_end_ms)
        )
        out_name = f"segment_{i:04d}{suffix}"
        out_path = output_dir / out_name
        padded.export(out_path, format=suffix.lstrip("."), bitrate="192k")
        out_paths.append(out_path)
        print(f"Exported {out_name}", file=sys.stderr)
    return out_paths


def main() -> int:
    """Entry point: parse args and config, run split, exit."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        settings = get_settings(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    input_path = settings["input"]
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        run_split(settings)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
