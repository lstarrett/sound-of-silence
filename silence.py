#!/usr/bin/env python3
"""
Sound of Silence — Split an audio file into multiple files based on detected silence.

Uses pydub (detect_silence). Splits at the midpoint of each qualified silence
so no audio is cut off. Accepts settings from command-line arguments and/or
a config file (configparser format). Command-line values override config.
"""

import argparse
import configparser
import sys
import threading
import time
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence


# Default config path: config.ini in the current working directory (overridden by --config / -c)
DEFAULT_CONFIG_PATH = Path.cwd() / "config.ini"
CONFIG_SECTION = "silence"
LARGE_FILE_NOTE_BYTES = 100 * 1024 * 1024  # 100 MB


def _format_duration_ms(ms: float) -> str:
    """Format duration in ms as HH:MM:SS, rounded to nearest second."""
    total_sec = round(ms / 1000.0)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_size_bytes(size_bytes: int) -> str:
    """Format size as human-readable string (e.g. '45.2 MB')."""
    mb = size_bytes / (1024 * 1024)
    return f"{mb:.1f} MB"


def _format_elapsed_mm_ss(seconds: float) -> str:
    """Format elapsed time as mm:ss."""
    total = int(seconds)
    m, s = total // 60, total % 60
    return f"{m:02d}:{s:02d}"


def _format_elapsed_hh_mm_ss(seconds: float) -> str:
    """Format elapsed time as hh:mm:ss."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run_with_elapsed(message: str, func, *args, **kwargs):
    """Run func(*args, **kwargs); show 'Message... [mm:ss]' and update every second until done."""
    result = [None]
    err = [None]
    start = time.perf_counter()

    def work():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            err[0] = e

    t = threading.Thread(target=work)
    t.start()
    while t.is_alive():
        elapsed = time.perf_counter() - start
        line = f"{message}... [{_format_elapsed_mm_ss(elapsed)}]"
        print(f"\r{line}", end="", file=sys.stderr)
        sys.stderr.flush()
        time.sleep(1)
    line = f"{message}... [{_format_elapsed_mm_ss(time.perf_counter() - start)}]"
    print(f"\r{line}", file=sys.stderr)
    if err[0] is not None:
        raise err[0]
    return result[0]


def _export_segment_with_elapsed(label: str, export_fn) -> None:
    """Run export_fn(); if it takes longer than 1s, show 'label [mm:ss]' updating every second."""
    err = [None]
    start = time.perf_counter()

    def work():
        try:
            export_fn()
        except Exception as e:
            err[0] = e

    t = threading.Thread(target=work)
    t.start()
    time.sleep(1)
    if not t.is_alive():
        print(f"  {label} [00:00]", file=sys.stderr)
        if err[0] is not None:
            raise err[0]
        return
    while t.is_alive():
        elapsed = time.perf_counter() - start
        line = f"  {label} [{_format_elapsed_mm_ss(elapsed)}]"
        print(f"\r{line}", end="", file=sys.stderr)
        sys.stderr.flush()
        time.sleep(1)
    line = f"  {label} [{_format_elapsed_mm_ss(time.perf_counter() - start)}]"
    print(f"\r{line}", file=sys.stderr)
    if err[0] is not None:
        raise err[0]


class ConfigError(Exception):
    """Raised when a config file contains an illegal or malformed value."""


# Required settings (must be present in config or CLI); used for validation and type conversion
_REQUIRED_SETTINGS = (
    "output_dir",
    "silence_threshold",
    "min_silence_duration",
    "min_segment_length",
    "max_segment_length",
    "max_padding",
    "segment_label",
    "segment_num_min_digits",
)
_TYPE_MAP = {
    "output_dir": str,
    "silence_threshold": float,
    "min_silence_duration": float,
    "min_segment_length": float,
    "max_segment_length": float,
    "max_padding": float,
    "segment_label": str,
    "segment_num_min_digits": int,
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
        "--max-padding",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Max silence kept at split points: if a silence is longer than 2× this value, only this much is kept on each side (seconds).",
    )
    parser.add_argument(
        "--segment-label",
        type=str,
        default=None,
        metavar="LABEL",
        help="Label used to name output files (e.g. 'chapter' -> chapter_00.mp3).",
    )
    parser.add_argument(
        "--segment-num-min-digits",
        type=int,
        default=None,
        metavar="N",
        help="Minimum number of digits for segment number, zero-padded (e.g. 2 -> chapter_01.mp3, chapter_02.mp3, ...)",
    )
    return parser


def _config_to_settings(config: configparser.ConfigParser) -> dict:
    """Build a settings dict from a config file (CONFIG_SECTION). Invalid values raise ConfigError."""
    settings = {}
    if not config.has_section(CONFIG_SECTION):
        return settings
    for key in config.options(CONFIG_SECTION):
        norm_key = key.lower().replace("-", "_")
        raw = config.get(CONFIG_SECTION, key).strip()
        if norm_key in _TYPE_MAP:
            conv = _TYPE_MAP[norm_key]
            try:
                settings[norm_key] = conv(raw) if conv is not str else raw
            except (ValueError, TypeError) as e:
                raise ConfigError(
                    f"Invalid value for '{key}' in config: {raw!r}. Expected a valid {conv.__name__}."
                ) from e
    return settings


def get_settings(args: argparse.Namespace) -> dict:
    """
    Merge default config (config.ini in cwd), then actual config file, then CLI.
    No defaults or fallbacks; caller must validate that all required settings are present.
    """
    settings = {}
    if DEFAULT_CONFIG_PATH.exists():
        settings.update(_config_to_settings(load_config(DEFAULT_CONFIG_PATH)))
    settings.update(_config_to_settings(load_config(args.config)))
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
    if args.max_padding is not None:
        settings["max_padding"] = args.max_padding
    if args.segment_label is not None:
        settings["segment_label"] = args.segment_label
    if args.segment_num_min_digits is not None:
        settings["segment_num_min_digits"] = args.segment_num_min_digits

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


def split_at_silence_midpoints(
    audio: AudioSegment,
    min_silence_len_ms: int,
    silence_thresh: float,
    max_padding_ms: int,
    seek_step_ms: int = 10,
) -> list[AudioSegment]:
    """
    Split audio at qualified silences so no content is cut off. When a silence
    is longer than 2× max_padding_ms, we keep at most max_padding_ms on each
    side (trimming the middle); otherwise we split at the midpoint.
    """
    silences = detect_silence(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh,
        seek_step=seek_step_ms,
    )
    if not silences:
        return [audio]
    split_points: list[int] = []
    drop_ranges: list[tuple[int, int]] = []
    for start_ms, end_ms in silences:
        dur_ms = end_ms - start_ms
        if dur_ms > 2 * max_padding_ms:
            split_points.append(start_ms + max_padding_ms)
            split_points.append(end_ms - max_padding_ms)
            drop_ranges.append((start_ms + max_padding_ms, end_ms - max_padding_ms))
        else:
            split_points.append((start_ms + end_ms) // 2)
    split_points = sorted(set([0] + split_points + [len(audio)]))
    segments = [
        audio[split_points[i] : split_points[i + 1]]
        for i in range(len(split_points) - 1)
    ]
    # Drop segments that are the trimmed "middle" of a long silence
    drop_set = set(drop_ranges)
    return [
        seg
        for i, seg in enumerate(segments)
        if (split_points[i], split_points[i + 1]) not in drop_set
    ]


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
        if len(acc) < min_len_ms and merged:
            # Final segment is below minimum; merge into previous so no output is under min
            merged[-1] = merged[-1] + acc
        else:
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

    file_size = input_path.stat().st_size
    run_start = time.perf_counter()

    min_silence_ms = sec_to_ms(settings["min_silence_duration"])
    silence_thresh = settings["silence_threshold"]
    min_seg_ms = sec_to_ms(settings["min_segment_length"])
    max_seg_ms = sec_to_ms(settings["max_segment_length"])
    max_padding_ms = sec_to_ms(settings["max_padding"])

    audio = _run_with_elapsed("Loading input file", load_audio, input_path)
    duration_ms = len(audio)
    print(f"  File: {input_path.name}", file=sys.stderr)
    print(f"  Length: {_format_duration_ms(duration_ms)}", file=sys.stderr)
    print(f"  Size: {_format_size_bytes(file_size)}", file=sys.stderr)
    if file_size > LARGE_FILE_NOTE_BYTES:
        print(
            "  Note: Larger files may take several minutes to analyze and export.",
            file=sys.stderr,
        )

    def do_analyze():
        return split_at_silence_midpoints(
            audio,
            min_silence_len_ms=min_silence_ms,
            silence_thresh=silence_thresh,
            max_padding_ms=max_padding_ms,
            seek_step_ms=10,
        )

    chunks = _run_with_elapsed("Analyzing silence", do_analyze)
    print(f"  Built {len(chunks)} segment(s).", file=sys.stderr)

    def do_enforce():
        return enforce_min_max_segments(
            chunks, min_seg_ms, max_seg_ms, min_silence_ms, silence_thresh
        )

    chunks = _run_with_elapsed(
        "Enforcing min/max segment length", do_enforce
    )
    print(f"  Done. {len(chunks)} segment(s).", file=sys.stderr)

    total = len(chunks)
    segment_label = settings["segment_label"]
    min_digits = max(1, int(settings["segment_num_min_digits"]))
    # Always export as MP3 for maximum compatibility, regardless of input format
    print("Exporting segments...", file=sys.stderr)
    out_paths = []
    for i, chunk in enumerate(chunks):
        # Pad to at least min_digits; more digits used when segment count exceeds 10^min_digits (e.g. 100+ with min_digits=2 -> 100, 101, ...)
        num_str = str(i).zfill(min_digits)
        out_name = f"{segment_label}_{num_str}.mp3"
        out_path = output_dir / out_name
        label = f"[{i + 1}/{total}] {out_name}"

        def do_export(ch=chunk, p=out_path):
            ch.export(p, format="mp3", bitrate="192k")

        _export_segment_with_elapsed(label, do_export)
        out_paths.append(out_path)
    total_elapsed = time.perf_counter() - run_start
    print("Success!", file=sys.stderr)
    print(f"  Wrote {total} file(s) to {output_dir}.", file=sys.stderr)
    print(
        f"  Total time elapsed [{_format_elapsed_hh_mm_ss(total_elapsed)}]",
        file=sys.stderr,
    )
    return out_paths


def main() -> int:
    """Entry point: parse args and config, run split, exit."""
    parser = build_parser()
    args = parser.parse_args()

    # If using default config location and it's missing, show a friendly error and help
    if args.config.resolve() == DEFAULT_CONFIG_PATH.resolve() and not args.config.exists():
        print(
            "config.ini not found. Use --config to indicate config file, or use command line arguments.",
            file=sys.stderr,
        )
        parser.print_help(sys.stderr)
        return 1

    try:
        settings = get_settings(args)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    missing = [k for k in _REQUIRED_SETTINGS if k not in settings]
    if missing:
        names = ", ".join(k.replace("_", "-") for k in missing)
        print(
            f"Missing required value(s): {names}. Set in config file or use command-line arguments.",
            file=sys.stderr,
        )
        parser.print_help(sys.stderr)
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
