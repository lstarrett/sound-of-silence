# Sound of Silence

A single-file Python script that splits an input audio file into multiple files based on detected periods of silence.

## Features

- **Silence-based splitting**: Detects silent segments in an audio file and outputs one file per segment, splitting in the middle of each qualified silence so no content is cut off.
- **Min/max segment length**: Set minimum and maximum split segment lengths; the script automatically finds and splits on the best moment of silence between those gates.
- **Max padding**: For silences longer than 2× `max_padding`, only `max_padding` seconds are kept at the end of one segment and the start of the next; the middle of the silence is removed (e.g. 10 s silence with 2 s max padding → 2 s at end of first segment, 2 s at start of second, 6 s removed).
- **Configurable**: Options can be set via command-line arguments or a configuration file (configparser-style `.ini` or `.conf`).
- **Single-file design**: All logic lives in `silence.py` for easy distribution and use.

## Usage

```bash
# Using command-line arguments (other formats like .wav and .m4a are supported)
python silence.py input.mp3 [options]

# Using a config file (default: config.ini)
python silence.py --config config.ini input.mp3
```

## Configuration

Settings can be provided in a config file (see `config.ini`) or overridden on the command line. Command-line values take precedence over config file values. Key options include `min_silence_duration`, `min_segment_length`, `max_segment_length`, and `max_padding` (max silence kept at split points when a silence is longer than 2× that value).

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

Then run the script with `python silence.py input.mp3` (or your audio file). For MP3/M4A and similar formats, [ffmpeg](https://ffmpeg.org/) must be installed on your system; pydub uses it for encoding/decoding.

## Requirements

- Python 3.x (3.13+ needs `audioop-lts` from requirements — the stdlib `audioop` was removed)
- [pydub](https://github.com/jiaaro/pydub) (see `requirements.txt`)
- ffmpeg (for MP3, M4A, and other non-WAV formats)

## License

MIT License — see [LICENSE](LICENSE).
