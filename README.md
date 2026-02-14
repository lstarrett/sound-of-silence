# Sound of Silence

A single-file Python script that splits an input audio file into multiple files based on detected periods of silence.

## Features

- **Silence-based splitting**: Detects silent segments in an audio file and outputs one file per “non-silent” segment.
- **Min/max segment length**: Set minimum and maximum split segment lengths; the script automatically finds and splits on the best moment of silence between those gates.
- **Configurable**: Options can be set via command-line arguments or a configuration file (configparser-style `.ini` or `.conf`).
- **Single-file design**: All logic lives in `silence.py` for easy distribution and use.

## Usage

```bash
# Using command-line arguments (other formats like .wav and .m4a are supported)
python silence.py input.mp3 [options]

# Using a config file (default: default.conf)
python silence.py --config default.conf input.mp3
```

## Configuration

Settings can be provided in a config file (see `default.conf`) or overridden on the command line. Command-line values take precedence over config file values.

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
