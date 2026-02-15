# Sound of Silence

A single-file Python script that splits an input audio file into multiple files based on detected periods of silence and desired output segment length, and exports the results as a sequence of mp3 files.

## Features

- **Silence-based splitting**: Detects silent segments in an audio file and outputs one file per segment, splitting in the middle of each qualified silence so no content is cut off.
- **Min silence length**: Only considers a stretch of silence qualified as a split point if is below the configured amplitude threshold, and longer than the minimum configured length (e.g., quieter than -40dB and at least 2 seconds long)
- **Min/max segment length**: Set minimum and maximum split segment lengths; the script automatically finds and splits on the best moment of silence between those gates. For example, if you are splitting a large audio file and desire that each segment be at least 5 minutes long, but no more than 10 minutes long, min/max segment length settings at those two limits will direct the script to find the best qualified silence to split the file into segments with lengths between those values
- **Output filename formatting**: Set segment label (e.g., "chapter") and segment number minimum digits (e.g. 2), to produce segment filenames like "chapter_01", "chapter_02", and so on.

## Requirements & Dependencies

- Python 3.x (3.13+ needs `audioop-lts` from requirements — the stdlib `audioop` was removed)
- [pydub](https://github.com/jiaaro/pydub) (see `requirements.txt`)
- ffmpeg (for MP3, M4A, and other non-WAV formats)

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Using command-line arguments (other formats like .wav and .m4a are also supported)
python silence.py --help

# Using command-line arguments (other formats like .wav and .m4a are also supported)
python silence.py input.mp3 [options]

# Using a config file (default: config.ini in current working directory)
python silence.py --config path/to/config.ini input.mp3
```

## Configuration

Settings can be provided in a config file (see `config.ini`) or overridden on the command line. Command-line values take precedence over config file values. Key options include `min_silence_duration`, `min_segment_length`, `max_segment_length`, and `max_padding` (max silence kept at split points when a silence is longer than 2× that value).

## License

MIT License — see [LICENSE](LICENSE).
