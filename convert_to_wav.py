"""
Usage: convert_to_wav.py <file or directory name>

Converts a file or directory of files from mp3 to mono wav and saves them to
the same location. This script requires ffmpeg to be installed.
"""

from pydub import AudioSegment
import os
import sys

if len(sys.argv) < 2:
    print "Usage: python convert_to_wav.py <file or directory name>"
    sys.exit(0)

filename = sys.argv[1]

if os.path.isdir(filename):
    for file in [os.path.join(filename, f) for f in os.listdir(filename)]:
        _, filetype = os.path.splitext(file)
        if filetype == ".mp3":
            song = AudioSegment.from_mp3(file).set_channels(1)
            song.export(file.rsplit('.', 1)[0] + ".wav", format='wav')
else:
    _, filetype = os.path.splitext(file)
    if filetype == ".mp3":
        song = AudioSegment.from_mp3(filename).set_channels(1)
        song.export(filename.rsplit('.', 1)[0] + ".wav", format='wav')