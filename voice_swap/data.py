"""
Iterate over chunks of audio from audio files.
"""

import os
import random
import subprocess

from librosa.feature import mfcc
from librosa.feature.inverse import mfcc_to_audio
import numpy as np


class ChunkDataset:
    def __init__(self, dir_path, sample_rate, chunk_size, num_chunks):
        self.dir_path = dir_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

        self.paths = []
        for root, _, files in os.walk(dir_path):
            for filename in files:
                path = os.path.join(root, filename)
                if os.path.splitext(path)[1][1:] in [
                    "mp3",
                    "m4a",
                    "m4b",
                    "wav",
                    "flac",
                ]:
                    self.paths.append(path)
        if not len(self.paths):
            raise FileNotFoundError(f"no audio files found in: {dir_path}")
        self.num_chunks = num_chunks

    def __len__(self):
        return self.num_chunks

    def __iter__(self):
        counter = 0
        while counter < self.num_chunks:
            paths = self.paths.copy()
            random.shuffle(paths)
            for path in paths:
                reader = ChunkReader(path, self.sample_rate)
                try:
                    # Random misalignment by up to the chunk size.
                    reader.read(random.randrange(self.chunk_size))

                    while True:
                        chunk = reader.read(self.chunk_size)
                        if chunk is None or len(chunk) < self.chunk_size:
                            break
                        yield chunk
                        counter += 1
                        if counter >= self.num_chunks:
                            return
                finally:
                    reader.close()


class ChunkReader:
    """
    An API for reading chunks of audio samples from an audio file.

    :param path: the path to the audio file.
    :param sample_rate: the number of samples per second, used for resampling.
    """

    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate
        self._done = False

        audio_reader, audio_writer = os.pipe()
        try:
            args = [
                "ffmpeg",
                "-i",
                path,
                "-f",
                "s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "pipe:%i" % audio_writer,
            ]
            self._ffmpeg_proc = subprocess.Popen(
                args,
                pass_fds=(audio_writer,),
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            self._audio_reader = audio_reader
            audio_reader = None
        finally:
            os.close(audio_writer)
            if audio_reader is not None:
                os.close(audio_reader)

        self._reader = os.fdopen(self._audio_reader, "rb")

    def read(self, chunk_size):
        """
        Read a chunk of audio samples from the file.

        :param chunk_size: the number of samples to read.
        :return: A chunk of audio, represented as a 1-D numpy array of floats,
                 where each sample is in the range [-1, 1].
                 When there are no more samples left, None is returned.
        """
        if self._done:
            return None
        buffer_size = chunk_size * 2
        buf = self._reader.read(buffer_size)
        if not len(buf):
            self._done = True
            return None
        res = np.frombuffer(buf, dtype="int16").astype("float32") / (2 ** 15)
        if len(buf) < buffer_size:
            self._done = True
        return res

    def close(self):
        if not self._done:
            self._reader.close()
            self._ffmpeg_proc.wait()
        else:
            self._ffmpeg_proc.wait()
            self._reader.close()


class ChunkWriter:
    """
    An API for writing chunks of audio samples from an audio file.

    :param path: the path to the audio file.
    :param sample_rate: the number of samples per second to write.
    """

    def __init__(self, path, sample_rate):
        self.path = path
        self.sample_rate = sample_rate

        audio_reader, audio_writer = os.pipe()
        try:

            audio_format = ["-ar", str(sample_rate), "-ac", "1", "-f", "s16le"]
            audio_params = audio_format + [
                "-probesize",
                "32",
                "-thread_queue_size",
                "60",
                "-i",
                "pipe:%i" % audio_reader,
            ]
            output_params = [path]
            self._ffmpeg_proc = subprocess.Popen(
                ["ffmpeg", "-y", *audio_params, *output_params],
                pass_fds=(audio_reader,),
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            self._audio_writer = audio_writer
            audio_writer = None
        finally:
            if audio_writer is not None:
                os.close(audio_writer)
            os.close(audio_reader)

        self._writer = os.fdopen(self._audio_writer, "wb", buffering=1024)

    def write(self, chunk):
        """
        Read a chunk of audio samples from the file.

        :param chunk: a chunk of samples, stored as a 1-D numpy array of floats,
                      where each sample is in the range [-1, 1].
        """
        data = bytes((chunk * (2 ** 15 - 1)).astype("int16"))
        self._writer.write(data)

    def close(self):
        self._writer.close()
        self._ffmpeg_proc.wait()


class MFCCReader(ChunkReader):
    """
    A ChunkReader that computes MFCCs for every chunk of audio.
    """

    def read(self, chunk_size):
        buf = super().read(chunk_size)
        fft_size = _best_fft_size(self.sample_rate)
        coeffs = mfcc(
            y=buf,
            sr=self.sample_rate,
            n_mfcc=128,
            n_fft=fft_size,
            hop_length=fft_size // 2,
            n_mels=128,
        )
        return coeffs.flatten()


class MFCCWriter(ChunkWriter):
    """
    A ChunkWriter that decodes MFCCs for every chunk of audio.
    """

    def write(self, chunk):
        fft_size = _best_fft_size(self.sample_rate)
        converted = mfcc_to_audio(
            chunk.reshape([128, -1]),
            sr=self.sample_rate,
            n_fft=fft_size,
            hop_length=fft_size // 2,
            n_mels=128,
        )
        super().write(converted)


def _best_fft_size(sample_rate):
    if sample_rate < 44100:
        return 1024
    else:
        return 2048
