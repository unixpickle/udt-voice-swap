"""
See what PCA does to an audio file.
"""

import argparse

import numpy as np
from tqdm.auto import tqdm

from voice_swap.data import MFCCReader, MFCCWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--num_chunks", type=int, default=50)
    parser.add_argument("--pca_vecs", type=str, default="pca_components.npy")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    pca_vecs = np.load(args.pca_vecs)
    pca_vecs = pca_vecs / np.sqrt(np.sum(pca_vecs ** 2, axis=-1, keepdims=True))

    reader = MFCCReader(args.input_file, args.sample_rate)
    writer = MFCCWriter(args.output_file, args.sample_rate)

    try:
        for _ in tqdm(range(args.num_chunks)):
            chunk = reader.read(args.chunk_size)
            projected = (pca_vecs.T @ (pca_vecs @ chunk[:, None])).flatten()
            writer.write(projected)
    finally:
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
