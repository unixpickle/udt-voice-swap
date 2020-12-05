import argparse

import numpy as np
from tqdm.auto import tqdm

import voice_swap.audio_pca as audio_pca
from voice_swap.data import ChunkDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--num_chunks", type=int, default=100000)
    parser.add_argument("--pca_dim", type=int, default=500)
    parser.add_argument("--output_path", type=str, default="pca_components.npy")
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    if args.num_chunks < args.chunk_size:
        raise ValueError("require more chunks than samples per chunk")

    data = ChunkDataset(
        args.data_dir, args.sample_rate, args.chunk_size, args.num_chunks
    )

    print("Computing PCA...")
    pca_vecs = audio_pca.audio_chunk_pca(tqdm(data), args.pca_dim)

    print("Fixing skew...")
    pca_vecs = audio_pca.audio_chunk_pca_fix_skew(tqdm(data), pca_vecs)

    print(f"Saving to: {args.output_path}")
    np.save(args.output_path, pca_vecs)

    print("Computing MSE...")
    mse = audio_pca.audio_chunk_pca_mse(tqdm(data), pca_vecs)
    print(f"MSE: {mse}")


if __name__ == "__main__":
    main()
