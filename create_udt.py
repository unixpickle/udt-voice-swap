import argparse

import numpy as np
from tqdm.auto import tqdm

from voice_swap.audio_pca import audio_chunks_apply_pca
from voice_swap.data import ChunkDataset
from voice_swap.ortho_udt import ortho_udt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--num_chunks", type=int, default=20000)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--output_path", type=str, default="udt_rotation.npy")
    parser.add_argument("--no_cycle_check", action="store_true", default=False)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("pca_1", type=str)
    parser.add_argument("data_dir_1", type=str)
    parser.add_argument("pca_2", type=str)
    parser.add_argument("data_dir_2", type=str)
    args = parser.parse_args()

    print("Loading source and target datasets...")
    pca_1 = np.load(args.pca_1)
    pca_2 = np.load(args.pca_2)
    source, target = [
        np.stack(
            [
                transformed
                for _, transformed in audio_chunks_apply_pca(
                    tqdm(
                        ChunkDataset(
                            data_dir, args.sample_rate, args.chunk_size, args.num_chunks
                        )
                    ),
                    pca,
                )
            ]
        )
        for pca, data_dir in [(pca_1, args.data_dir_1), (pca_2, args.data_dir_2)]
    ]

    print("Performing orthogonal UDT...")
    matrix = ortho_udt(
        source,
        target,
        verbose=True,
        no_cycle_check=args.no_cycle_check,
        max_iters=args.max_iters,
    )

    print("Saving full matrix...")
    pca_2 /= np.sum(pca_2 ** 2, axis=-1, keepdims=True)
    full_matrix = pca_1.T @ matrix @ pca_2
    np.save(args.output_path, full_matrix)


if __name__ == "__main__":
    main()
