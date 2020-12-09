import argparse

import numpy as np
from tqdm.auto import tqdm

import voice_swap.audio_pca as audio_pca
from voice_swap.data import ChunkDataset
from voice_swap.ortho_udt import ortho_udt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--num_chunks", type=int, default=200000)
    parser.add_argument("--pca_dim", type=int, default=500)
    parser.add_argument("--no_cycle_check", action="store_true", default=False)
    parser.add_argument("--no_orthogonal", action="store_true", default=False)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--output_path", type=str, default="translation_model.npz")
    parser.add_argument("source_data_dir", type=str)
    parser.add_argument("target_data_dir", type=str)
    args = parser.parse_args()

    if args.num_chunks < args.chunk_size:
        raise ValueError("require more chunks than samples per chunk")

    source_ds, target_ds = [
        process_dataset(args, dd) for dd in [args.source_data_dir, args.target_data_dir]
    ]
    source, target = [ds["pca_data"] for ds in tqdm([source_ds, target_ds])]

    print("Performing orthogonal UDT...")
    matrix = ortho_udt(
        source,
        target,
        verbose=True,
        no_cycle_check=args.no_cycle_check,
        max_iters=args.max_iters,
        orthogonal=not args.no_orthogonal,
    )

    print("Saving full matrix...")
    full_matrix = source_ds["pca"].T @ matrix @ target_ds["pca"]
    np.savez(
        args.output_path,
        udt=full_matrix,
        source_mean=source_ds["mean"],
        target_mean=target_ds["mean"],
        source_pca=source_ds["pca"],
        target_pca=target_ds["pca"],
    )


def process_dataset(args, data_dir):
    print(f"Loading chunks from {data_dir}...")
    data = np.stack(
        list(
            tqdm(
                ChunkDataset(
                    data_dir, args.sample_rate, args.chunk_size, args.num_chunks
                )
            )
        )
    )
    mean = np.mean(data, axis=0)
    data -= mean

    print(f"Computing PCA for {data_dir}...")
    pca_vecs = audio_pca.audio_chunk_pca(tqdm(data), args.pca_dim)

    print("Computing MSE...")
    mse = audio_pca.audio_chunk_pca_mse(tqdm(data), pca_vecs)
    print(f"MSE: {mse}")

    print("Fixing skew...")
    pca_data = data @ pca_vecs.T
    audio_pca.audio_chunk_pca_fix_skew(pca_data, pca_vecs)

    return {"pca_data": pca_data, "pca": pca_vecs, "mean": mean}


if __name__ == "__main__":
    main()
