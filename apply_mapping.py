import argparse

import numpy as np
from tqdm.auto import tqdm

from voice_swap.data import MFCCReader, MFCCWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--num_chunks", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("model_file", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    print("Loading UDT...")
    model = np.load(args.model_file)

    print("Opening audio files...")
    reader = MFCCReader(args.input_file, args.sample_rate)
    writer = MFCCWriter(args.output_file, args.sample_rate)

    print("Translating...")
    try:
        for _ in tqdm(range(args.num_chunks)):
            chunk = reader.read(args.chunk_size)
            chunk -= model["source_mean"]
            chunk_out = (chunk[None] @ model["udt"]).flatten()
            chunk += model["target_mean"]
            writer.write(chunk_out)
    finally:
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
