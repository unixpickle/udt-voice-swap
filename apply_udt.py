import argparse

import numpy as np
from tqdm.auto import tqdm

from voice_swap.data import MFCCReader, MFCCWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--num_chunks", type=int, default=250)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("udt_matrix", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    print("Loading UDT...")
    matrix = np.load(args.udt_matrix)

    print("Translating...")
    reader = MFCCReader(args.input_file, args.sample_rate)
    writer = MFCCWriter(args.output_file, args.sample_rate)

    try:
        for _ in tqdm(range(args.num_chunks)):
            chunk = reader.read(args.chunk_size)
            chunk_out = (chunk[None] @ matrix).flatten()
            writer.write(chunk_out)
    finally:
        reader.close()
        writer.close()


if __name__ == "__main__":
    main()
