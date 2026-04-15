#!/usr/bin/env python3
"""
Generate a random symmetric distance matrix and save it as TSV.

Useful for manual testing of dist2mst.

Usage:
    python generate_matrix.py 100 matrix_100.tsv
    python generate_matrix.py 500 matrix_500.tsv --max-dist 200 --seed 123
"""

import argparse
import sys

import numpy as np


def generate_and_save(size, output, max_dist=100, seed=None):
    """Generate a symmetric distance matrix and write it as a TSV file."""
    rng = np.random.default_rng(seed)
    upper = rng.integers(0, max_dist, size=(size, size)).astype(np.float64)
    matrix = (upper + upper.T) / 2.0
    np.fill_diagonal(matrix, 0.0)

    names = [f"S{i:06d}" for i in range(size)]

    with open(output, "w") as f:
        # Header row
        f.write("\t" + "\t".join(names) + "\n")
        # Data rows
        for i in range(size):
            row_vals = "\t".join(f"{matrix[i, j]:.1f}" for j in range(size))
            f.write(f"{names[i]}\t{row_vals}\n")

    print(f"Generated {size}x{size} symmetric distance matrix -> {output}")
    print(f"  max-dist: {max_dist}, seed: {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random symmetric distance matrix as TSV."
    )
    parser.add_argument(
        "size",
        type=int,
        help="Number of samples (matrix will be size x size)",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output TSV file path",
    )
    parser.add_argument(
        "--max-dist",
        type=int,
        default=100,
        help="Maximum distance value (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.size < 2:
        print("Error: size must be at least 2", file=sys.stderr)
        sys.exit(1)

    generate_and_save(args.size, args.output, max_dist=args.max_dist, seed=args.seed)


if __name__ == "__main__":
    main()
