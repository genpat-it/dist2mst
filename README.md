# dist2nwk

Version: 0.0.1  
Author: GenPat  
Contact: genpat@izs.it

## Overview

MST Builder is a high-performance tool for constructing Minimum Spanning Trees (MSTs) from symmetric distance matrices. The tool uses Numba JIT compilation and parallelization to significantly accelerate tree construction and branch recrafting operations.

## Features

- **Numba-accelerated algorithms**: Core numerical functions are JIT-compiled for maximum performance
- **Parallel processing**: Uses multiple CPU cores for faster computation on large matrices
- **Zero-distance grouping**: Special handling for grouping nodes with zero Hamming distance
- **Branch recrafting**: Optional verification and correction of branch lengths
- **Progress monitoring**: Real-time progress tracking with tqdm progress bars
- **Detailed logging**: Comprehensive timing and performance metrics
- **Newick format output**: Produces standard Newick tree format for compatibility with phylogenetic tools

## Requirements

- Python 3.7+
- NumPy
- Numba
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/genpat-it/dist2nwk.git
cd dist2nwk

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
python dist2nwk.py input_matrix.tsv output_tree.nwk [options]
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | TSV file containing the symmetric distance matrix |
| `output_file` | Output file for the Newick tree |
| `--recraft` | Enable branch recrafting to verify/correct branch lengths |
| `--threads N` | Number of threads to use (0=auto-detect) |
| `--quiet` | Minimize progress output (disable progress bars) |
| `--version` | Show version information and exit |

### Input Format

The input file should be a tab-separated values (TSV) file with:
- First row: Column headers with sample names (first cell can be empty)
- First column: Sample names
- Remaining cells: Distance values between samples

Example:
```
	Sample1	Sample2	Sample3
Sample1	0	0.5	0.8
Sample2	0.5	0	0.6
Sample3	0.8	0.6	0
```

## Algorithm Details

1. **Central Node Selection**: Identifies the optimal starting node with minimum sum of distances
2. **MST Construction**: Builds tree structure by repeatedly adding nearest neighbor nodes
3. **Zero-distance Grouping**: Groups samples with zero distance into a single node
4. **Branch Recrafting** (optional): Verifies and corrects branch lengths by comparing tree distances with input matrix

## Performance Optimizations

- Uses Numba JIT compilation for core numerical functions
- Parallelizes distance calculations with `prange`
- Optimizes memory access patterns for Numba compatibility
- Uses vectorized numpy operations where possible
- Provides thread count control via command line

## Example

```bash
# Basic usage
python dist2nwk.py distance_matrix.tsv output_tree.nwk

# With branch recrafting and 8 threads
python dist2nwk.py distance_matrix.tsv output_tree.nwk --recraft --threads 8

# Quiet mode (minimal output)
python dist2nwk.py distance_matrix.tsv output_tree.nwk --quiet
```

## Output

The tool produces:
1. A Newick format tree file at the specified output path
2. Progress information and timing statistics in the console output

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This is a free and open-source software that allows anyone to:
- Use the software for any purpose
- Study how the software works and modify it
- Redistribute the software
- Improve the software and release your improvements to the public