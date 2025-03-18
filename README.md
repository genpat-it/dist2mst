# dist2mst

Version: 0.0.2  
Author: GenPat  
Contact: genpat@izs.it

## Overview

`dist2mst` is a high-performance tool for constructing Minimum Spanning Trees (MSTs) from symmetric distance matrices. The tool uses Numba JIT compilation and parallelization to significantly accelerate tree construction for large datasets. This implementation provides advanced clustering capabilities and subtree extraction.

## Features

- **Numba-accelerated algorithms**: Core numerical functions are JIT-compiled for maximum performance
- **Parallel processing**: Uses multiple CPU cores for faster computation on large matrices
- **Zero-distance grouping**: Special handling for grouping nodes with zero Hamming distance
- **Newick format output**: Produces standard Newick tree format for compatibility with visualization tools
- **Hierarchical clustering**: Identify clusters based on maximum path distance thresholds
- **Targeted clustering**: Find clusters starting from specific samples of interest
- **Cluster filtering**: Include only clusters within specific size ranges
- **NWK file generation**: Export individual Newick files for each eligible cluster
- **Detailed statistics**: Comprehensive metrics about clusters and performance

## Requirements

- Python 3.7+
- NumPy
- Numba
- tqdm
- Pandas

## Installation

### Using Pip

```bash
# Clone the repository
git clone https://github.com/genpat-it/dist2mst.git
cd dist2mst

# Install required packages
pip install -r requirements.txt
```

### Using Docker

A Docker container is available for easy deployment without worrying about dependencies:

```bash
# Build the Docker image
docker build -t dist2mst .

# Run the container with your data
docker run -v /path/to/your/data:/data dist2mst /data/input_matrix.tsv /data/output_tree.nwk
```

The Docker container uses a Conda environment with the following specifications:
- Python 3.10.6
- Numba 0.56.4
- NumPy 1.23.5
- Pandas 1.5.3
- tqdm 4.64.1

## Usage

### Command Line

```bash
python dist2mst.py input_matrix.tsv output_tree.nwk [options]
```

### With Docker

```bash
docker run -v /path/to/your/data:/data dist2mst /data/input_matrix.tsv /data/output_tree.nwk [options]
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `input_file` | TSV file containing the symmetric distance matrix |
| `output_file` | Output file for the Newick tree |
| `--cluster-threshold FLOAT` | Maximum path distance threshold for clusters |
| `--cluster-output FILE` | Output file for cluster data (TSV) |
| `--min-cluster-size INT` | Minimum number of samples required to generate a cluster NWK file |
| `--max-cluster-size INT` | Maximum number of samples for generating a cluster NWK file |
| `--cluster-nwk-dir DIR` | Directory to store individual NWK files for clusters |
| `--samples-of-interest FILE` | File containing sample IDs to use as starting points for clusters |
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

### Samples of Interest File Format

If using the `--samples-of-interest` option, the file should contain one sample ID per line:

```
Sample_A
Sample_B
Sample_C
```

## Algorithm Details

### MST Construction

`dist2mst` implements a modified Prim's algorithm for Minimum Spanning Tree (MST) construction:

1. **Central Node Selection**: The algorithm identifies the central node with minimum sum of distances to all other nodes.

2. **Greedy MST Construction**: The tree expands by always selecting the closest unvisited node to any node already in the tree.

3. **Zero-distance Grouping**: Nodes with zero distance are grouped into a single node in the tree - a feature particularly useful for identical sequences in genomic data.

### Clustering

The tool offers two clustering approaches:

1. **Global Clustering**: Identifies all clusters where the maximum path distance between any two nodes does not exceed the threshold.

2. **Targeted Clustering**: Starts from specified samples of interest and includes all other samples within the threshold distance.

## Examples

### Basic MST Construction

```bash
# Generate an MST from a distance matrix and save as Newick format
python dist2mst.py distance_matrix.tsv output_tree.nwk

# With Docker
docker run -v $(pwd):/data dist2mst /data/distance_matrix.tsv /data/output_tree.nwk
```

### Global Clustering

```bash
# Find all clusters with a maximum path distance of 10
python dist2mst.py distance_matrix.tsv output_tree.nwk --cluster-threshold 10 --cluster-output clusters.tsv

# With Docker
docker run -v $(pwd):/data dist2mst /data/distance_matrix.tsv /data/output_tree.nwk --cluster-threshold 10 --cluster-output /data/clusters.tsv
```

### Targeted Clustering with NWK Generation

```bash
# Find clusters starting from specific samples of interest
# Generate individual NWK files for clusters with 5-50 samples
python dist2mst.py distance_matrix.tsv output_tree.nwk \
  --cluster-threshold 10 \
  --samples-of-interest my_samples.txt \
  --cluster-output clusters.tsv \
  --min-cluster-size 5 \
  --max-cluster-size 50 \
  --cluster-nwk-dir ./cluster_nwk_files

# With Docker
docker run -v $(pwd):/data dist2mst /data/distance_matrix.tsv /data/output_tree.nwk \
  --cluster-threshold 10 \
  --samples-of-interest /data/my_samples.txt \
  --cluster-output /data/clusters.tsv \
  --min-cluster-size 5 \
  --max-cluster-size 50 \
  --cluster-nwk-dir /data/cluster_nwk_files
```

### Performance Optimization

```bash
# Set specific thread count and minimize output
python dist2mst.py distance_matrix.tsv output_tree.nwk --threads 16 --quiet

# With Docker
docker run -v $(pwd):/data dist2mst /data/distance_matrix.tsv /data/output_tree.nwk --threads 16 --quiet
```

## Output Files

### Main Newick Tree

The primary output is a Newick-format tree representing the entire MST.

### Cluster TSV File (optional)

When using `--cluster-output`, a TSV file will be generated with the following columns:
- `Cluster`: Cluster ID (sorted by size, largest first)
- `Samples`: Comma-separated list of sample IDs in the cluster
- `Size`: Number of samples in the cluster
- `NWK_Path`: Path to the individual NWK file (if cluster NWK generation is enabled)

### Individual Cluster NWK Files (optional)

When using `--cluster-nwk-dir` with `--min-cluster-size`, individual Newick files will be generated for each eligible cluster with the naming format:
```
cluster_<size>_<max_distance>_<unique_id>.nwk
```

## Benchmark

Below are performance results from running dist2mst on a large dataset.

**System Specifications:**
```bash
$ lscpu | grep -E "Model name|Socket|Core|Thread"
Thread(s) per core:  1
Core(s) per socket:  48
Socket(s):           4
Model name:          Intel(R) Xeon(R) Gold 6252N CPU @ 2.30GHz

$ uname -a
Linux gtc-collab-int.izs.intra 4.18.0-553.el8_10.x86_64 #1 SMP Fri May 24 08:32:12 EDT 2024 x86_64 x86_64 x86_64 GNU/Linux
```

**Benchmark Results:**
```
$ time python dist2mst.py benchmarks/distance_matrix_5000.tsv 5000.nwk --threads 32
MST Builder v0.0.2
Author: GenPat
Contact: genpat@izs.it
--------------------------------------------------
[+] Running with 32 threads
[+] Numba version: 0.58.0
[+] NumPy version: 1.25.0
[+] Reading distance matrix from benchmarks/distance_matrix_5000.tsv
[+] Loaded 4999x4999 distance matrix in 11.16 seconds
[+] Building MST for 4999 nodes with Numba acceleration
[+] Finding optimal central node...
[+] Selected node Sample_3275 as central node (index 3274)
[+] Building MST tree structure...
Building MST: 100%|█████████████████████████████████████████████████████████████████| 4998/4998 [02:39<00:00, 31.27node/s]
[+] MST construction completed in 161.16 seconds
[+] Added 4999 nodes to tree
[+] Converting tree to Newick format...
[+] Converted to Newick format in 0.01 seconds
[+] Writing Newick tree to 5000.nwk
[+] MST processing completed in 172.33 seconds
[+] Final Newick tree saved to 5000.nwk
[+] MST Builder v0.0.2 (GenPat) completed successfully
[+] Contact: genpat@izs.it
real    2m53.166s
user    2m51.651s
sys     0m15.955s
```

## Visualization

The Newick (.nwk) files generated by dist2mst can be visualized using various tools:

- [SPREAD](https://github.com/genpat-it/spread) - A modern tool for visualizing and exploring trees in Newick format
- [iTOL](https://itol.embl.de/) - Interactive Tree Of Life
- [FigTree](http://tree.bio.ed.ac.uk/software/figtree/) - A graphical viewer of phylogenetic trees

## New in Version 0.0.2

- **Docker support**: Easy deployment with containerized environment
- **Targeted clustering**: Find clusters starting from specific samples of interest
- **Cluster size filtering**: Generate NWK files only for clusters within specific size ranges
- **Individual cluster export**: Export separate Newick files for each eligible cluster
- **Cluster statistics**: Detailed metrics about cluster sizes and distribution
- **Pandas integration**: Improved data loading with Pandas
- **Enhanced debugging**: Better error reporting and diagnostic information

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please contact:
- Email: genpat@izs.it
- GitHub: https://github.com/genpat-it/dist2mst