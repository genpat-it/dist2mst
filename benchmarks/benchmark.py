#!/usr/bin/env python3
"""
Robust benchmark suite for dist2mst.

Tests performance AND correctness across realistic matrix topologies:
- Random uniform distances
- Clustered data (simulating outbreaks with intra/inter-cluster distances)
- Sparse zeros (many identical sequences, common in cgMLST)
- Skewed distributions (right-skewed, realistic for genomic distances)
- Pathological cases (all equal, near-zero, single large cluster)

Validates MST correctness against scipy reference implementation.

Usage:
    python benchmark.py                     # Full suite
    python benchmark.py --quick             # Quick smoke test
    python benchmark.py --sizes 100 500     # Custom sizes
    python benchmark.py --correctness-only  # Only run validation
"""

import argparse
import os
import sys
import time
import tracemalloc
import statistics

import numpy as np

# Add parent directory so we can import dist2mst
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist2mst import (
    build_mst_accelerated,
    find_clusters,
    find_central_node,
    find_closest_pair,
)


# ---------------------------------------------------------------------------
# Matrix generators — each simulates a different real-world scenario
# ---------------------------------------------------------------------------

def gen_random_uniform(n, seed=42):
    """Random uniform distances in [1, 100]. Baseline / worst-case for MST."""
    rng = np.random.default_rng(seed)
    upper = rng.integers(1, 101, size=(n, n)).astype(np.float64)
    sym = (upper + upper.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    return sym


def gen_clustered(n, n_clusters=None, seed=42):
    """
    Clustered topology simulating outbreak scenarios.
    Intra-cluster distances are small (0-5), inter-cluster are large (50-100).
    This is the most realistic scenario for cgMLST surveillance.
    """
    rng = np.random.default_rng(seed)
    if n_clusters is None:
        n_clusters = max(2, n // 20)  # ~20 samples per cluster

    labels = np.array([i % n_clusters for i in range(n)])
    rng.shuffle(labels)

    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                d = rng.integers(0, 6)  # intra-cluster: 0-5 SNPs
            else:
                d = rng.integers(50, 101)  # inter-cluster: 50-100 SNPs
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def gen_sparse_zeros(n, zero_fraction=0.3, seed=42):
    """
    Many zero distances (identical sequences). Common in cgMLST where
    multiple isolates from the same source share identical profiles.
    Tests the zero-distance grouping logic.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.integers(1, 50, size=(n, n)).astype(np.float64)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)

    # Set ~zero_fraction of off-diagonal pairs to zero
    n_zeros = int(n * (n - 1) / 2 * zero_fraction)
    indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    zero_idx = rng.choice(len(indices), size=min(n_zeros, len(indices)), replace=False)
    for idx in zero_idx:
        i, j = indices[idx]
        matrix[i, j] = 0.0
        matrix[j, i] = 0.0
    return matrix


def gen_skewed(n, seed=42):
    """
    Right-skewed distance distribution (exponential-like).
    Most pairs are close (1-10), few are far (up to 200).
    Realistic for within-species genomic distances.
    """
    rng = np.random.default_rng(seed)
    upper = rng.exponential(scale=8.0, size=(n, n))
    upper = np.clip(upper, 0, 200).astype(np.float64)
    sym = (upper + upper.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    # Round to integers (SNP counts are discrete)
    sym = np.round(sym)
    return sym


def gen_all_equal(n, value=10.0):
    """All distances equal. Pathological case — MST is arbitrary."""
    matrix = np.full((n, n), value, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def gen_star(n, seed=42):
    """
    Star topology: one central node close to all others, others far from each other.
    Tests that the algorithm correctly identifies the hub.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.integers(80, 101, size=(n, n)).astype(np.float64)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    # Node 0 is the hub — close to everyone
    for i in range(1, n):
        d = float(rng.integers(1, 6))
        matrix[0, i] = d
        matrix[i, 0] = d
    return matrix


TOPOLOGIES = {
    "random_uniform": gen_random_uniform,
    "clustered": gen_clustered,
    "sparse_zeros": gen_sparse_zeros,
    "skewed": gen_skewed,
    "all_equal": gen_all_equal,
    "star": gen_star,
}


# ---------------------------------------------------------------------------
# Correctness validation against scipy
# ---------------------------------------------------------------------------

def _kruskal_mst_weight(matrix):
    """
    Reference MST weight using Kruskal's algorithm with Union-Find.
    Handles zero-weight edges correctly (unlike scipy which ignores them).
    """
    n = matrix.shape[0]

    # Collect all edges from upper triangle
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((matrix[i, j], i, j))
    edges.sort()

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    total_weight = 0.0
    edges_used = 0
    for w, u, v in edges:
        if union(u, v):
            total_weight += w
            edges_used += 1
            if edges_used == n - 1:
                break

    return total_weight


def validate_mst_weight(matrix, names, root):
    """
    Validate that dist2mst produces an MST with total weight equal to (or
    very close to) a reference Kruskal's implementation.
    Returns (is_valid, dist2mst_weight, reference_weight, note).
    """
    n = len(names)

    # dist2mst weight from tree walk
    dist2mst_weight = _tree_weight(root)

    # Reference MST weight via Kruskal's (handles zero-weight edges correctly)
    ref_weight = _kruskal_mst_weight(matrix)

    # Allow small tolerance for float rounding
    tol = max(1e-6 * max(abs(ref_weight), abs(dist2mst_weight), 1.0), 1e-6)
    is_valid = abs(dist2mst_weight - ref_weight) <= tol

    note = ""
    has_zeros = np.any((matrix == 0) & ~np.eye(n, dtype=bool))
    if has_zeros:
        note = "has zero-edges"

    return is_valid, dist2mst_weight, ref_weight, note


def _tree_weight(node):
    """Recursively sum all edge weights in the tree."""
    total = 0.0
    for child, distance in node.children:
        total += distance + _tree_weight(child)
    return total


def _count_samples(node):
    """Recursively count all samples in the tree."""
    count = len(node.samples)
    for child, _ in node.children:
        count += _count_samples(child)
    return count


def validate_mst_structure(matrix, names, root):
    """
    Validate structural properties of the MST:
    1. All samples present in the tree
    2. Number of edges = n - 1 (for a tree)
    3. No negative edge weights
    """
    n = len(names)
    errors = []

    # Check all samples are present
    tree_samples = set()
    _collect_samples(root, tree_samples)
    name_set = set(names)
    missing = name_set - tree_samples
    extra = tree_samples - name_set
    if missing:
        errors.append(f"Missing {len(missing)} samples from tree")
    if extra:
        errors.append(f"{len(extra)} extra samples in tree")

    # Check sample count
    sample_count = _count_samples(root)
    if sample_count != n:
        errors.append(f"Tree has {sample_count} samples, expected {n}")

    # Check for negative weights
    neg = _check_negative_weights(root)
    if neg:
        errors.append(f"Found {neg} negative edge weights")

    return errors


def _collect_samples(node, sample_set):
    """Recursively collect all sample names from the tree."""
    for s in node.samples:
        sample_set.add(s)
    for child, _ in node.children:
        _collect_samples(child, sample_set)


def _check_negative_weights(node):
    """Count negative edge weights."""
    count = 0
    for child, dist in node.children:
        if dist < 0:
            count += 1
        count += _check_negative_weights(child)
    return count


# ---------------------------------------------------------------------------
# Benchmarking functions
# ---------------------------------------------------------------------------

def make_names(n):
    return [f"S{i:06d}" for i in range(n)]


def warmup():
    """Run a small matrix through JIT-compiled functions to trigger compilation."""
    print("[warmup] Compiling Numba functions ...")
    mat = gen_random_uniform(10, seed=0)
    names = make_names(10)
    find_central_node(mat)
    used = np.zeros(10, dtype=np.bool_)
    used[0] = True
    find_closest_pair(mat, used, 10)
    build_mst_accelerated(mat, names)
    print("[warmup] Done.\n")


def bench_mst(matrix, names):
    """Benchmark MST construction. Returns (wall_seconds, peak_memory_bytes, root)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    root = build_mst_accelerated(matrix, names)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, root


def bench_clustering(root, threshold=10.0):
    """Benchmark clustering. Returns (wall_seconds, peak_memory_bytes, n_clusters)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    clusters, _ = find_clusters(root, threshold)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, len(clusters)


def fmt_mem(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KiB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024**2:.1f} MiB"
    else:
        return f"{bytes_val / 1024**3:.2f} GiB"


# ---------------------------------------------------------------------------
# Correctness test suite
# ---------------------------------------------------------------------------

def run_correctness_tests(topologies=None):
    """Run correctness validation across all topologies and edge cases."""
    if topologies is None:
        topologies = TOPOLOGIES
    print("=" * 70)
    print("  CORRECTNESS VALIDATION")
    print("=" * 70)

    has_scipy = True
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: F401
    except ImportError:
        has_scipy = False
        print("[!] scipy not installed — skipping MST weight validation")
        print("[!]   Install with: pip install scipy")

    test_sizes = [10, 50, 100, 200]
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for topo_name, gen_func in topologies.items():
        for n in test_sizes:
            label = f"{topo_name} n={n}"
            print(f"  [{label}] ", end="", flush=True)

            names = make_names(n)
            try:
                if topo_name == "all_equal":
                    matrix = gen_func(n)
                else:
                    matrix = gen_func(n, seed=42)
            except TypeError:
                matrix = gen_func(n)

            # Build MST
            # Suppress dist2mst print output during validation
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                root = build_mst_accelerated(matrix, names)

            # Structural validation
            struct_errors = validate_mst_structure(matrix, names, root)
            if struct_errors:
                print(f"FAIL (structure: {'; '.join(struct_errors)})")
                failed += 1
                failures.append((label, struct_errors))
                continue

            # Weight validation (scipy)
            if has_scipy:
                is_valid, d2m_w, scipy_w, note = validate_mst_weight(matrix, names, root)
                if is_valid is None:
                    print(f"SKIP ({note})")
                    skipped += 1
                elif is_valid:
                    extra = f" [{note}]" if note else ""
                    print(f"OK (weight: {d2m_w:.1f}, scipy: {scipy_w:.1f}){extra}")
                    passed += 1
                else:
                    extra = f" [{note}]" if note else ""
                    print(f"FAIL (weight: {d2m_w:.1f} != scipy: {scipy_w:.1f}){extra}")
                    failed += 1
                    failures.append((label, [f"weight mismatch: {d2m_w:.1f} vs {scipy_w:.1f} {note}"]))
            else:
                # At least structural validation passed
                print("OK (structure only)")
                passed += 1

    # Edge cases
    print(f"\n  --- Edge Cases ---")

    # Single node
    label = "single_node n=1"
    print(f"  [{label}] ", end="", flush=True)
    matrix_1 = np.zeros((1, 1), dtype=np.float64)
    names_1 = ["S000000"]
    f = io.StringIO()
    with redirect_stdout(f):
        root_1 = build_mst_accelerated(matrix_1, names_1)
    errs = validate_mst_structure(matrix_1, names_1, root_1)
    if errs:
        print(f"FAIL ({'; '.join(errs)})")
        failed += 1
        failures.append((label, errs))
    else:
        print("OK")
        passed += 1

    # Two identical nodes (distance=0)
    label = "two_identical n=2"
    print(f"  [{label}] ", end="", flush=True)
    matrix_2 = np.array([[0.0, 0.0], [0.0, 0.0]])
    names_2 = ["A", "B"]
    f = io.StringIO()
    with redirect_stdout(f):
        root_2 = build_mst_accelerated(matrix_2, names_2)
    errs = validate_mst_structure(matrix_2, names_2, root_2)
    if errs:
        print(f"FAIL ({'; '.join(errs)})")
        failed += 1
        failures.append((label, errs))
    else:
        print("OK")
        passed += 1

    # All zeros
    label = "all_zeros n=50"
    print(f"  [{label}] ", end="", flush=True)
    matrix_z = np.zeros((50, 50), dtype=np.float64)
    names_z = make_names(50)
    f = io.StringIO()
    with redirect_stdout(f):
        root_z = build_mst_accelerated(matrix_z, names_z)
    errs = validate_mst_structure(matrix_z, names_z, root_z)
    if errs:
        print(f"FAIL ({'; '.join(errs)})")
        failed += 1
        failures.append((label, errs))
    else:
        print("OK")
        passed += 1

    # Very large distances
    label = "large_distances n=50"
    print(f"  [{label}] ", end="", flush=True)
    rng = np.random.default_rng(99)
    matrix_l = rng.integers(1_000_000, 10_000_000, size=(50, 50)).astype(np.float64)
    matrix_l = (matrix_l + matrix_l.T) / 2.0
    np.fill_diagonal(matrix_l, 0.0)
    names_l = make_names(50)
    f = io.StringIO()
    with redirect_stdout(f):
        root_l = build_mst_accelerated(matrix_l, names_l)
    errs = validate_mst_structure(matrix_l, names_l, root_l)
    if errs:
        print(f"FAIL ({'; '.join(errs)})")
        failed += 1
        failures.append((label, errs))
    else:
        if has_scipy:
            is_valid, d2m_w, scipy_w, note = validate_mst_weight(matrix_l, names_l, root_l)
            if is_valid:
                print(f"OK (weight: {d2m_w:.0f})")
                passed += 1
            else:
                print(f"FAIL (weight: {d2m_w:.0f} vs scipy: {scipy_w:.0f})")
                failed += 1
                failures.append((label, [f"weight mismatch"]))
        else:
            print("OK (structure only)")
            passed += 1

    print(f"\n{'=' * 70}")
    print(f"  CORRECTNESS RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print(f"\n  FAILURES:")
        for label, errs in failures:
            print(f"    - {label}: {'; '.join(errs)}")
    print(f"{'=' * 70}\n")

    return failed == 0


# ---------------------------------------------------------------------------
# Performance benchmark suite
# ---------------------------------------------------------------------------

def run_performance_benchmarks(sizes, repeats, topologies=None):
    """Run performance benchmarks across all topologies."""
    if topologies is None:
        topologies = TOPOLOGIES
    results = []

    for topo_name, gen_func in topologies.items():
        print(f"\n{'#' * 70}")
        print(f"  Topology: {topo_name}")
        print(f"{'#' * 70}")

        for n in sizes:
            print(f"\n{'=' * 60}")
            print(f"  {topo_name} | {n} x {n} | {repeats} repeats")
            print(f"{'=' * 60}")

            mst_times = []
            mst_mems = []
            clust_times = []
            clust_mems = []
            clust_counts = []

            for r in range(repeats):
                seed = 42 + r
                try:
                    if topo_name == "all_equal":
                        matrix = gen_func(n)
                    else:
                        matrix = gen_func(n, seed=seed)
                except TypeError:
                    matrix = gen_func(n)
                names = make_names(n)

                # MST
                t, m, root = bench_mst(matrix, names)
                mst_times.append(t)
                mst_mems.append(m)
                print(f"  [rep {r+1}/{repeats}] MST: {t:.4f}s  mem: {fmt_mem(m)}")

                # Clustering
                ct, cm, nc = bench_clustering(root, threshold=10.0)
                clust_times.append(ct)
                clust_mems.append(cm)
                clust_counts.append(nc)
                print(f"  [rep {r+1}/{repeats}] Clustering: {ct:.4f}s  mem: {fmt_mem(cm)}  clusters: {nc}")

            results.append({
                "topology": topo_name,
                "size": n,
                "repeats": repeats,
                "mst_time_mean": statistics.mean(mst_times),
                "mst_time_std": statistics.stdev(mst_times) if repeats > 1 else 0.0,
                "mst_peak_mem_bytes": statistics.mean(mst_mems),
                "cluster_time_mean": statistics.mean(clust_times),
                "cluster_time_std": statistics.stdev(clust_times) if repeats > 1 else 0.0,
                "cluster_peak_mem_bytes": statistics.mean(clust_mems),
                "cluster_count_mean": statistics.mean(clust_counts),
            })

    return results


def analyze_scaling(results):
    """Analyze computational complexity from benchmark results."""
    print(f"\n{'=' * 70}")
    print("  SCALING ANALYSIS")
    print(f"{'=' * 70}")

    # Group by topology
    topos = {}
    for r in results:
        topos.setdefault(r["topology"], []).append(r)

    for topo, rows in topos.items():
        if len(rows) < 2:
            continue
        rows.sort(key=lambda x: x["size"])
        print(f"\n  {topo}:")
        for i in range(1, len(rows)):
            n1, t1 = rows[i-1]["size"], rows[i-1]["mst_time_mean"]
            n2, t2 = rows[i]["size"], rows[i]["mst_time_mean"]
            if t1 > 0 and n1 > 0:
                size_ratio = n2 / n1
                time_ratio = t2 / t1 if t1 > 0 else float("inf")
                # For O(n^2): time_ratio should be ~size_ratio^2
                if time_ratio > 0:
                    import math
                    exponent = math.log(time_ratio) / math.log(size_ratio) if size_ratio > 1 else 0
                    print(f"    {n1:>5} -> {n2:>5}: time {t1:.4f}s -> {t2:.4f}s  "
                          f"(x{time_ratio:.1f}, exponent ~{exponent:.2f})")

    print(f"\n  Expected: exponent ~2.0 for O(n^2) Prim's algorithm")
    print(f"{'=' * 70}")


def print_summary(results):
    """Print formatted summary table."""
    header = (
        f"{'Topology':>16}  "
        f"{'Size':>6}  "
        f"{'MST time (s)':>20}  "
        f"{'MST mem':>12}  "
        f"{'Clust time (s)':>20}  "
        f"{'Clust mem':>12}  "
        f"{'Clusters':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("PERFORMANCE SUMMARY")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        mst_str = f"{r['mst_time_mean']:.4f} +/- {r['mst_time_std']:.4f}"
        clust_str = f"{r['cluster_time_mean']:.4f} +/- {r['cluster_time_std']:.4f}"
        print(
            f"{r['topology']:>16}  "
            f"{r['size']:>6}  "
            f"{mst_str:>20}  "
            f"{fmt_mem(r['mst_peak_mem_bytes']):>12}  "
            f"{clust_str:>20}  "
            f"{fmt_mem(r['cluster_peak_mem_bytes']):>12}  "
            f"{r['cluster_count_mean']:>8.0f}"
        )

    print(sep)


def write_tsv(results, path):
    """Write results to TSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    columns = [
        "topology", "size", "repeats",
        "mst_time_mean", "mst_time_std", "mst_peak_mem_bytes",
        "cluster_time_mean", "cluster_time_std", "cluster_peak_mem_bytes",
        "cluster_count_mean",
    ]
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in results:
            vals = [str(r[c]) for c in columns]
            f.write("\t".join(vals) + "\n")
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robust benchmark suite for dist2mst — tests performance and correctness "
                    "across realistic matrix topologies."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000],
        help="Matrix sizes to benchmark (default: 100 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Repetitions per size (default: 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: sizes 100,500,1000 x 1 repeat, correctness on small matrices",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.tsv"),
        help="TSV output path (default: benchmarks/results.tsv)",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip Numba JIT warmup",
    )
    parser.add_argument(
        "--correctness-only", action="store_true",
        help="Only run correctness validation, skip performance benchmarks",
    )
    parser.add_argument(
        "--topologies", type=str, nargs="+", default=None,
        choices=list(TOPOLOGIES.keys()),
        help="Run only specific topologies (default: all)",
    )
    args = parser.parse_args()

    if args.quick:
        args.sizes = [100, 500, 1000]
        args.repeats = 1

    print("dist2mst benchmark suite (robust)")
    print(f"  Sizes:      {args.sizes}")
    print(f"  Repeats:    {args.repeats}")
    print(f"  Topologies: {args.topologies or 'all'}")
    print(f"  Output:     {args.output}")
    print()

    # Filter topologies if requested
    active_topologies = TOPOLOGIES
    if args.topologies:
        active_topologies = {k: v for k, v in TOPOLOGIES.items() if k in args.topologies}

    if not args.no_warmup:
        warmup()

    # Always run correctness tests
    print("\n" + "=" * 70)
    correctness_ok = run_correctness_tests(active_topologies)

    if args.correctness_only:
        sys.exit(0 if correctness_ok else 1)

    # Performance benchmarks
    results = run_performance_benchmarks(args.sizes, args.repeats, active_topologies)
    print_summary(results)
    analyze_scaling(results)
    write_tsv(results, args.output)

    if not correctness_ok:
        print("\n[!!!] CORRECTNESS TESTS FAILED — benchmark results may be unreliable")
        sys.exit(1)


if __name__ == "__main__":
    main()
