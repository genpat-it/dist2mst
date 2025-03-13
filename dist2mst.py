import numpy as np
import time
from collections import OrderedDict
from numba import njit, prange
from tqdm import tqdm
import pandas as pd

"""
dist2mst
========
Version: 0.0.2
Author: GenPat
Contact: genpat@izs.it
"""

class Node:
    def __init__(self, samples=None):
        self.samples = samples if samples is not None else []
        self.children = []  # List of (Node, distance)
        
    def add_child(self, node, distance):
        self.children.append((node, distance))
        
    def to_newick(self):
        if not self.children:
            return f"({','.join(self.samples)})" if len(self.samples) > 1 else self.samples[0]
        
        child_strings = []
        for child, distance in sorted(self.children, key=lambda x: x[1]):
            child_str = child.to_newick()
            child_strings.append(f"{child_str}:{distance}")
        
        if self.samples:
            return f"({','.join(self.samples + [s for s in child_strings])})"
        return f"({','.join(child_strings)})"

def read_distance_matrix(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', index_col=0)
    matrix_np = df.values.astype(float)
    names = list(df.columns)
    return matrix_np, names

def read_samples_of_interest(file_path):
    """
    Read a list of sample IDs from a text file, one per line.
    """
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            sample = line.strip()
            if sample:  # Skip empty lines
                samples.append(sample)
    return samples

@njit
def find_central_node(matrix):
    """
    Find the central node (with minimum sum of distances)
    """
    dist_sums = np.sum(matrix, axis=0)
    return np.argmin(dist_sums)

@njit
def find_closest_pair(matrix, used_arr, n):
    """
    Find the closest unused node to any used node
    """
    min_dist = np.inf
    closest_i = -1
    closest_j = -1
    
    for i in range(n):
        if not used_arr[i]:
            for j in range(n):
                if used_arr[j] and matrix[i][j] < min_dist:
                    min_dist = matrix[i][j]
                    closest_i = i
                    closest_j = j
    
    return closest_i, closest_j, min_dist

@njit(parallel=True)
def calculate_distance_sums(matrix):
    """
    Calculate the sum of distances for each node in parallel
    """
    n = matrix.shape[0]
    dist_sums = np.zeros(n)
    
    for i in prange(n):
        dist_sum = 0.0
        for j in range(n):
            dist_sum += matrix[i][j]
        dist_sums[i] = dist_sum
        
    return dist_sums

def build_mst_accelerated(matrix, names):
    """
    Build MST with numba acceleration
    """
    
    idx1 = names.index("Lm_0199")
    idx2 = names.index("Lm_0686")
    print(f"[DEBUG] In build_mst_accelerated - Distance between Lm_0199 and Lm_0686: {matrix[idx1][idx2]}")
    print(f"[DEBUG] Matrix data type: {matrix.dtype}")
    
    print(f"[+] Building MST for {len(names)} nodes with Numba acceleration")
    start_time = time.time()
    n = len(names)
    
    # Convert to numpy array for numba compatibility
    used_arr = np.zeros(n, dtype=np.bool_)
    
    # Find central node using accelerated function
    print("[+] Finding optimal central node...")
    center = find_central_node(matrix)
    print(f"[+] Selected node {names[center]} as central node (index {center})")
    
    # Create root node and node map (can't be jitted directly)
    root = Node([names[center]])
    node_map = {}  # Map from indices to nodes
    node_map[center] = root
    used_arr[center] = True
    
    # Main loop for building MST
    used_count = 1
    print(f"[+] Building MST tree structure...")
    progress_bar = tqdm(total=n-1, desc="Building MST", unit="node")
    while used_count < n:
        # Find closest pair using accelerated function
        i, j, min_dist = find_closest_pair(matrix, used_arr, n)
        
        if i == -1:  # No more pairs found
            break
            
        used_arr[i] = True
        used_count += 1
        progress_bar.update(1)
        
        if used_count % max(1, n//10) == 0:
            print(f"[+] Added {used_count}/{n} nodes to MST ({(used_count/n)*100:.1f}%)")

        
        # Group nodes only if Hamming distance is 0
        if min_dist == 0:
            parent_node = node_map[j]
            if len(parent_node.samples) == 1:
                parent_node.samples.append(names[i])
                node_map[i] = parent_node
            else:
                new_node = Node([names[i]])
                parent_node.add_child(new_node, min_dist)
                node_map[i] = new_node
        else:
            parent_node = node_map[j]
            new_node = Node([names[i]])
            parent_node.add_child(new_node, min_dist)
            node_map[i] = new_node
    
    progress_bar.close()
    
    elapsed = time.time() - start_time
    print(f"[+] MST construction completed in {elapsed:.2f} seconds")
    print(f"[+] Added {used_count} nodes to tree")
    
    return root

def build_graph_from_mst(root):
    """
    Build a graph representation of the MST to work with path distances.
    Returns:
        - graph: Dict mapping sample -> list of (neighbor, distance) tuples
        - sample_to_node: Dict mapping sample -> Node object containing it
    """
    graph = OrderedDict()  # Sample name -> list of neighboring samples with distances
    sample_to_node = OrderedDict()  # Sample name -> Node object
    
    # Populate the graph and sample_to_node mapping
    def build_graph_helper(node, parent=None, parent_distance=0):
        for sample in sorted(node.samples):  # Sort for determinism
            if sample not in graph:
                graph[sample] = []
                sample_to_node[sample] = node
            
            # If we have a parent, add an edge to each sample in parent
            if parent:
                for parent_sample in sorted(parent.samples):  # Sort for determinism
                    graph[sample].append((parent_sample, parent_distance))
                    graph[parent_sample].append((sample, parent_distance))
        
        # Process children in a deterministic order
        for child, distance in sorted(node.children, key=lambda x: tuple(sorted(x[0].samples))):
            build_graph_helper(child, node, distance)
    
    build_graph_helper(root)
    return graph, sample_to_node

def find_clusters_from_samples(mst_root, samples_of_interest, threshold, min_cluster_size=None, max_cluster_size=None):
    """
    Find clusters starting from specified samples of interest.
    Each cluster contains the sample of interest and all samples within threshold distance.
    """
    print(f"[+] Finding clusters from {len(samples_of_interest)} samples of interest with maximum path distance threshold of {threshold}...")
    start_time = time.time()
    
    # Build graph representation of the MST
    graph, sample_to_node = build_graph_from_mst(mst_root)
    
    # Verify samples of interest exist in the graph
    valid_samples = []
    for sample in sorted(samples_of_interest):  # Sort for determinism
        if sample in graph:
            valid_samples.append(sample)
        else:
            print(f"[!] Warning: Sample {sample} not found in the MST, skipping...")
    
    if not valid_samples:
        print(f"[!] Error: None of the specified samples of interest were found in the MST.")
        return [], []
    
    # Find clusters for each sample of interest using BFS
    clusters = []
    soi_cluster_map = {}  # Map SOI to its cluster index
    
    for start_sample in valid_samples:
        cluster = []
        queue = [(start_sample, 0)]  # (sample, distance from start)
        distances = {start_sample: 0}
        visited = set()
        
        while queue:
            # Process queue in deterministic order
            queue.sort(key=lambda x: (x[1], x[0]))  # Sort by distance, then by sample name
            sample, dist = queue.pop(0)
            
            # Skip if already visited
            if sample in visited:
                continue
                
            # Add to cluster and mark as visited
            cluster.append(sample)
            visited.add(sample)
            
            # Check all neighbors in a deterministic order
            neighbors = sorted(graph[sample])  # Sort for determinism
            for neighbor, edge_dist in neighbors:
                if neighbor in visited:
                    continue
                    
                # Calculate new distance from start
                new_dist = dist + edge_dist
                
                # If new_dist <= threshold, add to queue
                if new_dist <= threshold:
                    queue.append((neighbor, new_dist))
                    distances[neighbor] = new_dist
        
        if cluster:
            # Store this cluster
            cluster = sorted(cluster)  # Sort for determinism
            clusters.append(cluster)
            # Map this SOI to the cluster index
            soi_cluster_map[start_sample] = len(clusters) - 1
    
    # Check for SOIs that will be discarded due to size constraints
    discarded_sois = []
    for soi, cluster_idx in soi_cluster_map.items():
        cluster_size = len(clusters[cluster_idx])
        if min_cluster_size is not None and cluster_size < min_cluster_size:
            discarded_sois.append((soi, cluster_size, "too small"))
        elif max_cluster_size is not None and cluster_size > max_cluster_size:
            discarded_sois.append((soi, cluster_size, "too large"))
    
    # Print warning for SOIs that will be discarded
    if discarded_sois:
        print(f"[!] Warning: The following samples of interest will be discarded because their clusters do not meet size criteria:")
        for soi, size, reason in discarded_sois:
            print(f"    - {soi}: Cluster size {size} is {reason}")
    
    # Extract subtrees for each cluster
    cluster_trees = []
    for cluster_samples in clusters:
        # For each cluster, create a subtree that preserves original structure
        subtree = extract_subtree(mst_root, set(cluster_samples))
        if subtree:
            cluster_trees.append(subtree)
        else:
            # Create a fallback tree if extraction fails
            fallback_tree = create_fallback_tree(cluster_samples)
            cluster_trees.append(fallback_tree)
    
    elapsed = time.time() - start_time
    print(f"[+] Found {len(clusters)} clusters starting from samples of interest in {elapsed:.2f} seconds")
    
    return clusters, cluster_trees

def find_clusters(mst_root, threshold):
    """
    Find clusters in the MST where the maximum path distance between any two nodes
    in the same cluster does not exceed the threshold.
    
    Returns:
        A list of clusters (each is a list of sample names) and
        a list of subtrees for each cluster
    """
    print(f"[+] Finding clusters with maximum path distance threshold of {threshold}...")
    start_time = time.time()
    
    # Build graph representation of the MST
    graph, sample_to_node = build_graph_from_mst(mst_root)
    
    # Find all clusters using BFS
    visited = set()
    clusters = []
    
    # Process samples in a deterministic order
    samples = sorted(graph.keys())
    for start_sample in samples:
        if start_sample in visited:
            continue
            
        # BFS to find all samples within threshold distance
        cluster = []
        queue = [(start_sample, 0)]  # (sample, distance from start)
        cluster_visited = {start_sample: 0}  # Sample -> distance from start
        
        while queue:
            # Process queue in deterministic order
            queue.sort(key=lambda x: (x[1], x[0]))  # Sort by distance, then by sample name
            sample, _ = queue.pop(0)
            
            # Add to cluster and mark as globally visited
            cluster.append(sample)
            visited.add(sample)
            
            # Check all neighbors in a deterministic order
            neighbors = sorted(graph[sample])  # Sort for determinism
            for neighbor, edge_dist in neighbors:
                if neighbor in visited:
                    continue
                
                # Find shortest path distance from start_sample to neighbor
                min_dist = float('inf')
                for s, dist in sorted(graph[neighbor]):  # Sort for determinism
                    if s in cluster_visited:
                        new_dist = cluster_visited[s] + dist
                        min_dist = min(min_dist, new_dist)
                
                # If min_dist <= threshold, add to queue
                if min_dist <= threshold:
                    queue.append((neighbor, min_dist))
                    cluster_visited[neighbor] = min_dist
        
        if cluster:
            clusters.append(sorted(cluster))  # Sort for determinism
    
    # Extract subtrees for each cluster
    cluster_trees = []
    for i, cluster_samples in enumerate(clusters):
        # Create a subtree containing only the nodes in this cluster
        subtree = extract_subtree(mst_root, set(cluster_samples))
        if subtree:
            cluster_trees.append(subtree)
        else:
            # If extraction fails, create a fallback tree
            fallback_tree = create_fallback_tree(cluster_samples)
            cluster_trees.append(fallback_tree)
    
    elapsed = time.time() - start_time
    print(f"[+] Found {len(clusters)} clusters in {elapsed:.2f} seconds")
    
    return clusters, cluster_trees

def extract_subtree(root, cluster_samples):
    """
    Extract a subtree from the MST containing only the samples in cluster_samples.
    This preserves the original structure and distances of the MST.
    
    Args:
        root: Root node of the MST
        cluster_samples: Set of sample names to include in the subtree
        
    Returns:
        A new tree containing only nodes with samples in cluster_samples
    """
    # Build a graph representation of the tree
    graph = OrderedDict()  # Sample name -> list of (neighbor, distance)
    sample_to_node = OrderedDict()  # Sample name -> Node object
    
    # Populate the graph and sample_to_node mapping
    def build_graph(node, parent=None, parent_distance=0):
        for sample in sorted(node.samples):  # Sort for determinism
            if sample not in graph:
                graph[sample] = []
                sample_to_node[sample] = node
            
            # If we have a parent, add an edge to each sample in parent
            if parent:
                for parent_sample in sorted(parent.samples):  # Sort for determinism
                    graph[sample].append((parent_sample, parent_distance))
                    graph[parent_sample].append((sample, parent_distance))
        
        # Process children in a deterministic order
        for child, distance in sorted(node.children, key=lambda x: tuple(sorted(x[0].samples))):
            build_graph(child, node, distance)
    
    build_graph(root)
    
    # Filter to only include samples in cluster_samples
    relevant_samples = sorted([s for s in graph if s in cluster_samples])  # Sort for determinism
    
    # If no relevant samples, return None
    if not relevant_samples:
        return None
    
    # Create a new tree with only the samples in cluster_samples
    # Start with the first sample
    first_sample = relevant_samples[0]
    new_root = Node([first_sample])
    visited = {first_sample}
    
    # Function to find the sample with the closest connected unvisited sample
    def find_closest():
        min_dist = float('inf')
        closest_sample = None
        closest_neighbor = None
        
        # For each visited sample (in deterministic order)
        for visited_sample in sorted(visited):
            # Check all its neighbors (in deterministic order)
            for neighbor, distance in sorted(graph[visited_sample]):
                if neighbor in relevant_samples and neighbor not in visited and distance < min_dist:
                    min_dist = distance
                    closest_sample = visited_sample
                    closest_neighbor = neighbor
        
        return closest_sample, closest_neighbor, min_dist
    
    # Keep adding samples until all are visited
    while len(visited) < len(relevant_samples):
        visited_sample, new_sample, distance = find_closest()
        
        if not new_sample:
            break  # No more connected samples
        
        # Find the node containing visited_sample
        def find_node(node, sample):
            if sample in node.samples:
                return node
            # Process children in a deterministic order
            for child, _ in sorted(node.children, key=lambda x: tuple(sorted(x[0].samples))):
                result = find_node(child, sample)
                if result:
                    return result
            return None
        
        # Add the new sample as a child to the node containing visited_sample
        parent_node = find_node(new_root, visited_sample)
        if parent_node:
            parent_node.add_child(Node([new_sample]), distance)
        
        visited.add(new_sample)
    
    return new_root

def create_fallback_tree(samples):
    """
    Create a simple star-like tree as fallback when extraction fails
    """
    if not samples:
        return Node([])
        
    # Use first sample as root
    samples = sorted(samples)  # Sort for determinism
    root = Node([samples[0]])
    
    # Add remaining samples as direct children
    for sample in samples[1:]:
        child = Node([sample])
        root.add_child(child, 1.0)  # Use a default distance of 1.0
        
    return root

def calculate_max_distance(subtree):
    """
    Calculate the maximum distance between any two nodes in the subtree.
    """
    # If the tree is empty or has no samples, return 0
    if not subtree or not subtree.samples:
        return 0.0
        
    # If the tree has no children, return 0
    if not subtree.children:
        return 0.0
        
    # Find the maximum edge distance
    max_edge = 0.0
    for _, distance in subtree.children:
        max_edge = max(max_edge, distance)
        
    # Find max distance in each child subtree
    max_child = 0.0
    for child, _ in subtree.children:
        child_max = calculate_max_distance(child)
        max_child = max(max_child, child_max)
        
    return max(max_edge, max_child)

def write_clusters_to_file(clusters, output_file, cluster_nwk_paths=None):
    """
    Write the clusters to a TSV file, sorted by size (largest first)
    """
    # Create a list of tuples (cluster, index) to sort by size
    indexed_clusters = [(cluster, i) for i, cluster in enumerate(clusters)]
    # Sort by cluster size in descending order
    sorted_clusters = sorted(indexed_clusters, key=lambda x: len(x[0]), reverse=True)
    
    with open(output_file, 'w') as f:
        # Add NWK path column if generating NWK files for clusters
        header = "Cluster\tSamples\tSize"
        if cluster_nwk_paths is not None:
            header += "\tNWK_Path"
        f.write(header + "\n")
        
        # Write clusters in order of size
        for new_idx, (cluster, orig_idx) in enumerate(sorted_clusters, 1):
            line = f"{new_idx}\t{','.join(sorted(cluster))}\t{len(cluster)}"
            if cluster_nwk_paths is not None and orig_idx in cluster_nwk_paths:
                line += f"\t{cluster_nwk_paths[orig_idx]}"
            f.write(line + "\n")

def main(input_file, output_file, cluster_threshold=None, cluster_output=None, 
         min_cluster_size=None, max_cluster_size=None, cluster_nwk_dir=None,
         samples_of_interest_file=None):
    """
    Main function to construct MST from distance matrix and save as Newick tree
    Also identifies clusters if threshold is provided
    
    Author: GenPat
    Contact: genpat@izs.it
    Version: 0.0.2
    """
    try:
        total_start_time = time.time()
        
        # Read the distance matrix
        print(f"[+] Reading distance matrix from {input_file}")
        read_start = time.time()
        matrix, names = read_distance_matrix(input_file)
        
         # Dopo aver letto la matrice
        dist_0199_0686 = matrix[names.index("Lm_0199")][names.index("Lm_0686")]
        print(f"[DEBUG] Distance between Lm_0199 and Lm_0686 in matrix: {dist_0199_0686}")

        
        read_time = time.time() - read_start
        print(f"[+] Loaded {len(names)}x{len(names)} distance matrix in {read_time:.2f} seconds")
        
        # Build the MST using accelerated functions
        tree = build_mst_accelerated(matrix, names)
        
        # Find clusters if threshold is provided
        if cluster_threshold is not None:
            # If samples of interest file is provided, use it to find clusters
            if samples_of_interest_file:
                print(f"[+] Reading samples of interest from {samples_of_interest_file}")
                samples_of_interest = read_samples_of_interest(samples_of_interest_file)
                print(f"[+] Found {len(samples_of_interest)} samples of interest")
                
                # Find clusters starting from samples of interest
                clusters, cluster_trees = find_clusters_from_samples(tree, samples_of_interest, cluster_threshold, 
                                                     min_cluster_size, max_cluster_size)
            else:
                # Find all clusters in the MST
                clusters, cluster_trees = find_clusters(tree, cluster_threshold)
            
            # Generate cluster statistics
            cluster_sizes = [len(cluster) for cluster in clusters]
            print(f"[+] Cluster statistics:")
            print(f"    Total clusters: {len(clusters)}")
            if clusters:
                print(f"    Average cluster size: {sum(cluster_sizes)/len(clusters):.2f}")
                print(f"    Largest cluster: {max(cluster_sizes)} samples")
                print(f"    Smallest cluster: {min(cluster_sizes)} samples")
            
            # Debug: Print distribution of cluster sizes
            size_counts = {}
            for size in cluster_sizes:
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"[+] Cluster size distribution:")
            for size in sorted(size_counts.keys()):
                print(f"    Size {size}: {size_counts[size]} clusters")
                
            # Debug: Check clusters that should meet criteria
            eligible_count = 0
            eligible_clusters = []
            for i, cluster in enumerate(clusters):
                size = len(cluster)
                if size >= min_cluster_size and (max_cluster_size is None or size <= max_cluster_size):
                    eligible_count += 1
                    eligible_clusters.append(i)
                    if eligible_count <= 5:  # Print only first 5 to avoid flooding
                        print(f"[DEBUG] Eligible cluster {i+1}: {size} samples, first few: {sorted(cluster)[:min(5, size)]}...")
            
            print(f"[DEBUG] Total eligible clusters: {eligible_count}")
            
            # Generate NWK files for clusters that meet the size criteria
            cluster_nwk_paths = {}
            if min_cluster_size is not None and cluster_nwk_dir is not None and eligible_count > 0:
                import os
                import uuid
                
                # Create directory if it doesn't exist
                os.makedirs(cluster_nwk_dir, exist_ok=True)
                
                size_range_str = f"at least {min_cluster_size}"
                if max_cluster_size is not None:
                    size_range_str = f"between {min_cluster_size} and {max_cluster_size}"
                
                print(f"[+] Generating NWK files for clusters with {size_range_str} samples...")
                
                nwk_count = 0
                # For each cluster that meets the size threshold, generate a NWK file
                for idx in eligible_clusters:
                    cluster = clusters[idx]
                    tree_node = cluster_trees[idx]
                    
                    # Check if cluster size is within the specified range
                    cluster_size = len(cluster)
                    
                    # If the tree_node has no samples, create a fallback
                    if not tree_node or not tree_node.samples:
                        print(f"[DEBUG] Creating fallback tree for cluster {idx+1} with {cluster_size} samples")
                        tree_node = create_fallback_tree(cluster)
                    
                    # Calculate the actual maximum distance in this cluster's subtree
                    max_distance = calculate_max_distance(tree_node)
                    
                    # Generate a unique filename with cluster size and actual max distance
                    unique_id = str(uuid.uuid4())[:8]
                    nwk_filename = f"cluster_{cluster_size}_{max_distance:.1f}_{unique_id}.nwk"
                    nwk_path = os.path.join(cluster_nwk_dir, nwk_filename)
                    
                    # Convert to Newick format and write to file
                    try:
                        cluster_newick = tree_node.to_newick() + ";"
                        with open(nwk_path, 'w') as f:
                            f.write(cluster_newick)
                        
                        # Store the path for inclusion in the TSV
                        cluster_nwk_paths[idx] = nwk_path
                        nwk_count += 1
                        
                        if nwk_count <= 3:  # Debug first few NWK files
                            print(f"[DEBUG] Created NWK file for cluster {idx+1}: {nwk_path}")
                            print(f"[DEBUG]   - Max distance: {max_distance:.1f}")
                            print(f"[DEBUG]   - First 100 chars: {cluster_newick[:100]}...")
                    except Exception as e:
                        print(f"[DEBUG] Error generating NWK for cluster {idx+1}: {str(e)}")
                        
                print(f"[+] Generated {nwk_count} NWK files for clusters meeting size criteria")
            
            # Write clusters to file if output file is provided
            if cluster_output:
                write_clusters_to_file(clusters, cluster_output, 
                                      cluster_nwk_paths if min_cluster_size is not None else None)
                print(f"[+] Cluster data written to {cluster_output}")
        
        # Convert to Newick format
        print(f"[+] Converting tree to Newick format...")
        newick_start = time.time()
        newick = tree.to_newick() + ";"
        newick_time = time.time() - newick_start
        print(f"[+] Converted to Newick format in {newick_time:.2f} seconds")
        
        # Write the Newick tree to the output file
        print(f"[+] Writing Newick tree to {output_file}")
        with open(output_file, 'w') as f:
            f.write(newick)
        
        total_time = time.time() - total_start_time
        print(f"[+] MST processing completed in {total_time:.2f} seconds")
        print(f"[+] Final Newick tree saved to {output_file}")
        print(f"[+] MST Builder v0.0.2 (GenPat) completed successfully")
        print(f"[+] Contact: genpat@izs.it")
    except Exception as e:
        print(f"[-] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse
    
    # Version and attribution info
    VERSION = "0.0.2"
    AUTHOR = "GenPat"
    CONTACT = "genpat@izs.it"
    
    print(f"MST Builder v{VERSION}")
    print(f"Author: {AUTHOR}")
    print(f"Contact: {CONTACT}")
    print("-" * 50)
    
    parser = argparse.ArgumentParser(description="Build MST with Numba acceleration and parallelization starting from a symmetric distance matrix. Constructs an optimal tree structure based on minimum spanning tree algorithm with zero-distance grouping.")
    parser.add_argument("input_file", help="TSV file containing the distance matrix")
    parser.add_argument("output_file", help="Output file for the Newick tree")
    parser.add_argument("--cluster-threshold", type=float, help="Maximum path distance threshold for clusters")
    parser.add_argument("--cluster-output", help="Output file for cluster data (TSV)")
    parser.add_argument("--min-cluster-size", type=int, help="Minimum number of samples required to generate a cluster NWK file")
    parser.add_argument("--max-cluster-size", type=int, help="Maximum number of samples for generating a cluster NWK file")
    parser.add_argument("--cluster-nwk-dir", help="Directory to store individual NWK files for clusters")
    parser.add_argument("--samples-of-interest", help="File containing sample IDs to use as starting points for clusters")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads (0=auto)")
    parser.add_argument("--quiet", action="store_true", help="Minimize progress output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = parser.parse_args()
    
    # Configure tqdm to disable if quiet mode
    if args.quiet:
        from functools import partial
        tqdm.__init__ = partial(tqdm.__init__, disable=True)
        print("[+] Running in quiet mode (progress bars disabled)")
    
    # Configure thread count
    if args.threads > 0:
        from numba import set_num_threads
        set_num_threads(args.threads)
        print(f"[+] Running with {args.threads} threads")
    else:
        import multiprocessing
        thread_count = multiprocessing.cpu_count()
        print(f"[+] Running with auto-detected {thread_count} threads")
    
    # Print Numba information
    from numba import __version__ as numba_version
    print(f"[+] Numba version: {numba_version}")
    print(f"[+] NumPy version: {np.__version__}")
    
    # Validate arguments
    if args.min_cluster_size is not None and args.cluster_nwk_dir is None:
        parser.error("--cluster-nwk-dir is required when --min-cluster-size is provided")
    
    if args.max_cluster_size is not None and args.min_cluster_size is None:
        parser.error("--min-cluster-size is required when --max-cluster-size is provided")
    
    if args.cluster_nwk_dir is not None and args.min_cluster_size is None:
        parser.error("--min-cluster-size is required when --cluster-nwk-dir is provided")
    
    if (args.min_cluster_size is not None or args.cluster_nwk_dir is not None) and args.cluster_threshold is None:
        parser.error("--cluster-threshold is required for cluster NWK generation")
    
    if args.min_cluster_size is not None and args.max_cluster_size is not None:
        if args.min_cluster_size > args.max_cluster_size:
            parser.error("--min-cluster-size must be less than or equal to --max-cluster-size")
    
    try:
        main(args.input_file, args.output_file, 
             cluster_threshold=args.cluster_threshold,
             cluster_output=args.cluster_output,
             min_cluster_size=args.min_cluster_size,
             max_cluster_size=args.max_cluster_size,
             cluster_nwk_dir=args.cluster_nwk_dir,
             samples_of_interest_file=args.samples_of_interest)
    except KeyboardInterrupt:
        print("\n[-] Process interrupted by user")
    except Exception as e:
        print(f"[-] Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()