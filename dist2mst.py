import numpy as np
import time
from collections import defaultdict
from numba import njit, prange
from numba.typed import List, Dict
from tqdm import tqdm

"""
MST Builder with Numba Acceleration
===================================
Version: 0.0.1
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
    """
    Reads a distance matrix from a TSV file.
    """
    with open(tsv_file, 'r') as f:
        header = f.readline().strip().split('\t')
        names = header[1:]  # Skip empty first column
        
        matrix = []
        for line in f:
            values = line.strip().split('\t')[1:]  # Skip sample name
            matrix.append([float(x) for x in values])
            
    return np.array(matrix), names

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

@njit
def compare_distances(distance, actual_distance, tolerance=1e-10):
    """
    Compare two distances with a small tolerance
    """
    return abs(distance - actual_distance) < tolerance

def build_mst_accelerated(matrix, names, recraft=False):
    """
    Build MST with numba acceleration
    """
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
    
    if recraft:
        print(f"[+] Starting branch recrafting process...")
        root = branch_recraft_accelerated(root, matrix, names)
        print(f"[+] Branch recrafting completed")
    
    return root

@njit
def get_actual_distance(matrix, parent_idx, child_idx):
    """
    Get actual distance from matrix
    """
    return matrix[parent_idx][child_idx]

def branch_recraft_accelerated(tree, matrix, names):
    """
    Accelerated branch recraft function
    """
    start_time = time.time()
    updates_count = 0
    total_nodes = 0
    
    def count_nodes(node):
        count = 1
        for child, _ in node.children:
            count += count_nodes(child)
        return count
    
    total_node_count = count_nodes(tree)
    progress_bar = tqdm(total=total_node_count, desc="Recrafting branches", unit="node")
    
    def recraft_node(node):
        nonlocal updates_count, total_nodes
        updated_children = []
        
        for child, distance in node.children:
            # Get the indices of the parent and child nodes
            node_id = names.index(node.samples[0])  # Parent node index
            child_id = names.index(child.samples[0])  # Child node index
            
            # Get actual distance using accelerated function
            actual_distance = get_actual_distance(matrix, node_id, child_id)

            # Compare distances with tolerance
            if not compare_distances(distance, actual_distance):
                if total_nodes < 10 or updates_count < 5:  # Limit detailed logging
                    print(f"[*] Discrepancy: {node.samples[0]} → {child.samples[0]}, "
                          f"Tree: {distance:.4f}, Actual: {actual_distance:.4f}")
                updates_count += 1
                updated_children.append((child, actual_distance))
            else:
                updated_children.append((child, distance))

            # Recraft the child node
            recraft_node(child)
            
            total_nodes += 1
            progress_bar.update(1)

        # Update the children of the current node
        node.children = updated_children

    # Start recrafting from the root node
    recraft_node(tree)
    progress_bar.close()
    
    elapsed = time.time() - start_time
    print(f"[+] Branch recrafting completed in {elapsed:.2f} seconds")
    print(f"[+] Processed {total_nodes} nodes, updated {updates_count} branches")
    
    return tree

def main(input_file, output_file, recraft=False):
    """
    Main function to construct MST from distance matrix and save as Newick tree
    
    Author: GenPat
    Contact: genpat@izs.it
    Version: 0.0.1
    """
    try:
        total_start_time = time.time()
        
        # Read the distance matrix
        print(f"[+] Reading distance matrix from {input_file}")
        read_start = time.time()
        matrix, names = read_distance_matrix(input_file)
        read_time = time.time() - read_start
        print(f"[+] Loaded {len(names)}x{len(names)} distance matrix in {read_time:.2f} seconds")
        
        # Build the MST using accelerated functions
        tree = build_mst_accelerated(matrix, names, recraft=recraft)
        
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
        print(f"[+] MST Builder v0.0.1 (GenPat) completed successfully")
        print(f"[+] Contact: genpat@izs.it")
    except Exception as e:
        print(f"[-] ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    # Version and attribution info
    VERSION = "0.0.1"
    AUTHOR = "GenPat"
    CONTACT = "genpat@izs.it"
    
    print(f"MST Builder v{VERSION}")
    print(f"Author: {AUTHOR}")
    print(f"Contact: {CONTACT}")
    print("-" * 50)
    
    parser = argparse.ArgumentParser(description="Build MST with Numba acceleration and parallelization starting from a symmetric distance matrix. Constructs an optimal tree structure based on minimum spanning tree algorithm with zero-distance grouping.")
    parser.add_argument("input_file", help="TSV file containing the distance matrix")
    parser.add_argument("output_file", help="Output file for the Newick tree")
    parser.add_argument("--recraft", action="store_true", help="Enable branch recraft")
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
    
    try:
        main(args.input_file, args.output_file, recraft=args.recraft)
    except KeyboardInterrupt:
        print("\n[-] Process interrupted by user")
    except Exception as e:
        print(f"[-] Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()