#!/usr/bin/env python3
"""Unit test for DAG + DFS/BFS selection (P22-23).

P22 example: a->b, b->c, a->c, e->f. d is island.
  - abc = connected subgraph, ef = connected subgraph, d = island
  - budget=3:
    DFS round 1: d, a, b  (island first, then exhaust subgraph abc)
    DFS round 2: c, e, f  (continue abc, then exhaust ef)
    BFS round 1: d, a, e  (island first, one from each subgraph)
    BFS round 2: b, c, f  (continue each subgraph)
"""
import sys
sys.path.insert(0, "/root/autodl-tmp/0829/five_stage")
from dchord_5stage import (
    DAG, select_nodes_depth_first, select_nodes_breadth_first,
)

attrs = ["a", "b", "c", "d", "e", "f"]
edges = [("a", "b"), ("b", "c"), ("a", "c"), ("e", "f")]
dag = DAG(attrs, edges)

print("=== DAG structure ===")
print(f"nodes: {dag.nodes}")
print(f"edges: {dag.edges}")
print(f"islands: {[k for k in attrs if dag.is_island(k)]}")
print(f"subgraphs: {dag.connected_subgraphs()}")
print(f"group_ids: {[(k, dag.group_id(k)) for k in attrs]}")
print()

# Simulate DFS
print("=== DFS (budget=3) ===")
remaining = set(attrs)
verified = set()
round_num = 0
while remaining:
    sel = select_nodes_depth_first(remaining, dag, verified, 3)
    if not sel:
        break
    print(f"  round {round_num}: {sel}")
    for k in sel:
        verified.add(k)
        remaining.discard(k)
    round_num += 1

print()
# Simulate BFS
print("=== BFS (budget=3) ===")
remaining = set(attrs)
verified = set()
round_num = 0
while remaining:
    sel = select_nodes_breadth_first(remaining, dag, verified, 3)
    if not sel:
        break
    print(f"  round {round_num}: {sel}")
    for k in sel:
        verified.add(k)
        remaining.discard(k)
    round_num += 1

print()
print("=== CelebA (all islands, budget=3) ===")
celeba = [f"attr_{i}" for i in range(40)]
dag_c = DAG(celeba)  # no edges
print(f"islands: {sum(1 for k in celeba if dag_c.is_island(k))}/{len(celeba)}")
print(f"subgraphs: {len(dag_c.connected_subgraphs())}")
remaining = set(celeba)
verified = set()
dfs_r1 = select_nodes_depth_first(remaining, dag_c, verified, 3)
bfs_r1 = select_nodes_breadth_first(remaining, dag_c, verified, 3)
print(f"DFS round 1: {dfs_r1}")
print(f"BFS round 1: {bfs_r1}")
print(f"DFS == BFS: {dfs_r1 == bfs_r1}")

print("\nALL_TESTS_DONE")
