"""Bounded directed graph exploration; anatomical connectivity, not causal proof."""

from collections import deque
import numpy as np


def directed_path(catalog, source, target, hops=4, breadth=48, budget=4000):
    if source not in catalog.lookup or target not in catalog.lookup: raise ValueError("Require two loaded neuron IDs")
    if type(hops) is not int or not 1 <= hops <= 6: raise ValueError("Path depth must be in [1,6]")
    start, end = catalog.lookup[source], catalog.lookup[target]
    graph = catalog.outgoing
    queue, previous = deque([(start, 0)]), {start: None}
    truncated = False
    while queue:
        node, depth = queue.popleft()
        if node == end: break
        if depth >= hops: continue
        a, b = graph.indptr[node:node+2]
        offsets = np.arange(a, b)
        nonzero = offsets[graph.data[offsets] != 0]
        # Always check a direct link to the requested target before pruning the
        # expansion frontier by strength. A weak direct edge is still a route.
        if end not in previous and np.any(graph.indices[nonzero] == end):
            previous[end] = node
            break
        if len(nonzero) > breadth: truncated = True
        offsets = nonzero[np.argsort(-np.abs(graph.data[nonzero]), kind="stable")[:breadth]]
        for neighbor in graph.indices[offsets]:
            neighbor = int(neighbor)
            if neighbor not in previous:
                if len(previous) >= budget: truncated = True; continue
                previous[neighbor] = node; queue.append((neighbor, depth+1))
    if end not in previous:
        return {"found": False, "visited": len(previous), "truncated": truncated,
                "note": "No path found within the bounded strongest-edge search; absence of connectivity is not established."}
    route = []; node = end
    while node is not None: route.append(str(catalog.ids[node])); node = previous[node]
    route.reverse()
    return {"found": True, "ids": route, "edges": [catalog.edge_details(a, b) for a, b in zip(route, route[1:])],
            "visited": len(previous), "truncated": truncated,
            "note": "Directed model connectivity; this path alone does not establish causal responsibility for an action."}
