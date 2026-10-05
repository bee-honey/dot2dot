"""Turn a 1-pixel-wide skeleton image into a list of traced line paths.

A skeleton is a network of lines that meet at junctions. We treat it as a
graph: pixels are nodes, touching pixels are edges.

1. Split the network into branches: runs of pixels between two "special"
   pixels (line ends or junctions).
2. Drop short spurs: tiny side branches that skeletonization leaves behind.
3. Re-join branches that continue smoothly through a junction, so a long
   curve crossing another line stays one stroke instead of breaking in two.
"""

import numpy as np

from dot2dot.models import Path

Pixel = tuple[int, int]  # (row, col), i.e. (y, x)

FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

# A joined curve may bend at most this much at a junction (cosine of ~45°).
MIN_CONTINUATION = 0.7
# How far back from a branch end to look when measuring its direction.
DIRECTION_SAMPLE = 10
# An open line whose two ends are this close (pixels) is really a loop.
LOOP_GAP = 6


def trace_skeleton(skeleton: np.ndarray, min_spur: float = 12) -> list[Path]:
    pixels: set[Pixel] = set(zip(*np.nonzero(skeleton)))
    graph = _SkeletonGraph(pixels)
    branches = graph.branches()
    branches = _prune_spurs(branches, graph, min_spur)
    return _join_branches(branches, graph)


class _SkeletonGraph:
    def __init__(self, pixels: set[Pixel]):
        self.pixels = pixels
        self.degree = {p: len(self.neighbors(p)) for p in pixels}
        # Anything that isn't a plain "line continues" pixel is a node.
        self.nodes = {p for p, d in self.degree.items() if d != 2}
        self.junction_of = self._group_junctions()

    def neighbors(self, pixel: Pixel) -> list[Pixel]:
        y, x = pixel
        found = [(y + dy, x + dx) for dy, dx in FOUR_NEIGHBORS if (y + dy, x + dx) in self.pixels]
        for dy, dx in DIAGONALS:
            # Skip a diagonal if we can already reach it through a side
            # neighbor. Otherwise every stair-step corner looks like a junction.
            if (y + dy, x + dx) in self.pixels and (y + dy, x) not in self.pixels and (
                y,
                x + dx,
            ) not in self.pixels:
                found.append((y + dy, x + dx))
        return found

    def _group_junctions(self) -> dict[Pixel, int]:
        """Label clusters of adjacent junction pixels as one junction."""
        junction_pixels = {p for p in self.nodes if self.degree[p] >= 3}
        label_of: dict[Pixel, int] = {}
        for seed in junction_pixels:
            if seed in label_of:
                continue
            label = len(set(label_of.values()))
            stack = [seed]
            while stack:
                p = stack.pop()
                if p in label_of:
                    continue
                label_of[p] = label
                stack.extend(n for n in self.neighbors(p) if n in junction_pixels)
        return label_of

    def branches(self) -> list[list[Pixel]]:
        """Walk every edge exactly once, splitting at nodes."""
        visited: set[frozenset[Pixel]] = set()
        branches = []

        def walk(start: Pixel, step: Pixel) -> list[Pixel]:
            path = [start, step]
            visited.add(frozenset((start, step)))
            prev, cur = start, step
            while cur not in self.nodes:
                options = [
                    n for n in self.neighbors(cur) if n != prev and frozenset((cur, n)) not in visited
                ]
                if not options:
                    break
                prev, cur = cur, options[0]
                visited.add(frozenset((prev, cur)))
                path.append(cur)
            return path

        for node in self.nodes:
            for n in self.neighbors(node):
                if frozenset((node, n)) not in visited:
                    branch = walk(node, n)
                    # Two touching junction pixels: not a real branch.
                    if not (len(branch) == 2 and branch[1] in self.junction_of):
                        branches.append(branch)

        # Whatever is left are loops with no nodes at all (e.g. an eye outline).
        seen = {p for branch in branches for p in branch}
        for p in self.pixels:
            if p not in seen and self.degree[p] == 2:
                loop = walk(p, self.neighbors(p)[0])
                seen.update(loop)
                branches.append(loop)
        return branches


def _branch_length(branch: list[Pixel]) -> float:
    pts = np.array(branch, dtype=np.float64)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _prune_spurs(branches, graph: _SkeletonGraph, min_spur: float):
    """Remove short branches that dangle off a junction with a free end."""

    def is_spur(branch):
        ends = (branch[0], branch[-1])
        free = sum(graph.degree.get(e, 0) == 1 for e in ends)
        at_junction = sum(e in graph.junction_of for e in ends)
        return free == 1 and at_junction == 1 and _branch_length(branch) < min_spur

    return [b for b in branches if not is_spur(b)]


def _end_direction(branch: list[Pixel], at_end: bool) -> np.ndarray:
    """Unit vector pointing out of the branch at one end (into the junction)."""
    pts = np.array(branch if at_end else branch[::-1], dtype=np.float64)
    back = pts[max(0, len(pts) - 1 - DIRECTION_SAMPLE)]
    direction = pts[-1] - back
    norm = np.linalg.norm(direction)
    return direction / norm if norm else direction


def _join_branches(branches, graph: _SkeletonGraph) -> list[Path]:
    # Each branch has two ends: (branch index, is_end). Collect ends per junction.
    ends_at: dict[int, list[tuple[int, bool]]] = {}
    for i, branch in enumerate(branches):
        if branch[0] == branch[-1]:
            continue  # Already a loop.
        for is_end, pixel in ((False, branch[0]), (True, branch[-1])):
            if pixel in graph.junction_of:
                ends_at.setdefault(graph.junction_of[pixel], []).append((i, is_end))

    # At each junction, pair up the ends that continue most smoothly.
    link: dict[tuple[int, bool], tuple[int, bool]] = {}
    for ends in ends_at.values():
        candidates = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                (ia, ea), (ib, eb) = ends[a], ends[b]
                if ia == ib:
                    continue
                # Arriving along branch A should mean leaving along branch B.
                smooth = float(np.dot(_end_direction(branches[ia], ea), -_end_direction(branches[ib], eb)))
                if smooth >= MIN_CONTINUATION or len(ends) == 2:
                    candidates.append((smooth, ends[a], ends[b]))
        for _, end_a, end_b in sorted(candidates, reverse=True):
            if end_a not in link and end_b not in link:
                link[end_a], link[end_b] = end_b, end_a

    # Follow the links to build chains of branches.
    used: set[int] = set()
    paths = []

    def follow(i: int, enter_at_end: bool) -> tuple[list[Pixel], bool]:
        """Chain branches starting at branch i, entering from one of its ends."""
        chain: list[Pixel] = []
        while i not in used:
            used.add(i)
            branch = branches[i][::-1] if enter_at_end else branches[i]
            chain.extend(branch)
            exit_end = not enter_at_end
            nxt = link.get((i, exit_end))
            if nxt is None:
                return chain, False
            i, enter_at_end = nxt
        return chain, True  # Came back to a branch we already used: a loop.

    # Start from free ends first so open curves are traced end to end.
    for i in range(len(branches)):
        for start_end in (False, True):
            if i not in used and (i, start_end) not in link:
                chain, closed = follow(i, enter_at_end=start_end)
                paths.append(_to_path(chain, closed))
    for i in range(len(branches)):
        if i not in used:
            chain, _ = follow(i, enter_at_end=False)
            paths.append(_to_path(chain, closed=True))
    return paths


def _to_path(chain: list[Pixel], closed: bool) -> Path:
    if chain[0] == chain[-1] and len(chain) > 2:
        chain, closed = chain[:-1], True
    elif len(chain) > 4 * LOOP_GAP and np.hypot(chain[0][0] - chain[-1][0], chain[0][1] - chain[-1][1]) <= LOOP_GAP:
        # Both ends meet at the same junction, e.g. a head outline that
        # touches the neck line. Treat it as a loop so it can start anywhere.
        closed = True
    # Pixels are (row, col); paths use (x, y).
    points = np.array([(x, y) for y, x in chain], dtype=np.float64)
    return Path(points, closed)
