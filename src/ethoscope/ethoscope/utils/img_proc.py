# author: quentin
# refactor: moomurrs
"""Image-processing utilities for blob (connected-region) manipulation.

This module provides :func:`merge_blobs`, which consolidates a list of
contours into a smaller list of convex hulls by merging nearby blobs that
likely belong to the same physical object. The merge criterion is
geometric: two blobs are merged when the Euclidean distance between their
centroids is strictly smaller than ``prop`` times the larger of the two
blobs' sizes (width + height of their minimum-area rectangles). Merging is
transitive, so chains of pairwise-close blobs collapse into a single
component. Standalone blobs that merge with nothing are preserved in the
output.

The module exposes one public type alias, :data:`Contour`, describing the
exact shape and dtype this package uses for OpenCV contours
(``(N, 1, 2)`` int32 arrays), and one public function, :func:`merge_blobs`.
"""

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np

type Contour = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]


class _UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self._parent: list[int] = list(range(size))
        self._rank: list[int] = [0] * size

    def find(self, node: int) -> int:
        """Return the root of ``node`` with path compression."""
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, a: int, b: int) -> None:
        """Merge the sets containing ``a`` and ``b`` (union by rank)."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1


def merge_blobs(contours: Sequence[Contour], prop: float = 0.5) -> list[Contour]:
    """Merge nearby blobs into convex hulls of their connected components.

    Two blobs are merged when the Euclidean distance between their centroids
    is strictly smaller than ``prop`` times the larger of the two blobs' sizes
    (width + height of their minimum-area rectangles). Merging is transitive,
    so chains of pairwise-close blobs collapse into a single component. Each
    component is returned as its convex hull, including standalone blobs that
    merge with nothing.

    Args:
        contours: Contours to merge, each an ``(N, 1, 2)`` int32 array of
            points. Each contour must contain at least two points.
        prop: Non-negative scale factor applied to the size threshold.

    Returns:
        One convex hull per connected component (merged groups and singletons).
    """
    if not contours:
        return []

    n = len(contours)
    centroids = np.empty(n, dtype=complex)
    sizes = np.empty(n, dtype=float)
    for i, contour in enumerate(contours):
        (x, y), (w, h), _ = cv2.minAreaRect(contour)
        centroids[i] = x + 1j * y
        sizes[i] = w + h

    distances = np.abs(centroids[:, None] - centroids)
    thresholds = np.maximum(sizes[:, None], sizes) * prop
    merge_mask = np.triu(distances < thresholds, k=1)

    uf = _UnionFind(n)
    rows, cols = np.nonzero(merge_mask)
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(uf.find(index), []).append(index)

    return [
        cast("Contour", cv2.convexHull(np.concatenate([contours[i] for i in indices])))
        for indices in groups.values()
    ]
