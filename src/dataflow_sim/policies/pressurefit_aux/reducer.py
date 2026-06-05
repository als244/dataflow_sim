"""Pressure reduction for PressureFit candidate interval sets."""
from __future__ import annotations

import heapq
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from dataflow_sim.policies.pressurefit_aux.core import (
    _Facts,
    _anchors,
    _effective_a,
    _fire_task_for_interval,
    _modeled_boundary_need,
    _pool_size,
)

_SplitRank = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _SplitOption:
    rank: _SplitRank
    oid: str
    interval_idx: int
    left_end: int | None
    right_start: int | None


class _PressureReducer:
    """Greedy interval splitter for one candidate spec."""

    def __init__(
        self,
        facts: _Facts,
        intervals: dict[str, list[tuple[int, int]]],
        cap: int,
        extra_pressure: list[int],
        protected_initial: set[str],
    ) -> None:
        self.facts = facts
        self.intervals = intervals
        self.cap = cap
        self.extra_pressure = extra_pressure
        self.protected_initial = protected_initial
        self.anchors_by_oid = {oid: _anchors(oid, facts) for oid in intervals}
        self.pool = _pool_size(facts, intervals)
        self.overflow_heap = [
            (-self._strict_overflow(i), i) for i in range(len(self.pool))
        ]
        heapq.heapify(self.overflow_heap)
        self.boundary_candidate_oids = self._build_boundary_candidate_oids()

    def run(self) -> None:
        max_splits = max(1, 2 * (self.facts.n + 2) * max(1, len(self.facts.sizes)))
        for _ in range(max_splits):
            worst_idx, worst_overflow = self._worst_strict_overflow()
            if worst_overflow <= 0:
                return

            split = self._best_split_at(worst_idx, allow_timing_relief=False)
            if split is None:
                worst_idx = self._worst_relaxed_boundary()
                if self._relaxed_overflow(worst_idx) <= 0:
                    return
                split = self._best_split_at(worst_idx, allow_timing_relief=True)

            if split is None:
                self._raise_unreducible(worst_idx)

            self._apply_split(split)

        raise ValueError(
            "infeasible: pressurefit pressure reduction exceeded "
            f"{max_splits} split attempts"
        )

    def _strict_overflow(self, idx: int) -> int:
        return (
            self.pool[idx]
            + self.facts.next_outputs[idx]
            + self.extra_pressure[idx]
            - self.cap
        )

    def _relaxed_overflow(self, idx: int) -> int:
        modeled_need = _modeled_boundary_need(
            self.facts, self.intervals, idx, self.pool,
        )
        return modeled_need + self.extra_pressure[idx] - self.cap

    def _worst_strict_overflow(self) -> tuple[int, int]:
        while self.overflow_heap:
            neg_overflow, idx = self.overflow_heap[0]
            overflow = self._strict_overflow(idx)
            if -neg_overflow == overflow:
                return idx, overflow
            heapq.heappop(self.overflow_heap)
        raise RuntimeError("pressurefit internal error: empty overflow heap")

    def _worst_relaxed_boundary(self) -> int:
        return max(range(len(self.pool)), key=self._relaxed_overflow)

    def _best_split_at(
        self,
        boundary_idx: int,
        *,
        allow_timing_relief: bool,
    ) -> _SplitOption | None:
        options = self._split_options_at(
            boundary_idx, allow_timing_relief=allow_timing_relief,
        )
        if not options:
            return None
        return min(options, key=lambda option: option.rank)

    def _split_options_at(
        self,
        boundary_idx: int,
        *,
        allow_timing_relief: bool,
    ) -> list[_SplitOption]:
        boundary = boundary_idx - 1
        if allow_timing_relief:
            return self._scan_split_options(boundary, allow_timing_relief=True)

        out: list[_SplitOption] = []
        for oid in self.boundary_candidate_oids[boundary_idx]:
            out.extend(self._split_options_for_oid(
                oid, boundary, allow_timing_relief=False,
            ))
        # Defensive fallback: the boundary index is an optimization, not a
        # correctness precondition for future seed transformations.
        if not out:
            return self._scan_split_options(boundary, allow_timing_relief=False)
        return out

    def _scan_split_options(
        self,
        boundary: int,
        *,
        allow_timing_relief: bool,
    ) -> list[_SplitOption]:
        out: list[_SplitOption] = []
        for oid in self.intervals:
            out.extend(self._split_options_for_oid(
                oid, boundary, allow_timing_relief=allow_timing_relief,
            ))
        return out

    def _split_options_for_oid(
        self,
        oid: str,
        boundary: int,
        *,
        allow_timing_relief: bool,
    ) -> list[_SplitOption]:
        out: list[_SplitOption] = []
        ivs = self.intervals.get(oid)
        if not ivs:
            return out
        p = self.facts.producer.get(oid, -1)
        for idx, (a, b) in enumerate(ivs):
            if not (_effective_a(a, p) <= boundary <= b):
                continue
            split_edges = self._split_edges_for_interval(
                oid, a, b, boundary, allow_timing_relief=allow_timing_relief,
            )
            if split_edges is None:
                continue
            left_end, right_start = split_edges
            option = self._ranked_split_option(
                oid, idx, a, b, left_end, right_start,
            )
            if option is not None:
                out.append(option)
        return out

    def _split_edges_for_interval(
        self,
        oid: str,
        a: int,
        b: int,
        boundary: int,
        *,
        allow_timing_relief: bool,
    ) -> tuple[int | None, int | None] | None:
        anchors = self.anchors_by_oid.get(oid)
        if anchors is None:
            anchors = _anchors(oid, self.facts)
            self.anchors_by_oid[oid] = anchors
        lo = bisect_left(anchors, a)
        hi = bisect_right(anchors, b)
        exact_pos = bisect_left(anchors, boundary, lo, hi)
        is_anchor = exact_pos < hi and anchors[exact_pos] == boundary
        if is_anchor:
            if not allow_timing_relief:
                return None
            right_pos = bisect_left(anchors, boundary + 1, lo, hi)
            if right_pos >= hi:
                return None
            left_end = boundary
            right_start = anchors[right_pos]
            if _fire_task_for_interval(oid, a, left_end, self.facts) != boundary:
                return None
            return left_end, right_start

        left_pos = bisect_right(anchors, boundary - 1, lo, hi) - 1
        right_pos = bisect_left(anchors, boundary + 1, lo, hi)
        left_end = anchors[left_pos] if left_pos >= lo else None
        right_start = anchors[right_pos] if right_pos < hi else None
        return left_end, right_start

    def _ranked_split_option(
        self,
        oid: str,
        interval_idx: int,
        interval_start: int,
        interval_end: int,
        left_end: int | None,
        right_start: int | None,
    ) -> _SplitOption | None:
        left_b = left_end if left_end is not None else interval_start - 1
        right_a = right_start if right_start is not None else interval_end + 1
        gap_len = right_a - left_b - 1
        if gap_len <= 0:
            return None
        drops_init = left_end is None and interval_start == -1
        if drops_init and oid in self.protected_initial:
            return None
        left_dirty = (
            left_end is not None
            and any(
                interval_start <= m - 1 <= left_end
                for m in self.facts.mutators.get(oid, set())
            )
        )
        release_eligible = oid in self.facts.host_ids and not left_dirty
        stream_cost = 0 if (drops_init or release_eligible) else 1
        first_use = self.facts.uses.get(oid, [self.facts.n])[0]
        rank = (
            stream_cost,
            0 if drops_init else 1,
            -first_use,
            -self.facts.sizes[oid],
            -gap_len,
        )
        return _SplitOption(rank, oid, interval_idx, left_end, right_start)

    def _apply_split(self, split: _SplitOption) -> None:
        a, b = self.intervals[split.oid][split.interval_idx]
        pieces: list[tuple[int, int]] = []
        if split.left_end is not None:
            pieces.append((a, split.left_end))
        if split.right_start is not None:
            pieces.append((split.right_start, b))
        if pieces == [(a, b)]:
            raise ValueError(
                "infeasible: pressurefit pressure reduction selected a "
                "non-progressing split"
            )

        changed_indices = _subtract_removed_interval_pressure(
            self.facts, self.pool, split.oid, (a, b), pieces,
        )
        for changed_idx in changed_indices:
            heapq.heappush(
                self.overflow_heap,
                (-self._strict_overflow(changed_idx), changed_idx),
            )

        self.intervals[split.oid][split.interval_idx:split.interval_idx + 1] = pieces
        if not self.intervals[split.oid]:
            del self.intervals[split.oid]

    def _build_boundary_candidate_oids(self) -> list[list[str]]:
        """Map each boundary to objects with a strict removable gap there."""
        by_boundary: list[list[str]] = [[] for _ in range(self.facts.n + 1)]
        for oid, ivs in self.intervals.items():
            p = self.facts.producer.get(oid, -1)
            anchors = self.anchors_by_oid.get(oid)
            if anchors is None:
                anchors = _anchors(oid, self.facts)
                self.anchors_by_oid[oid] = anchors
            anchor_set = set(anchors)
            for a, b in ivs:
                start = max(-1, _effective_a(a, p))
                end = min(self.facts.n - 1, b)
                if start > end:
                    continue
                for boundary in range(start, end + 1):
                    if boundary not in anchor_set:
                        by_boundary[boundary + 1].append(oid)
        return by_boundary

    def _raise_unreducible(self, boundary_idx: int) -> None:
        boundary = boundary_idx - 1
        raise ValueError(
            f"infeasible: pressurefit cannot reduce boundary {boundary} "
            f"under device_capacity={self.cap}"
        )


def _reduce_to_fit(
    facts: _Facts,
    intervals: dict[str, list[tuple[int, int]]],
    cap: int | None,
    extra_pressure: list[int] | None = None,
    protected_initial: set[str] | None = None,
) -> None:
    """Mutate `intervals` into a pressure-fit interval set."""
    if cap is None:
        return
    if extra_pressure is None:
        extra_pressure = [0] * (facts.n + 1)
    if protected_initial is None:
        protected_initial = set()
    _PressureReducer(
        facts, intervals, cap, extra_pressure, protected_initial,
    ).run()


def _subtract_removed_interval_pressure(
    facts: _Facts,
    pool: list[int],
    oid: str,
    old: tuple[int, int],
    new_pieces: list[tuple[int, int]],
) -> list[int]:
    """Update a precomputed pool after splitting one interval."""
    changed_indices: list[int] = []
    p = facts.producer.get(oid, -1)
    old_a, old_b = old
    old_start = max(-1, _effective_a(old_a, p))
    old_end = min(facts.n - 1, old_b)
    if old_start > old_end:
        return changed_indices

    normalized_pieces: list[tuple[int, int]] = []
    for a, b in new_pieces:
        start = max(-1, _effective_a(a, p))
        end = min(facts.n - 1, b)
        if start <= end:
            normalized_pieces.append((start, end))
    normalized_pieces.sort()

    size = facts.sizes[oid]
    cursor = old_start
    for start, end in normalized_pieces:
        if cursor <= start - 1:
            for boundary in range(cursor, start):
                idx = boundary + 1
                pool[idx] -= size
                changed_indices.append(idx)
        cursor = max(cursor, end + 1)
    if cursor <= old_end:
        for boundary in range(cursor, old_end + 1):
            idx = boundary + 1
            pool[idx] -= size
            changed_indices.append(idx)
    return changed_indices
