"""Group 8 player: score-driven search over many enclosure shapes

Pipeline: backtracking piece allocation -> best-first template search (rectangles,
cut-corner octagons/hexagons, L/U/T) -> grid-traced outlines for big rooms ->
greedy bump growth -> randomized restarts, every candidate checked with the
simulator's own validator. Returns the best valid enclosure, or None.
"""

import bisect
import heapq
import math
import random
import time

from shapely.geometry import Polygon

from players.player0 import Player
from src.constants import MIN_WALL_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

try:  # prepared geometry makes repeated containment tests much faster
    from shapely.prepared import prep
except ImportError:  # pragma: no cover
    prep = None

# unit vectors for the 8 directions, exact for the axis ones
_R2 = math.sqrt(0.5)
UNIT = [
    (1, 0),
    (_R2, _R2),
    (0, 1),
    (-_R2, _R2),
    (-1, 0),
    (-_R2, -_R2),
    (0, -1),
    (_R2, -_R2),
]

# turn (in 45-degree steps, counter-clockwise) -> connector at that corner
TURN_CONNECTOR = {
    1: Connector(ConnectorType.DIAGONAL),  # 135 inside: turn 45 left
    2: Connector(ConnectorType.RIGHT),  # 90 inside: turn 90 left
    # 270 inside: turn 90 right
    6: Connector(ConnectorType.RIGHT, reflex=True),
    # 225 inside: turn 45 right
    7: Connector(ConnectorType.DIAGONAL, reflex=True),
}
TURN_SIGNED = {1: 1, 2: 2, 6: -2, 7: -1}


class Shape:
    """A closed, counter-clockwise loop of sides in its own local frame."""

    def __init__(self, family: str, params: tuple, dirs: list[int], lengths: list[int]):
        self.family = family
        self.params = params
        self.dirs = dirs
        self.lengths = lengths
        n = len(dirs)
        self.turns = [(dirs[i] - dirs[i - 1]) % 8 for i in range(n)]
        self.ok = all(t in TURN_CONNECTOR for t in self.turns) and (
            sum(TURN_SIGNED.get(t, 0) for t in self.turns) == 8
        )
        self.right = sum(1 for t in self.turns if t in (2, 6))
        self.diagonal = sum(1 for t in self.turns if t in (1, 7))
        pts = [(0.0, 0.0)]
        for d, length in zip(dirs, lengths):
            ux, uy = UNIT[d]
            x, y = pts[-1]
            pts.append((x + ux * length, y + uy * length))
        self.closes = math.hypot(*pts[-1]) < 1e-6
        self.vertices = pts[:-1]
        area2 = 0.0
        for i in range(n):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % n]
            area2 += x1 * y2 - x2 * y1
        self.area = area2 / 2
        self.perimeter = float(sum(lengths))
        self.diameter = max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a in self.vertices
            for b in self.vertices
        )

    def valid(self) -> bool:
        return self.ok and self.closes and self.area > 0

    def interior_angle(self, i: int) -> float:
        """Inside angle at vertex i, degrees."""
        return 180.0 - 45.0 * TURN_SIGNED[self.turns[i]]

    def world_vertices(self, theta_deg: float, tx: float, ty: float):
        c, s = math.cos(math.radians(theta_deg)), math.sin(
            math.radians(theta_deg))
        return [(tx + c * x - s * y, ty + s * x + c * y) for x, y in self.vertices]

    def construction(
        self,
        sides_pieces: list[list[tuple[str, int]]],
        theta_deg: float,
        tx: float,
        ty: float,
    ) -> Construction:
        """The Construction for this shape rotated by theta and moved so its
        local origin (vertex 0) lands on (tx, ty)."""
        x0, y0 = self.world_vertices(theta_deg, tx, ty)[0]
        pieces: list[Piece] = []
        connectors: list[Connector] = []
        for i, side in enumerate(sides_pieces):
            for j, (kind, length) in enumerate(side):
                pt = PieceType.GATE if kind == "gate" else PieceType.WALL
                pieces.append(Piece(pt, length))
                if j == 0:
                    connectors.append(TURN_CONNECTOR[self.turns[i]])
                else:
                    connectors.append(Connector(ConnectorType.STRAIGHT))
        return Construction(
            start=(x0, y0),
            start_heading=theta_deg + 45.0 * self.dirs[0],
            pieces=pieces,
            connectors=connectors,
        )


# ------------------------------------------------------------------ templates
def rectangle(a: int, b: int) -> Shape:
    return Shape("rect", (a, b), [0, 2, 4, 6], [a, b, a, b])


def octagon(a: int, b: int, c: int, d: int) -> Shape:
    """A rectangle with its corners cut at 45 degrees (c on NE/SW, d on NW/SE)."""
    return Shape(
        "oct", (a, b, c, d), [0, 1, 2, 3, 4, 5, 6, 7], [a, c, b, d, a, c, b, d]
    )


def hexagon(a: int, b: int, c: int) -> Shape:
    """A rectangle with two opposite corners cut at 45 degrees."""
    return Shape("hex", (a, b, c), [0, 1, 2, 4, 5, 6], [a, c, b, a, c, b])


def l_shape(a1: int, a2: int, b1: int, b2: int) -> Shape:
    """An L: bottom a1+a2 wide, right arm b1 high, upper-left arm b1+b2 high and
    a1 wide (the notch is a2 x b2)."""
    return Shape(
        "L", (a1, a2, b1, b2), [0, 2, 4, 2, 4, 6], [
            a1 + a2, b1, a2, b2, a1, b1 + b2]
    )


def u_shape(a1: int, a2: int, a3: int, b1: int, b2: int) -> Shape:
    """A U opening upwards: base a1+a2+a3 wide and b1 high, arms a1 and a3 wide
    going up b2 more, the gap between them a2 wide."""
    return Shape(
        "U",
        (a1, a2, a3, b1, b2),
        [0, 2, 4, 6, 4, 2, 4, 6],
        [a1 + a2 + a3, b1 + b2, a3, b2, a2, b2, a1, b1 + b2],
    )


def t_shape(a1: int, a2: int, a3: int, b1: int, b2: int) -> Shape:
    """A T upside down (a wide base with a stem on top): base a1+a2+a3 wide, b1
    high; the stem is a2 wide and b2 high, centred over the middle part."""
    return Shape(
        "T",
        (a1, a2, a3, b1, b2),
        [0, 2, 4, 2, 4, 6, 4, 6],
        [a1 + a2 + a3, b1, a3, b2, a2, b2, a1, b1],
    )


WALL = "wall"
GATE = "gate"

MAX_PARTS = 6  # a side of <= 30 units built from pieces of >= 5 units
NODE_LIMIT = 4000  # backtracking budget per allocation attempt


class Allocator:
    _relax_limit = 24  # pre-check applies to shapes with at most this many sides
    # Single-gate allocations may borrow spare gates as walls only if the owner says
    # so (not worth it when a two-gate enclosure, with the bonus, is also on offer).
    single_gate_pool = True

    def __init__(self, walls: dict[int, int], gates: dict[int, int], straight: int):
        self.walls = {
            int(n): k for n, k in walls.items() if n >= MIN_WALL_LENGTH and k > 0
        }
        self.gates = {
            int(n): k for n, k in gates.items() if n >= MIN_WALL_LENGTH and k > 0
        }
        self.straight = max(0, straight)
        # Spare gates are real pieces too: any gate we don't need as "the gate" can
        # serve as plain wall (the simulator only needs >= 1 gate, +G for two faces).
        self.pool = dict(self.walls)
        for n, k in self.gates.items():
            self.pool[n] = self.pool.get(n, 0) + k
        self.walls_only = self.walls
        # Two stocks: walls only (fast, tried first) and walls + spare gates. The
        # default view (side_ok, partitions) is the bigger one; allocate() tries
        # the small one first.
        self._modes: dict[bool, dict] = {}
        self._cache: dict[tuple, list | None] = {}
        self.default_pool = False
        self._set_mode(self.default_pool)

    def _set_mode(self, use_pool: bool) -> None:
        st = self._modes.setdefault(use_pool, {"parts": {}, "relax": {}})
        self.walls = self.pool if use_pool else self.walls_only
        self.wall_lengths = sorted(self.walls, reverse=True)
        self._parts = st["parts"]
        self._relax = st["relax"]

    # ------------------------------------------------------------- partitions
    def wall_partitions(self, total: int) -> list[tuple[int, ...]]:
        """Every multiset of wall lengths that adds up to `total` (each length used
        at most as often as it is in stock), fewest pieces first."""
        if total in self._parts:
            return self._parts[total]
        out: list[tuple[int, ...]] = []

        def rec(remaining: int, start: int, chosen: list[int]) -> None:
            if remaining == 0:
                out.append(tuple(chosen))
                return
            if len(chosen) >= MAX_PARTS:
                return
            for k in range(start, len(self.wall_lengths)):
                n = self.wall_lengths[k]
                if n > remaining or chosen.count(n) >= self.walls[n]:
                    continue
                chosen.append(n)
                rec(remaining - n, k, chosen)
                chosen.pop()

        if total > 0:
            rec(total, 0, [])
        # fewest pieces first; among equals, prefer lengths we have lots of
        out.sort(key=lambda p: (len(p), sum(1.0 / self.walls[n] for n in p)))
        self._parts[total] = out
        return out

    def side_ok(self, total: int) -> bool:
        """Can a side of this length be built at all (walls only, or with a gate)?"""
        if self.wall_partitions(total):
            return True
        return any(
            g == total
            or (total - g >= MIN_WALL_LENGTH and self.wall_partitions(total - g))
            for g in self.gates
        )

    def gate_options(self, total: int) -> list[int]:
        """Gate lengths that can sit on a side of this length."""
        return [
            g
            for g in sorted(self.gates, reverse=True)
            if g == total
            or (total - g >= MIN_WALL_LENGTH and self.wall_partitions(total - g))
        ]

    def min_parts(self, total: int) -> int:
        parts = self.wall_partitions(total)
        return len(parts[0]) if parts else 99

    # ------------------------------------------------------------- allocation

    def allocate(
        self, sides: list[int], n_gates: int = 1
    ) -> list[list[tuple[str, int]]] | None:
        key = (tuple(sides), n_gates)
        if key in self._cache:
            return self._cache[key]
        self._set_mode(True)
        rkey = tuple(sorted(sides))
        ok = self._relax.get(rkey) if len(sides) <= self._relax_limit else True
        if ok is None:
            ok = (
                self._fill(list(sides), dict(self.pool),
                           self.straight, [0] * len(sides))
                is not None
            )
            self._relax[rkey] = ok
        if not ok:
            self._set_mode(self.default_pool)
            self._cache[key] = None
            return None
        result = None
        spare = sum(self.gates.values()) - n_gates
        use_pool = spare >= 1 and (n_gates >= 2 or self.single_gate_pool)
        modes = [False, True] if use_pool else [False]
        for mode in modes:
            self._set_mode(mode)
            result = self._allocate(list(sides), n_gates)
            if result is not None:
                break
        self._set_mode(self.default_pool)
        self._cache[key] = result
        return result

    def _allocate(self, sides: list[int], n_gates: int) -> list | None:
        if n_gates < 1 or sum(self.gates.values()) < n_gates:
            return None
        if any(not self.side_ok(s) for s in sides):
            return None
        # which (side, gate length) pairs are possible, longest gate first
        options = []
        seen = set()
        for i, s in enumerate(sides):
            for g in self.gate_options(s):
                if (s, g) in seen and n_gates == 1:
                    continue
                seen.add((s, g))
                options.append((0 if g == s else 1, -g, i, g))
        options.sort()
        if not options:
            return None

        tries = 0
        if n_gates == 1:
            for _, _, i, g in options:
                tries += 1
                if tries > 12:
                    break
                res = self._fill_with_gates(sides, {i: g})
                if res is not None:
                    return res
            return None

        for a in range(len(options)):
            for b in range(a + 1, len(options)):
                _, _, i, g = options[a]
                _, _, j, h = options[b]
                if i == j:
                    continue
                if g == h and self.gates[g] < 2:
                    continue
                tries += 1
                if tries > 25:
                    return None
                res = self._fill_with_gates(sides, {i: g, j: h})
                if res is not None:
                    return res
        return None

    def _fill_with_gates(
        self, sides: list[int], gate_at: dict[int, int]
    ) -> list | None:
        walls = dict(self.walls)
        if self.walls is self.pool:
            for g in gate_at.values():
                walls[g] -= 1  # reserved as the gate(s) themselves
        remaining = [s - gate_at.get(i, 0) for i, s in enumerate(sides)]
        # a gate sharing its side with walls costs one more STRAIGHT joint
        extra = [
            1 if i in gate_at and remaining[i] > 0 else 0 for i in range(len(sides))
        ]
        filled = self._fill(remaining, walls, self.straight, extra)
        if filled is None:
            return None
        # real walls are used first; once a length runs out of walls, gates fill in
        wall_left = dict(self.walls_only)
        out = []
        for i, parts in enumerate(filled):
            side = []
            if i in gate_at:
                side.append((GATE, gate_at[i]))
            for n in parts:
                if wall_left.get(n, 0) > 0:
                    wall_left[n] -= 1
                    side.append((WALL, n))
                else:
                    side.append((GATE, n))
            out.append(side)
        return out

    def _fill(
        self,
        lengths: list[int],
        walls: dict[int, int],
        straight_left: int,
        extra: list[int],
    ):
        """Walls for each length (0 = nothing needed). Backtracking, longest first."""
        n = len(lengths)
        order = sorted(range(n), key=lambda i: -lengths[i])
        min_need = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            i = order[k]
            need = 0 if lengths[i] == 0 else self.min_parts(
                lengths[i]) - 1 + extra[i]
            min_need[k] = min_need[k + 1] + need
        choice: list[tuple[int, ...] | None] = [None] * n
        nodes = [0]

        def rec(k: int, straight_left: int, prev_key: tuple, prev_idx: int) -> bool:
            if k == n:
                return True
            nodes[0] += 1
            if nodes[0] > NODE_LIMIT:
                return False
            if min_need[k] > straight_left:
                return False
            i = order[k]
            length = lengths[i]
            if length == 0:
                choice[i] = ()
                return rec(k + 1, straight_left, (), 0)
            parts = self.wall_partitions(length)
            key = (length, extra[i])
            start = prev_idx if key == prev_key else 0
            for idx in range(start, len(parts)):
                p = parts[idx]
                cost = len(p) - 1 + extra[i]
                if cost > straight_left:
                    break
                ok = True
                taken = []
                for m in p:
                    if walls.get(m, 0) > 0:
                        walls[m] -= 1
                        taken.append(m)
                    else:
                        ok = False
                        break
                if ok:
                    choice[i] = p
                    if rec(k + 1, straight_left - cost, key, idx):
                        return True
                for m in taken:
                    walls[m] += 1
            return False

        if not rec(0, straight_left, (), 0):
            return None
        return [list(c) for c in choice]

# ============================================================================
# Placement
# Finding where in the room a shape fits.
#
# Good placements almost always touch the room's walls and line up with them, so
# instead of sweeping angles blindly we put each corner of the shape into each
# corner of the room, with one of the shape's sides lying along one of the room's
# walls. From there we also
# slide the shape along that wall. Only if none of that works do we fall back to a
# sweep of positions and angles.
# ============================================================================


def _angle(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _norm(a: float) -> float:
    """Angle into (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180:
        a += 360
    elif a > 180:
        a -= 360
    return a


class Placer:
    def __init__(self, room, deadline_check):
        self.room = room
        self.out_of_time = deadline_check
        pts = list(room.polygon.exterior.coords)[:-1]
        area2 = sum(
            pts[i][0] * pts[(i + 1) % len(pts)][1]
            - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts))
        )
        if area2 < 0:
            pts.reverse()
        self.pts = pts
        n = len(pts)
        self.edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
        self.out_angle = [_angle(b[0] - a[0], b[1] - a[1])
                          for a, b in self.edges]
        self.in_angle = [self.out_angle[i - 1] for i in range(n)]
        self.interior = [
            180.0 - _norm(self.out_angle[i] - self.in_angle[i]) for i in range(n)
        ]
        self.edge_len = [math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in self.edges]
        self.bounds = room.polygon.bounds
        buffered = room.polygon.buffer(TOL)
        self._checker = prep(buffered) if prep is not None else buffered
        self.checks = 0
        self.max_radius = self._max_inscribed_radius()

    def _max_inscribed_radius(self) -> float:
        """An upper bound on the radius of the largest circle that fits in the room
        (sampled on a grid, plus the sampling error). A shape whose own inner circle
        is bigger can't fit anywhere, so we skip it without searching."""
        minx, miny, maxx, maxy = self.bounds
        steps = 70
        hx, hy = (maxx - minx) / steps, (maxy - miny) / steps
        best = 0.0
        pts = self.pts
        for a in range(steps + 1):
            x = minx + a * hx
            for b in range(steps + 1):
                y = miny + b * hy
                inside = False
                j = len(pts) - 1
                for i in range(len(pts)):
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    if (yi > y) != (yj > y):
                        if x < xi + (y - yi) * (xj - xi) / (yj - yi):
                            inside = not inside
                    j = i
                if not inside:
                    continue
                d = min(self._seg_dist(x, y, e) for e in self.edges)
                best = max(best, d)
        return best + math.hypot(hx, hy) / 2 + 1e-6

    @staticmethod
    def _seg_dist(x, y, edge):
        (ax, ay), (bx, by) = edge
        dx, dy = bx - ax, by - ay
        seg = dx * dx + dy * dy
        t = (
            0.0
            if seg == 0
            else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg))
        )
        return math.hypot(x - ax - t * dx, y - ay - t * dy)

    # ---------------------------------------------------------------- testing

    def _point_ok(self, x: float, y: float) -> bool:
        """Inside the room, or within a hair of its boundary (cheap pre-filter)."""
        inside = False
        pts = self.pts
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                if x < xi + (y - yi) * (xj - xi) / (yj - yi):
                    inside = not inside
            j = i
        if inside:
            return True
        for (ax, ay), (bx, by) in self.edges:
            dx, dy = bx - ax, by - ay
            seg = dx * dx + dy * dy
            t = (
                0.0
                if seg == 0
                else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg))
            )
            if math.hypot(x - ax - t * dx, y - ay - t * dy) <= 1e-3:
                return True
        return False

    def fits(self, world: list[tuple[float, float]]) -> bool:
        minx, miny, maxx, maxy = self.bounds
        for x, y in world:
            if x < minx - 1e-3 or x > maxx + 1e-3 or y < miny - 1e-3 or y > maxy + 1e-3:
                return False
        for x, y in world:
            if not self._point_ok(x, y):
                return False
        self.checks += 1
        return self._checker.contains(Polygon(world))

    # -------------------------------------------------------------- searching

    def candidates(self, shape, step45: bool):
        """(theta, tx, ty) placements to try, best guesses first."""
        n_room = len(self.pts)
        n_shape = len(shape.vertices)
        shape_out = [45.0 * d for d in shape.dirs]
        shape_in = [shape_out[i - 1] for i in range(n_shape)]
        pairs = []
        for j in range(n_room):
            for i in range(n_shape):
                gap = self.interior[j] - shape.interior_angle(i)
                if gap < -1e-6:
                    continue
                pairs.append((gap, j, i))
        pairs.sort()
        for _, j, i in pairs:
            rx, ry = self.pts[j]
            for theta, along in (
                (self.out_angle[j] - shape_out[i], self.out_angle[j]),
                (self.in_angle[j] - shape_in[i], self.in_angle[j] + 180.0),
            ):
                c, s = math.cos(math.radians(theta)), math.sin(
                    math.radians(theta))
                vx, vy = shape.vertices[i]
                tx, ty = rx - (c * vx - s * vy), ry - (s * vx + c * vy)
                yield theta, tx, ty
        # slide along each wall from its corners
        for _, j, i in pairs:
            rx, ry = self.pts[j]
            theta = self.out_angle[j] - shape_out[i]
            c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
            vx, vy = shape.vertices[i]
            ex, ey = (
                math.cos(math.radians(self.out_angle[j])),
                math.sin(math.radians(self.out_angle[j])),
            )
            room_len = self.edge_len[j]
            for frac in (0.5, 0.25, 0.75, 1.0):
                d = (room_len - shape.lengths[i]) * frac
                if d <= 1e-6:
                    continue
                tx = rx + ex * d - (c * vx - s * vy)
                ty = ry + ey * d - (s * vx + c * vy)
                yield theta, tx, ty
        # fallback: centre the shape on interior points at room-aligned angles
        yield from self._sweep(shape, step45)

    def _sweep(self, shape, step45: bool):
        period = 45.0 if step45 else 90.0
        base = sorted({round(a % period, 6) for a in self.out_angle})
        angles = []
        for b in base:
            k = 0.0
            while k < 360.0:
                angles.append(b + k)
                k += period
        for a in range(0, 360, 10):
            angles.append(float(a))
        cx = sum(v[0] for v in shape.vertices) / len(shape.vertices)
        cy = sum(v[1] for v in shape.vertices) / len(shape.vertices)
        minx, miny, maxx, maxy = self.bounds
        steps = 7
        points = []
        rp = self.room.polygon.representative_point()
        points.append((rp.x, rp.y))
        for a in range(steps):
            for b in range(steps):
                x = minx + (maxx - minx) * (a + 0.5) / steps
                y = miny + (maxy - miny) * (b + 0.5) / steps
                if self._point_ok(x, y):
                    points.append((x, y))
        for px, py in points:
            for theta in angles:
                c, s = math.cos(math.radians(theta)), math.sin(
                    math.radians(theta))
                yield theta, px - (c * cx - s * cy), py - (s * cx + c * cy)

    def place(self, shape, max_checks: int = 4000):
        """A (theta, tx, ty) that puts the whole shape inside the room, or None."""
        step45 = shape.diagonal > 0
        seen = set()
        tries = 0
        for theta, tx, ty in self.candidates(shape, step45):
            key = (round(theta % 360.0, 4), round(tx, 4), round(ty, 4))
            if key in seen:
                continue
            seen.add(key)
            tries += 1
            if tries > max_checks or (tries % 64 == 0 and self.out_of_time()):
                return None
            if self.fits(shape.world_vertices(theta, tx, ty)):
                return theta, tx, ty
        return None

# ============================================================================
# Refinement
# Growing a right-angled enclosure by adding rectangular bumps.
#
# Starting from a valid right-angled enclosure (a rectangle, an L, a grid outline...),
# we repeatedly push out a rectangle w wide and h deep from one of its sides, into
# free floor, whenever that raises the score. Lengths are whole numbers that the
# inventory can build, so the result still closes exactly; we check that the bump
# stays in the room, doesn't touch the rest of the enclosure (which would make a
# hole or a pinch), and that no face gets longer than 30.
# ============================================================================


class Refiner:
    def __init__(
        self, placer, side_lengths: list[int], weights, max_corners: int, out_of_time
    ):
        self.placer = placer
        self.R = sorted(side_lengths)
        self.Rset = set(side_lengths)
        self.weights = weights
        self.max_corners = max_corners
        self.out_of_time = out_of_time

    def refine(
        self,
        shape: Shape,
        theta: float,
        tx: float,
        ty: float,
        max_moves: int = 400,
        rng=None,
        top_k: int = 1,
    ):
        if any(d % 2 for d in shape.dirs):
            return
        self.rng, self.top_k = rng, top_k
        # work in the shape's own frame (rotated by -theta), origin at (0, 0) world
        verts = [_rot(x, y, -theta)
                 for x, y in shape.world_vertices(theta, tx, ty)]
        room = [_rot(x, y, -theta) for x, y in self.placer.pts]
        self._room = room
        room_edges = [(room[i], room[(i + 1) % len(room)])
                      for i in range(len(room))]
        for _ in range(max_moves):
            if self.out_of_time():
                return
            move = self._best_move(verts, room_edges)
            if move is None:
                return
            verts = move
            got = self._to_shape(verts, theta)
            if got is None:
                return
            yield got

    def _best_move(self, verts, room_edges):
        n = len(verts)
        A, C = self.weights.A, self.weights.C
        cands = []
        best = None
        greedy = getattr(self, "rng", None) is None or getattr(
            self, "top_k", 1) <= 1
        for i in range(n):
            if i % 16 == 0 and self.out_of_time():
                return None
            p, q = verts[i], verts[(i + 1) % n]
            L = math.hypot(q[0] - p[0], q[1] - p[1])
            Li = int(round(L))
            ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            nx, ny = uy, -ux  # outward (right of travel) for a CCW loop
            prev = verts[i - 1]
            nxt = verts[(i + 2) % n]
            # is the corner at p (and q) convex? then o = 0 (post = 0) merges with it
            p_convex = (p[0] - prev[0]) * nx + (p[1] - prev[1]) * ny > 1e-9
            q_convex = (nxt[0] - q[0]) * nx + (nxt[1] - q[1]) * ny < -1e-9
            prev_len = int(round(math.hypot(p[0] - prev[0], p[1] - prev[1])))
            next_len = int(round(math.hypot(nxt[0] - q[0], nxt[1] - q[1])))

            # everything near this side, in its frame (u along it, v outwards)
            room_obs = self._near(room_edges, p, ux, uy, nx, ny, Li)
            own = [(verts[k], verts[(k + 1) % n]) for k in range(n) if k != i]
            poly_obs = self._near(own, p, ux, uy, nx, ny, Li)

            offsets = [0] if p_convex else []
            offsets += [o for o in self.R if o <= Li - 5]
            for o in offsets:
                for w in self.R:
                    post = Li - o - w
                    if post < 0:
                        break
                    if post == 0 and not q_convex:
                        continue
                    if post != 0 and post not in self.Rset:
                        continue
                    if A * w + 2 * C <= 0:
                        continue
                    h_cap = self._clearance(room_obs, poly_obs, o, o + w)
                    if h_cap < 5:
                        continue
                    # the floor just outside this side must be room, not outdoors
                    um = o + w / 2
                    if not self._in_room(
                        p[0] + ux * um + nx * 1e-3, p[1] + uy * um + ny * 1e-3
                    ):
                        continue
                    h = self._best_h(h_cap, o, post, prev_len, next_len)
                    if h is None:
                        continue
                    gain = A * w * h + 2 * C * h
                    added = (2 if o > 0 else 0) + (2 if post > 0 else 0)
                    if n + added > self.max_corners:
                        continue
                    if greedy:
                        if best is None or gain > best[0]:
                            best = (gain, i, o, w, h, post)
                    elif gain > 1e-9:
                        cands.append((gain, i, o, w, h, post))
        if not greedy and cands:
            cands.sort(key=lambda c: -c[0])
            top = cands[: self.top_k]
            best = self.rng.choices(top, weights=[c[0] for c in top])[0]
        if best is None or best[0] <= 1e-9:
            return None
        _, i, o, w, h, post = best
        return self._apply(verts, i, o, w, h, post)

    def _in_room(self, x: float, y: float) -> bool:
        pts = self._room
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y):
                if x < xi + (y - yi) * (xj - xi) / (yj - yi):
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _near(edges, p, ux, uy, nx, ny, Li):
        out = []
        for a, b in edges:
            au = (a[0] - p[0]) * ux + (a[1] - p[1]) * uy
            av = (a[0] - p[0]) * nx + (a[1] - p[1]) * ny
            bu = (b[0] - p[0]) * ux + (b[1] - p[1]) * uy
            bv = (b[0] - p[0]) * nx + (b[1] - p[1]) * ny
            if max(av, bv) <= 1e-9 or min(av, bv) > 31:
                continue
            if max(au, bu) < -1e-9 or min(au, bu) > Li + 1e-9:
                continue
            out.append((au, av, bu, bv))
        return out

    @staticmethod
    def _v_range(au, av, bu, bv, lo, hi, closed):
        """(v_min, v_max) of the part of the edge with u in [lo, hi] (or the open
        interval), or None if it has no such part."""
        e = 1e-9
        if abs(bu - au) < 1e-12:
            inside = (lo - e <= au <= hi +
                      e) if closed else (lo + e < au < hi - e)
            if not inside:
                return None
            return min(av, bv), max(av, bv)
        t0 = (lo - au) / (bu - au)
        t1 = (hi - au) / (bu - au)
        tlo, thi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if closed:
            if tlo > thi + 1e-12:
                return None
        else:
            if thi - tlo <= 1e-12:
                return None
        v1 = av + (bv - av) * tlo
        v2 = av + (bv - av) * thi
        return min(v1, v2), max(v1, v2)

    def _clearance(self, room_obs, poly_obs, u0: float, u1: float) -> float:
        """How deep a bump over u in [u0, u1] can go: it may touch the room's walls
        but must stay clear of the rest of the enclosure."""
        cap = 30.0
        for au, av, bu, bv in room_obs:
            r = self._v_range(au, av, bu, bv, u0, u1, closed=False)
            if r is None or r[1] <= 1e-9:
                continue
            cap = min(cap, max(r[0], 0.0))
            if cap <= 0:
                return 0.0
        for au, av, bu, bv in poly_obs:
            r = self._v_range(au, av, bu, bv, u0, u1, closed=True)
            if r is None or r[1] <= 1e-7:
                continue
            cap = min(cap, max(r[0], 0.0) - 1e-6)
            if cap <= 0:
                return 0.0
        return cap

    def _best_h(self, cap, o, post, prev_len, next_len):
        for h in reversed(self.R):
            if h > cap + 1e-9:
                continue
            if o == 0 and prev_len + h not in self.Rset:
                continue
            if post == 0 and next_len + h not in self.Rset:
                continue
            return h
        return None

    @staticmethod
    def _apply(verts, i, o, w, h, post):
        n = len(verts)
        p, q = verts[i], verts[(i + 1) % n]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        nx, ny = uy, -ux
        a = (p[0] + ux * o, p[1] + uy * o)
        b = (a[0] + nx * h, a[1] + ny * h)
        c = (b[0] + ux * w, b[1] + uy * w)
        d = (c[0] - nx * h, c[1] - ny * h)
        new = verts[: i + 1]
        if o > 0:
            new.append(a)
        new += [b, c]
        if post > 0:
            new.append(d)
        new += verts[i + 1:]
        # drop vertices that became straight (collinear with both neighbours)
        out = []
        m = len(new)
        for k in range(m):
            x0, y0 = new[k - 1]
            x1, y1 = new[k]
            x2, y2 = new[(k + 1) % m]
            cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
            if abs(cross) > 1e-6:
                out.append(new[k])
        return out

    @staticmethod
    def _to_shape(verts, theta):
        n = len(verts)
        dirs, lengths = [], []
        for k in range(n):
            (x0, y0), (x1, y1) = verts[k], verts[(k + 1) % n]
            L = math.hypot(x1 - x0, y1 - y0)
            if abs(L - round(L)) > 1e-6:
                return None
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360.0
            d = int(round(ang / 45.0)) % 8
            if abs(ang - 45.0 * round(ang / 45.0)) > 1e-4:
                return None
            dirs.append(d)
            lengths.append(int(round(L)))
        shape = Shape("bumped", (n,), dirs, lengths)
        if not shape.valid():
            return None
        tx, ty = _rot(verts[0][0], verts[0][1], theta)
        return shape, theta, tx, ty


N4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
# ring around a cell, in order: E, NE, N, NW, W, SW, S, SE
RING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _rot(x: float, y: float, deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * x - s * y, s * x + c * y


class Region:
    """A set of grid cells kept 4-connected, hole-free and free of corner pinches."""

    def __init__(self, inside: set):
        self.inside = inside
        self.cells: set = set()

    def _ring(self, c):
        x, y = c
        return [(x + dx, y + dy) in self.cells for dx, dy in RING]

    def can_add(self, c) -> bool:
        if c in self.cells or c not in self.inside:
            return False
        if not self.cells:
            return True
        r = self._ring(c)
        edge_in = [r[0], r[2], r[4], r[6]]
        if not any(edge_in):
            return False
        # a diagonal neighbour touching c only at a corner -> pinch
        for k in (1, 3, 5, 7):
            if r[k] and not r[k - 1] and not r[(k + 1) % 8]:
                return False
        return self._one_component(r)

    def can_remove(self, c) -> bool:
        if c not in self.cells or len(self.cells) <= 1:
            return False
        r = self._ring(c)
        if all(r[k] for k in (0, 2, 4, 6)):
            return False  # interior cell: removing it makes a hole
        # removing c must not leave two cells touching only at a corner
        for k in (1, 3, 5, 7):
            if not r[k] and r[k - 1] and r[(k + 1) % 8]:
                return False
        return self._one_component(r)

    @staticmethod
    def _one_component(r) -> bool:
        """Are the in-region edge neighbours connected around c (through the
        corner cells)?"""
        edges = [k for k in (0, 2, 4, 6) if r[k]]
        if len(edges) <= 1:
            return True
        parent = {k: k for k in edges}

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        for k in edges:
            nxt = (k + 2) % 8
            if r[nxt] and r[(k + 1) % 8]:
                parent[find(k)] = find(nxt)
        return len({find(k) for k in edges}) == 1

    def add(self, c):
        self.cells.add(c)

    def remove(self, c):
        self.cells.discard(c)

    # ---------------------------------------------------------------- outline

    def outline(self):
        """Boundary edges in counter-clockwise order: (dir, start_vertex, inner, outer).
        dir is in 45-degree units (0 E, 2 N, 4 W, 6 S). None if not one loop."""
        nxt = {}
        for x, y in self.cells:
            if (x, y - 1) not in self.cells:
                nxt[(x, y)] = (0, (x + 1, y), (x, y), (x, y - 1))
            if (x + 1, y) not in self.cells:
                nxt[(x + 1, y)] = (2, (x + 1, y + 1), (x, y), (x + 1, y))
            if (x, y + 1) not in self.cells:
                nxt[(x + 1, y + 1)] = (4, (x, y + 1), (x, y), (x, y + 1))
            if (x - 1, y) not in self.cells:
                nxt[(x, y + 1)] = (6, (x, y), (x, y), (x - 1, y))
        if not nxt:
            return None
        start = min(nxt)
        edges = []
        v = start
        while True:
            d, w, inner, outer = nxt[v]
            edges.append((d, v, inner, outer))
            v = w
            if v == start:
                break
            if len(edges) > len(nxt):
                return None
        if len(edges) != len(nxt):
            return None  # more than one loop: a hole somewhere
        return edges

    @staticmethod
    def runs(edges):
        """Group the outline into straight runs: list of lists of edges."""
        # rotate so that edges[0] starts a new run
        n = len(edges)
        k = 0
        while k < n and edges[k][0] == edges[k - 1][0]:
            k += 1
        if k == n:
            return None
        edges = edges[k:] + edges[:k]
        out = [[edges[0]]]
        for e in edges[1:]:
            if e[0] == out[-1][-1][0]:
                out[-1].append(e)
            else:
                out.append([e])
        return out


def _area(pts) -> float:
    n = len(pts)
    return (
        sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )
        / 2
    )


def _clip(pts, box):
    """Sutherland-Hodgman: the polygon cut down to an axis-aligned box."""
    x0, y0, x1, y1 = box

    def clip_edge(poly, inside, cross):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i - 1], poly[i]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(cross(a, b))
                out.append(b)
            elif ia:
                out.append(cross(a, b))
        return out

    def at_x(xv):
        return lambda a, b: (xv, a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]))

    def at_y(yv):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yv - a[1]) / (b[1] - a[1]), yv)

    poly = list(pts)
    for inside, cross in (
        (lambda p: p[0] >= x0, at_x(x0)),
        (lambda p: p[0] <= x1, at_x(x1)),
        (lambda p: p[1] >= y0, at_y(y0)),
        (lambda p: p[1] <= y1, at_y(y1)),
    ):
        if not poly:
            break
        poly = clip_edge(poly, inside, cross)
    return poly


class Grid:
    """Cut positions along x and y (in the rotated, local frame) and which of the
    resulting cells lie completely inside the room."""

    def __init__(self, theta: float, xc: list[float], yc: list[float], inside: set):
        self.theta = theta
        self.xc = xc
        self.yc = yc
        self.w = [xc[i + 1] - xc[i] for i in range(len(xc) - 1)]
        self.h = [yc[j + 1] - yc[j] for j in range(len(yc) - 1)]
        self.inside = inside

    def world(self, gx: float, gy: float) -> tuple[float, float]:
        return _rot(gx, gy, self.theta)

    def cell_corners(self, c):
        i, j = c
        return [
            self.world(self.xc[i + a], self.yc[j + b])
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]

    def edge_len(self, d: int, cell) -> float:
        return self.w[cell[0]] if d in (0, 4) else self.h[cell[1]]


class GridEngine:
    def __init__(self, placer, allocator, connectors, weights, out_of_time):
        self.placer = placer
        self.alloc = allocator
        self.right = connectors["right"]
        self.weights = weights
        self.out_of_time = out_of_time
        self.total_length = sum(n * k for n, k in allocator.pool.items())
        self.side_lengths = [n for n in range(5, 31) if allocator.side_ok(n)]

    # ------------------------------------------------------------- set-up

    def frame_angles(self, limit: int = 3) -> list[float]:
        weight: dict[float, float] = {}
        for ang, length in zip(self.placer.out_angle, self.placer.edge_len):
            a = round(ang % 90.0, 6)
            if a > 90.0 - 1e-6:
                a = 0.0
            weight[a] = weight.get(a, 0.0) + length
        return [a for a, _ in sorted(weight.items(), key=lambda t: -t[1])[:limit]]

    def cell_sizes(self, limit: int = 3) -> list[int]:
        sizes = []
        for s in range(5, 16):
            mults = [k * s for k in range(1, 30 // s + 1)]
            if not all(self.alloc.side_ok(m) for m in mults):
                continue
            if not any(self.alloc.gate_options(m) for m in mults):
                continue
            sizes.append(s)
            if len(sizes) >= limit:
                break
        return sizes

    def _uniform_cuts(self, lo: float, hi: float, s: int, off: float) -> list[float]:
        start = off + s * math.floor((lo - off) / s + 1e-9)
        cuts = [start]
        while cuts[-1] < hi - 1e-9:
            cuts.append(cuts[-1] + s)
        return cuts

    @staticmethod
    def _cross(pts, y: float) -> list[tuple[float, float]]:
        """Intervals of x where the horizontal line at height y is inside."""
        xs = []
        n = len(pts)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            if (ay > y) != (by > y):
                xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
        xs.sort()
        return list(zip(xs[::2], xs[1::2]))

    @staticmethod
    def _intersect(a, b):
        out = []
        i = j = 0
        while i < len(a) and j < len(b):
            lo = max(a[i][0], b[j][0])
            hi = min(a[i][1], b[j][1])
            if hi > lo:
                out.append((lo, hi))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        return out

    def _slab_cuts(self, pts, gmax: int):
        """Cut positions along y (gaps = buildable side lengths <= gmax), chosen by
        dynamic programming to maximise the room area that lies in full-height
        columns of each slab -- so slabs line up with the room's bars and arms."""
        gaps = [g for g in self.side_lengths if g <= gmax]
        if not gaps:
            return None
        ys = [p[1] for p in pts]
        lo, hi = min(ys), max(ys)
        weight: dict[float, float] = {}
        for y in ys:
            f = round(y - math.floor(y), 6) % 1.0
            weight[f] = weight.get(f, 0.0) + 1
        frac = max(weight.items(), key=lambda t: t[1])[0]
        base = math.floor(lo) + frac
        if base > lo + 1e-9:
            base -= 1
        span = int(math.ceil(hi - base - 1e-9))
        if span <= 0 or span > 2000:
            return None
        vert_y = sorted(set(ys))
        eps = 1e-7
        cache = {}

        def value(a: int, b: int) -> float:
            key = (a, b)
            if key in cache:
                return cache[key]
            ya, yb = base + a, base + b
            iv = self._cross(pts, ya + eps)
            iv = self._intersect(iv, self._cross(pts, yb - eps))
            k = bisect.bisect_right(vert_y, ya)
            while iv and k < len(vert_y) and vert_y[k] < yb:
                yv = vert_y[k]
                iv = self._intersect(iv, self._cross(pts, yv - eps))
                iv = self._intersect(iv, self._cross(pts, yv + eps))
                k += 1
            v = (yb - ya) * sum(hi - lo for lo, hi in iv)
            cache[key] = v
            return v

        neg = -1.0
        top = span + max(gaps)
        dp = [neg] * (top + 1)
        back = [-1] * (top + 1)
        dp[0] = 0.0
        for t in range(1, top + 1):
            for g in gaps:
                if t - g >= 0 and dp[t - g] >= 0:
                    val = dp[t - g] + \
                        (value(t - g, t) if t - g < span else 0.0)
                    if val > dp[t] + 1e-9:
                        dp[t] = val
                        back[t] = t - g
        ends = [t for t in range(span, top + 1) if dp[t] >= 0]
        if not ends:
            return None
        t = max(ends, key=lambda e: (dp[e], -e))
        cuts = []
        while t >= 0:
            cuts.append(base + t)
            t = back[t] if t > 0 else -1
        cuts.reverse()
        return cuts

    def grids(self):
        """Grids to try: room-aligned adaptive cuts, then uniform cells."""
        sizes = self.cell_sizes()
        for theta in self.frame_angles():
            local = [_rot(x, y, -theta) for x, y in self.placer.pts]
            xs = [p[0] for p in local]
            ys = [p[1] for p in local]
            swapped = [(y, x) for x, y in local]
            for gmax in (10, 14):
                yc = self._slab_cuts(local, gmax)
                xc = self._slab_cuts(swapped, gmax)
                if xc and yc:
                    yield theta, xc, yc
            # big rooms: also grids fitted to one window of the room at a time, so
            # the cut lines can follow that part's arms instead of a compromise
            yield from self._window_grids(theta, local)
            for s in sizes:
                for ox, oy in self._offsets(local, s):
                    yield (
                        theta,
                        self._uniform_cuts(min(xs), max(xs), s, ox),
                        self._uniform_cuts(min(ys), max(ys), s, oy),
                    )

    def _window_grids(self, theta, local, size: float = 56.0, step: float = 28.0):
        xs = [p[0] for p in local]
        ys = [p[1] for p in local]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        if maxx - minx <= size + 1e-6 and maxy - miny <= size + 1e-6:
            return
        windows = []
        wx = minx
        while True:
            wy = miny
            while True:
                box = (wx, wy, min(wx + size, maxx), min(wy + size, maxy))
                clipped = _clip(local, box)
                if len(clipped) >= 3:
                    area = abs(_area(clipped))
                    if area > 50:
                        windows.append((area, box, clipped))
                if wy + size >= maxy - 1e-6:
                    break
                wy += step
            if wx + size >= maxx - 1e-6:
                break
            wx += step
        windows.sort(key=lambda t: -t[0])
        for _, box, clipped in windows[:10]:
            swapped = [(y, x) for x, y in clipped]
            yc = self._slab_cuts(clipped, 10)
            xc = self._slab_cuts(swapped, 10)
            if xc and yc:
                yield theta, xc, yc

    def _offsets(self, local, s):
        def best_offsets(values):
            count: dict[float, int] = {}
            for v in values:
                r = round(v % s, 6)
                if r > s - 1e-6:
                    r = 0.0
                count[r] = count.get(r, 0) + 1
            return [o for o, _ in sorted(count.items(), key=lambda t: -t[1])[:2]]

        xs = best_offsets([p[0] for p in local])
        ys = best_offsets([p[1] for p in local])
        return [(x, y) for x in xs for y in ys][:3]

    def rasterize(self, theta: float, xc: list[float], yc: list[float]):
        nx, ny = len(xc) - 1, len(yc) - 1
        if nx <= 0 or ny <= 0 or nx * ny > 60000:
            return None
        local = [_rot(x, y, -theta) for x, y in self.placer.pts]
        eps = 1e-7
        blocked = set()
        n = len(local)
        step = (
            min(
                min(xc[i + 1] - xc[i] for i in range(nx)),
                min(yc[j + 1] - yc[j] for j in range(ny)),
            )
            / 24.0
        )

        def index(cuts, v):
            k = bisect.bisect_right(cuts, v) - 1
            on = (0 <= k < len(cuts) and abs(v - cuts[k]) < eps) or (
                k + 1 < len(cuts) and abs(cuts[k + 1] - v) < eps
            )
            return k, on

        for i in range(n):
            ax, ay = local[i]
            bx, by = local[(i + 1) % n]
            steps = max(1, int(math.hypot(bx - ax, by - ay) / step) + 1)
            for k in range(steps + 1):
                t = k / steps
                ci, on_u = index(xc, ax + (bx - ax) * t)
                cj, on_v = index(yc, ay + (by - ay) * t)
                if not on_u and not on_v:
                    blocked.add((ci, cj))
        inside = set()
        for j in range(ny):
            ym = (yc[j] + yc[j + 1]) / 2
            cross = []
            for i in range(n):
                ax, ay = local[i]
                bx, by = local[(i + 1) % n]
                if (ay > ym) != (by > ym):
                    cross.append(ax + (ym - ay) * (bx - ax) / (by - ay))
            cross.sort()
            for a, b in zip(cross[::2], cross[1::2]):
                for i in range(nx):
                    xm = (xc[i] + xc[i + 1]) / 2
                    if a < xm < b and (i, j) not in blocked:
                        inside.add((i, j))
        return Grid(theta, xc, yc, inside)

    # ------------------------------------------------------------- growing

    def seeds(self, grid: Grid, count: int = 2):
        inside = grid.inside
        dist = {}
        frontier = [
            c
            for c in inside
            if any((c[0] + dx, c[1] + dy) not in inside for dx, dy in RING)
        ]
        for c in frontier:
            dist[c] = 1
        k = 0
        while k < len(frontier):
            c = frontier[k]
            k += 1
            for dx, dy in RING:
                d = (c[0] + dx, c[1] + dy)
                if d in inside and d not in dist:
                    dist[d] = dist[c] + 1
                    frontier.append(d)

        def size(c):
            return grid.w[c[0]] * grid.h[c[1]]

        ranked = sorted(inside, key=lambda c: (-dist.get(c, 0), -size(c), c))
        out = []
        for c in ranked:
            if all(
                max(abs(c[0] - o[0]), abs(c[1] - o[1])) > 2 * dist.get(o, 1)
                for o in out
            ):
                out.append(c)
            if len(out) >= count:
                break
        return out

    def grow(self, grid: Grid, seed, max_corners: int):
        region = Region(grid.inside)
        sx = (grid.xc[seed[0]] + grid.xc[seed[0] + 1]) / 2
        sy = (grid.yc[seed[1]] + grid.yc[seed[1] + 1]) / 2

        def pri(c):
            cx = (grid.xc[c[0]] + grid.xc[c[0] + 1]) / 2
            cy = (grid.yc[c[1]] + grid.yc[c[1] + 1]) / 2
            return max(abs(cx - sx), abs(cy - sy))

        order = []
        corners = 0
        area = 0.0
        perim = 0.0
        heap = [(0.0, 0, seed)]
        queued = {seed}
        stats = []
        while heap:
            _, tries, c = heapq.heappop(heap)
            if c in region.cells:
                continue
            if not region.can_add(c):
                if tries < 3:
                    heapq.heappush(
                        heap, (pri(c) + 10.0 * (tries + 1), tries + 1, c))
                continue
            before = self._corner_count(region, c)
            i, j = c
            w, h = grid.w[i], grid.h[j]
            for (dx, dy), length in (
                ((1, 0), h),
                ((-1, 0), h),
                ((0, 1), w),
                ((0, -1), w),
            ):
                perim += -length if (i + dx, j +
                                     dy) in region.cells else length
            region.add(c)
            corners += self._corner_count(region, c) - before
            area += w * h
            order.append(c)
            stats.append((len(order), area, perim, corners))
            if corners > max_corners:
                break
            for dx, dy in N4:
                d = (i + dx, j + dy)
                if d in grid.inside and d not in region.cells and d not in queued:
                    queued.add(d)
                    heapq.heappush(heap, (pri(d), 0, d))
        return order, stats

    @staticmethod
    def _corner_count(region: Region, c) -> int:
        x, y = c
        total = 0
        cells = region.cells
        for vx, vy in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)):
            k = (
                ((vx - 1, vy - 1) in cells)
                + ((vx, vy - 1) in cells)
                + ((vx - 1, vy) in cells)
                + ((vx, vy) in cells)
            )
            if k in (1, 3):
                total += 1
        return total

    # ------------------------------------------------------------- finishing

    def break_long_runs(self, region: Region, grid: Grid) -> bool:
        for _ in range(4 * len(region.cells) + 50):
            if self.out_of_time():
                return False
            edges = region.outline()
            if edges is None:
                return False
            runs = Region.runs(edges)
            if runs is None:
                return False
            run = None
            for r in runs:
                if sum(grid.edge_len(e[0], e[2]) for e in r) > 30 + 1e-6:
                    run = r
                    break
            if run is None:
                return True
            # the longest prefix that is still <= 30 tells us where to break
            prefix = 0.0
            last_ok = 0
            for k, e in enumerate(run):
                prefix += grid.edge_len(e[0], e[2])
                if prefix > 30 + 1e-6:
                    break
                last_ok = k
            positions = list(range(min(last_ok + 1, len(run) - 2), 0, -1))
            fixed = False
            for k in positions:
                _, _, inner, outer = run[k]
                if region.can_add(outer):
                    region.add(outer)
                    fixed = True
                    break
                if region.can_remove(inner):
                    region.remove(inner)
                    fixed = True
                    break
            if not fixed:
                fixed = self._cut_off(region, grid, run)
            if not fixed:
                return False
        return False

    def _cut_off(self, region: Region, grid: Grid, run) -> bool:
        """Last resort for a run that can't be bumped or notched: remove one of the
        cells along it and drop whatever part of the blob that disconnects,
        choosing the cut that loses the least area."""
        best = None
        for k in range(len(run)):
            c = run[k][2]
            rest = set(region.cells)
            rest.discard(c)
            if not rest:
                continue
            comps = self._components(rest)
            keep = max(
                comps, key=lambda comp: sum(
                    grid.w[x] * grid.h[y] for x, y in comp)
            )
            lost = sum(grid.w[x] * grid.h[y] for x, y in region.cells - keep)
            trial = Region(region.inside)
            trial.cells = set(keep)
            if trial.outline() is None or self._has_pinch(keep):
                continue
            if best is None or lost < best[0]:
                best = (lost, keep)
        if best is None:
            return False
        region.cells = set(best[1])
        return True

    @staticmethod
    def _components(cells: set) -> list[set]:
        left = set(cells)
        out = []
        while left:
            start = left.pop()
            comp = {start}
            stack = [start]
            while stack:
                x, y = stack.pop()
                for dx, dy in N4:
                    d = (x + dx, y + dy)
                    if d in left:
                        left.discard(d)
                        comp.add(d)
                        stack.append(d)
            out.append(comp)
        return out

    @staticmethod
    def _has_pinch(cells: set) -> bool:
        for x, y in cells:
            if (
                (x + 1, y + 1) in cells
                and (x + 1, y) not in cells
                and (x, y + 1) not in cells
            ):
                return True
            if (
                (x - 1, y + 1) in cells
                and (x - 1, y) not in cells
                and (x, y + 1) not in cells
            ):
                return True
        return False

    def to_shape(self, region: Region, grid: Grid):
        edges = region.outline()
        if edges is None:
            return None
        runs = Region.runs(edges)
        if runs is None:
            return None
        dirs = [r[0][0] for r in runs]
        lengths = []
        for r in runs:
            total = sum(grid.edge_len(e[0], e[2]) for e in r)
            if abs(total - round(total)) > 1e-6:
                return None
            lengths.append(int(round(total)))
        if any(n < 5 or n > 30 for n in lengths):
            return None
        shape = Shape("grid", (len(region.cells),), dirs, lengths)
        if not shape.valid():
            return None
        vx, vy = runs[0][0][1]
        tx, ty = grid.world(grid.xc[vx], grid.yc[vy])
        return shape, tx, ty

    # ------------------------------------------------------------- driver

    def run(self, try_shape, best_score):
        """`try_shape(shape, theta, tx, ty)` returns (score, ok, reason)."""
        w = self.weights
        best = best_score
        if self.right < 4:
            return
        for theta, xc, yc in self.grids():
            if self.out_of_time():
                return
            grid = self.rasterize(theta, xc, yc)
            if grid is None or len(grid.inside) < 1:
                continue
            for seed in self.seeds(grid):
                if self.out_of_time():
                    return
                order, stats = self.grow(grid, seed, self.right)
                est = []
                for k, area, perim, corners in stats:
                    if corners > self.right or perim > self.total_length:
                        continue
                    est.append((1000 + w.A * area + w.C * perim, k))
                if not est:
                    continue
                est.sort(reverse=True)
                if est[0][0] + w.G <= best:
                    continue

                def evaluate(k):
                    return self._evaluate(order[:k], grid, try_shape)

                k_best = est[0][1]
                res = evaluate(k_best)
                if res is not None:
                    best = max(best, res)
                    continue
                lo, hi, n_eval = 1, k_best - 1, 0
                while lo <= hi and n_eval < 12 and not self.out_of_time():
                    mid = (lo + hi + 1) // 2
                    n_eval += 1
                    res = evaluate(mid)
                    if res is None:
                        hi = mid - 1
                    else:
                        best = max(best, res)
                        lo = mid + 1

    def _evaluate(self, cells, grid: Grid, try_shape):
        inside = set(grid.inside)
        region = Region(inside)
        for c in cells:
            region.add(c)
        for _attempt in range(4):
            if not self.break_long_runs(region, grid):
                return None
            got = self.to_shape(region, grid)
            if got is None:
                return None
            shape, tx, ty = got
            if shape.right > self.right:
                return None
            score, ok, reason = try_shape(shape, grid.theta, tx, ty)
            if ok:
                return score
            if "fit" not in (reason or ""):
                return None
            # the raster was a little optimistic somewhere: drop cells that poke out
            bad = [
                c for c in region.cells if not self.placer.fits(grid.cell_corners(c))
            ]
            if not bad:
                return None
            for c in bad:
                inside.discard(c)
                if region.can_remove(c):
                    region.remove(c)
                else:
                    return None
        return None

# ============================================================================
# Player
# ============================================================================


TIME_BUDGET = 50.0
TEMPLATE_SHARE = 0.4
GRID_SHARE = 0.70
TEMPLATE_RESUME_SHARE = 0.85
N_SEEDS = 4
MAX_FAILED_MEMO = 400


class Player8(Player):
    """Group 8: score-driven search over many enclosure shapes (rectangles, cut-corner
    octagons/hexagons, L/U/T shapes, and grid-traced outlines for big rooms), each
    side built from real pieces by a backtracking allocator, placed by fitting the
    shape's corners into the room's corners, and checked with the simulator's own
    validator. Returns the highest-scoring valid enclosure, or None if nothing
    beats 0."""

    def build_enclosure(self) -> Construction | None:
        self._t0 = time.process_time()
        self._deadline = TIME_BUDGET
        inv = self.inventory
        self.conn = {
            "right": inv.connectors.get(ConnectorType.RIGHT, 0),
            "diagonal": inv.connectors.get(ConnectorType.DIAGONAL, 0),
            "straight": inv.connectors.get(ConnectorType.STRAIGHT, 0),
        }
        self.alloc = Allocator(inv.walls, inv.gates, self.conn["straight"])
        if not self.alloc.gates:
            self.note = (
                "No gate of length >= 5 in the inventory -- nothing valid exists"
            )
            return None
        self.placer = Placer(self.room, self._out_of_time)
        self.best_score = 0.0
        self.best: Construction | None = None
        self.best_desc = ""
        two_gates = self.weights.G > 0 and sum(self.alloc.gates.values()) >= 2
        self.n_gates_max = 2 if two_gates else 1
        self.alloc.single_gate_pool = not two_gates

        # good right-angled enclosures to grow from
        self.seeds: list[tuple] = []
        try:
            self._deadline = TIME_BUDGET * TEMPLATE_SHARE
            self._templates()
            self._deadline = TIME_BUDGET * GRID_SHARE
            GridEngine(
                self.placer, self.alloc, self.conn, self.weights, self._out_of_time
            ).run(self._try_grid_shape, self.best_score)
            self._deadline = TIME_BUDGET
            self._refine()
            # shapes the first pass had no time for get whatever is left
            self._deadline = TIME_BUDGET * TEMPLATE_RESUME_SHARE
            self._templates()
            self._deadline = TIME_BUDGET
            self._explore()
        except Exception as e:  # never crash the run: keep what we already have
            self.note = f"search stopped early ({type(e).__name__}: {e})"

        if self.best is None:
            self.note = self.note or "No valid enclosure with a positive score found"
            return None
        self.note = (
            f"{self.best_desc}, expected score {self.best_score:.1f} "
            f"(cpu {time.process_time() - self._t0:.1f}s)"
        )
        return self.best

    # ------------------------------------------------------------------ helpers

    def _out_of_time(self) -> bool:
        return time.process_time() - self._t0 > self._deadline

    def _score(self, area: float, perimeter: float, bonus: bool) -> float:
        w = self.weights
        return 1000.0 + w.A * area + w.C * perimeter + (w.G if bonus else 0.0)

    def _connectors_ok(self, shape) -> bool:
        return (
            shape.right <= self.conn["right"]
            and shape.diagonal <= self.conn["diagonal"]
        )

    def _commit(self, shape, pieces, theta, tx, ty, desc) -> tuple[float, bool, str]:
        """Build, validate with the simulator's own rules, keep if best."""
        construction = shape.construction(pieces, theta, tx, ty)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return 0.0, False, result.reason
        score = score_construction(result, self.weights)
        if all(d % 2 == 0 for d in shape.dirs):
            self.seeds.append((score, shape, theta, tx, ty))
            self.seeds.sort(key=lambda t: -t[0])
            del self.seeds[N_SEEDS * 3:]
        if score > self.best_score:
            self.best_score = score
            self.best = construction
            self.best_desc = desc
        return score, True, "valid"

    def _gate_counts(self) -> list[int]:
        return [2, 1] if self.n_gates_max == 2 else [1]

    # ---------------------------------------------------------------- templates

    def _side_lengths(self) -> list[int]:
        return [n for n in range(5, 31) if self.alloc.side_ok(n)]

    def _template_candidates(self):
        """(upper-bound score, family, params) for every template the connector
        stock allows, best first."""
        L = self._side_lengths()
        if not L:
            return []
        Ls = set(L)
        minx, miny, maxx, maxy = self.room.polygon.bounds
        room_diag = math.hypot(maxx - minx, maxy - miny) + 1e-6
        room_area = self.room.polygon.area + 1e-6
        right, diag = self.conn["right"], self.conn["diagonal"]
        G = self.weights.G if self.n_gates_max == 2 else 0.0
        out = []
        r2 = math.sqrt(0.5)

        max_r = self.placer.max_radius

        def add(family, params, area, perim, diameter, radius):
            if area > room_area or diameter > room_diag or radius > max_r:
                return
            sc = self._score(area, perim, False) + G
            if sc > self.best_score:
                out.append((sc, family, params))

        if right >= 4:
            for i, a in enumerate(L):
                for b in L[i:]:
                    add("rect", (a, b), a * b, 2 *
                        (a + b), math.hypot(a, b), a / 2)
        if diag >= 8:
            for a in L:
                for b in L:
                    for c in L:
                        for d in L:
                            k = (c + d) * r2
                            W, H = a + k, b + k
                            area = W * H - (c * c + d * d) / 2
                            perim = 2 * (a + b + c + d)
                            r = min(W, H, (W + H - max(c, d) * 2 * r2) * r2) / 2
                            dm = math.hypot(W, H)
                            add("oct", (a, b, c, d), area, perim, dm, r)
        if right >= 2 and diag >= 4:
            for a in L:
                for b in L:
                    for c in L:
                        k = c * r2
                        W, H = a + k, b + k
                        area = W * H - c * c / 2
                        perim = 2 * (a + b + c)
                        r = min(W, H, (W + H - c * 2 * r2) * r2) / 2
                        dm = math.hypot(W, H)
                        add("hex", (a, b, c), area, perim, dm, r)
                        add("hex2", (a, b, c), area, perim, dm, r)
        if right >= 6:
            for a1 in L:
                for a2 in L:
                    if a1 + a2 not in Ls:
                        continue
                    for b1 in L:
                        for b2 in L:
                            if b1 + b2 not in Ls:
                                continue
                            W, H = a1 + a2, b1 + b2
                            area = W * b1 + a1 * b2
                            r = max(min(W, b1), min(a1, H)) / 2
                            dm = math.hypot(W, H)
                            add("L", (a1, a2, b1, b2), area, 2 * (W + H), dm, r)
        if right >= 8:
            for a1 in L:
                for a2 in L:
                    for a3 in L:
                        W = a1 + a2 + a3
                        if W not in Ls:
                            continue
                        for b1 in L:
                            for b2 in L:
                                if b1 + b2 not in Ls:
                                    continue
                                H = b1 + b2
                                d = math.hypot(W, H)
                                p = (a1, a2, a3, b1, b2)
                                u_area = W * b1 + (a1 + a3) * b2
                                u_r = max(min(W, b1), min(max(a1, a3), H)) / 2
                                t_r = max(min(W, b1), min(a2, H)) / 2
                                add("U", p, u_area, 2 * (W + H) + 2 * b2, d, u_r)
                                add("T", p, W * b1 + a2 *
                                    b2, 2 * (W + H), d, t_r)
        out.sort(key=lambda t: -t[0])
        return out

    @staticmethod
    def _make(family, p):
        if family == "rect":
            return rectangle(*p)
        if family == "oct":
            return octagon(*p)
        if family == "hex":
            return hexagon(*p)
        if family == "hex2":
            a, b, c = p
            return Shape("hex2", p, [0, 2, 3, 4, 6, 7], [a, b, c, a, b, c])
        if family == "L":
            return l_shape(*p)
        if family == "U":
            return u_shape(*p)
        if family == "T":
            return t_shape(*p)
        raise ValueError(family)

    def _templates(self):
        """Work through the template candidates, best upper bound first. Resumable:
        if time runs out the position is kept, so leftover time later in the run
        can carry on where this stopped."""
        if not hasattr(self, "_tc"):
            self._tc = self._template_candidates()
            self._ti = 0
            self._tfailed: dict[str, list[tuple]] = {}
        failed = self._tfailed
        while self._ti < len(self._tc):
            upper, family, params = self._tc[self._ti]
            if upper <= self.best_score:
                # sorted: nothing left can beat our best
                self._ti = len(self._tc)
                return
            if self._out_of_time():
                return
            self._ti += 1
            key = tuple(sorted(params)) if family == "rect" else params
            if any(
                all(k >= f for k, f in zip(key, fp)) for fp in failed.get(family, ())
            ):
                continue  # a smaller version of this shape already failed to fit
            shape = self._make(family, params)
            if not shape.valid() or not self._connectors_ok(shape):
                continue
            placement = None
            tried_placing = False
            for n_g in self._gate_counts():
                est = self._score(shape.area, shape.perimeter, n_g == 2)
                if est <= self.best_score:
                    continue
                pieces = self.alloc.allocate(shape.lengths, n_g)
                if pieces is None:
                    continue
                if not tried_placing:
                    tried_placing = True
                    placement = self.placer.place(shape)
                if placement is None:
                    if self._out_of_time():
                        self._ti -= 1  # gave up for lack of time, not a real miss
                        return
                    lst = failed.setdefault(family, [])
                    if len(lst) < MAX_FAILED_MEMO:
                        lst.append(key)
                    break
                theta, tx, ty = placement
                desc = f"{family} {params}"
                if n_g == 2:
                    desc += " with 2 gates"
                self._commit(shape, pieces, theta, tx, ty, desc)

    # --------------------------------------------------------------------- grid

    def _try_grid_shape(self, shape, theta, tx, ty):
        if not self._connectors_ok(shape):
            return 0.0, False, "connectors"
        reason = "pieces"
        for n_g in self._gate_counts():
            est = self._score(shape.area, shape.perimeter, n_g == 2)
            if est <= self.best_score:
                # cannot beat what we have; report "ok" so the engine stops shrinking
                return est, True, "worse"
            pieces = self.alloc.allocate(shape.lengths, n_g)
            if pieces is None:
                continue
            desc = f"grid outline ({len(shape.lengths)} sides)"
            if n_g == 2:
                desc += " with 2 gates"
            score, ok, reason = self._commit(
                shape, pieces, theta, tx, ty, desc)
            if ok:
                return score, ok, reason
        return 0.0, False, reason

    # ------------------------------------------------------------------- refine

    def _refine(self):
        if self.conn["right"] < 6:
            return
        refiner = Refiner(
            self.placer,
            self._side_lengths(),
            self.weights,
            self.conn["right"],
            self._out_of_time,
        )
        seeds, seen = [], set()
        for item in list(self.seeds):
            key = (item[1].family, tuple(item[1].lengths))
            if key not in seen:
                seen.add(key)
                seeds.append(item)
        for _, shape, theta, tx, ty in seeds[:N_SEEDS]:
            if self._out_of_time():
                return
            self._grow(refiner, shape, theta, tx, ty)

    def _grow(self, refiner, shape, theta, tx, ty, rng=None, top_k=1) -> bool:
        """Grow `shape` by bumps, committing every valid step. False = out of time."""
        fails = 0
        for grown, th, gx, gy in refiner.refine(
            shape, theta, tx, ty, rng=rng, top_k=top_k
        ):
            if self._out_of_time():
                return False
            if not self._connectors_ok(grown):
                break
            ok = False
            for n_g in self._gate_counts():
                pieces = self.alloc.allocate(grown.lengths, n_g)
                if pieces is None:
                    continue
                desc = f"{shape.family} grown by bumps ({len(grown.lengths)} sides)"
                if n_g == 2:
                    desc += " with 2 gates"
                _, ok, _ = self._commit(grown, pieces, th, gx, gy, desc)
                if ok:
                    break
            fails = 0 if ok else fails + 1
            if fails >= 3:
                break
        return True

    # ------------------------------------------------------------------ explore

    def _explore(self):
        """Use the time that is left for randomized restarts: drop a small square at
        a random spot and orientation, then grow it by randomly chosen bumps. A
        single greedy growth gets stuck in one local optimum; many different growth
        orders find the winding shapes (star arms, letter strokes) it misses."""
        if self.conn["right"] < 6 or self._out_of_time():
            return
        L = self._side_lengths()
        if not L:
            return
        rng = random.Random(8)
        refiner = Refiner(
            self.placer, L, self.weights, self.conn["right"], self._out_of_time
        )
        angles = sorted({round(a % 90.0, 4)
                        for a in self.placer.out_angle}) or [0.0]
        minx, miny, maxx, maxy = self.placer.bounds
        small = L[:4]
        misses = 0
        while not self._out_of_time():
            start = None
            for _ in range(60):
                a, b = rng.choice(small), rng.choice(small)
                shape = rectangle(a, b)
                if not shape.valid() or not self._connectors_ok(shape):
                    continue
                px = rng.uniform(minx, maxx)
                py = rng.uniform(miny, maxy)
                if not self.placer._point_ok(px, py):
                    continue
                theta = rng.choice(angles) + 90.0 * rng.randrange(4)
                c, s_ = math.cos(math.radians(theta)), math.sin(
                    math.radians(theta))
                cx, cy = a / 2, b / 2
                tx, ty = px - (c * cx - s_ * cy), py - (s_ * cx + c * cy)
                if self.placer.fits(shape.world_vertices(theta, tx, ty)):
                    start = (shape, theta, tx, ty)
                    break
            if start is None:
                pool = [x for x in self.seeds if x[1].family !=
                        "rect" or x[0] > 0]
                if not pool:
                    misses += 1
                    if misses > 200:
                        return
                    continue
                _, shape, theta, tx, ty = rng.choice(pool)
                start = (shape, theta, tx, ty)
            if not self._grow(refiner, *start, rng=rng, top_k=rng.choice((2, 3, 5))):
                return
