//! Plane triangulation for imported CAD faces: ear clipping of a polygon with
//! holes (after the earcut algorithm: hole bridging, z-order hashed ear
//! tests, and the intersection-curing and splitting fallback passes), then a
//! triangle mesh that is made Delaunay by edge flips and can be refined by
//! inserting points. Edges on the polygon boundary are never flipped.

use glam::DVec2;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    /// Vertex index in the input.
    vertex: u32,
    point: DVec2,
    prev: usize,
    next: usize,
    z: u32,
    prev_z: Option<usize>,
    next_z: Option<usize>,
    steiner: bool,
}

struct Earcut {
    nodes: Vec<Node>,
    triangles: Vec<[u32; 3]>,
    min: DVec2,
    inverse_size: f64,
}

/// Triangulates the polygon `points[rings[0]..rings[1]]` with holes
/// `points[rings[i]..rings[i + 1]]` (the last ring runs to the end). Rings
/// may wind either way. Triangles are counter-clockwise.
pub(crate) fn earcut(points: &[DVec2], rings: &[usize]) -> Vec<[u32; 3]> {
    let mut ranges = Vec::new();
    for (index, start) in rings.iter().enumerate() {
        let end = rings.get(index + 1).copied().unwrap_or(points.len());
        if *start < end && end <= points.len() {
            ranges.push(*start..end);
        }
    }
    let Some(outer) = ranges.first().cloned() else {
        return Vec::new();
    };
    let mut state = Earcut {
        nodes: Vec::with_capacity(points.len() * 3 / 2 + 8),
        triangles: Vec::new(),
        min: DVec2::ZERO,
        inverse_size: 0.0,
    };
    let Some(mut outer_node) = state.linked_list(points, outer.clone(), true) else {
        return Vec::new();
    };
    if state.nodes[outer_node].next == state.nodes[outer_node].prev {
        return Vec::new();
    }
    if ranges.len() > 1 {
        outer_node = state.eliminate_holes(points, &ranges[1..], outer_node);
    }
    if points.len() > 80 {
        let mut min = points[outer.start];
        let mut max = min;
        for point in &points[outer] {
            min = min.min(*point);
            max = max.max(*point);
        }
        let size = (max - min).max_element();
        state.min = min;
        state.inverse_size = if size > 0.0 { 32767.0 / size } else { 0.0 };
    }
    state.earcut_linked(Some(outer_node), 0);
    state.triangles
}

fn area(p: DVec2, q: DVec2, r: DVec2) -> f64 {
    (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
}

fn point_in_triangle(a: DVec2, b: DVec2, c: DVec2, p: DVec2) -> bool {
    (c.x - p.x) * (a.y - p.y) >= (a.x - p.x) * (c.y - p.y)
        && (a.x - p.x) * (b.y - p.y) >= (b.x - p.x) * (a.y - p.y)
        && (b.x - p.x) * (c.y - p.y) >= (c.x - p.x) * (b.y - p.y)
}

fn sign(value: f64) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

fn on_segment(p: DVec2, q: DVec2, r: DVec2) -> bool {
    q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
}

fn intersects(p1: DVec2, q1: DVec2, p2: DVec2, q2: DVec2) -> bool {
    let o1 = sign(area(p1, q1, p2));
    let o2 = sign(area(p1, q1, q2));
    let o3 = sign(area(p2, q2, p1));
    let o4 = sign(area(p2, q2, q1));
    (o1 != o2 && o3 != o4)
        || (o1 == 0 && on_segment(p1, p2, q1))
        || (o2 == 0 && on_segment(p1, q2, q1))
        || (o3 == 0 && on_segment(p2, p1, q2))
        || (o4 == 0 && on_segment(p2, q1, q2))
}

fn z_order(point: DVec2, min: DVec2, inverse_size: f64) -> u32 {
    let spread = |value: f64| {
        let mut value = (value.max(0.0) as u32) & 0xFFFF;
        value = (value | (value << 8)) & 0x00FF_00FF;
        value = (value | (value << 4)) & 0x0F0F_0F0F;
        value = (value | (value << 2)) & 0x3333_3333;
        (value | (value << 1)) & 0x5555_5555
    };
    let x = spread((point.x - min.x) * inverse_size);
    let y = spread((point.y - min.y) * inverse_size);
    x | (y << 1)
}

impl Earcut {
    fn p(&self, node: usize) -> DVec2 {
        self.nodes[node].point
    }

    fn next(&self, node: usize) -> usize {
        self.nodes[node].next
    }

    fn prev(&self, node: usize) -> usize {
        self.nodes[node].prev
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        self.p(a) == self.p(b)
    }

    fn node_area(&self, a: usize, b: usize, c: usize) -> f64 {
        area(self.p(a), self.p(b), self.p(c))
    }

    fn insert_node(&mut self, vertex: u32, point: DVec2, last: Option<usize>) -> usize {
        let index = self.nodes.len();
        let mut node = Node {
            vertex,
            point,
            prev: index,
            next: index,
            z: 0,
            prev_z: None,
            next_z: None,
            steiner: false,
        };
        if let Some(last) = last {
            node.next = self.nodes[last].next;
            node.prev = last;
            let after = self.nodes[last].next;
            self.nodes.push(node);
            self.nodes[after].prev = index;
            self.nodes[last].next = index;
        } else {
            self.nodes.push(node);
        }
        index
    }

    fn remove_node(&mut self, node: usize) {
        let Node {
            prev,
            next,
            prev_z,
            next_z,
            ..
        } = self.nodes[node];
        self.nodes[next].prev = prev;
        self.nodes[prev].next = next;
        if let Some(prev_z) = prev_z {
            self.nodes[prev_z].next_z = next_z;
        }
        if let Some(next_z) = next_z {
            self.nodes[next_z].prev_z = prev_z;
        }
    }

    fn linked_list(
        &mut self,
        points: &[DVec2],
        range: std::ops::Range<usize>,
        counter_clockwise: bool,
    ) -> Option<usize> {
        let mut signed = 0.0;
        let mut previous = range.end.checked_sub(1)?;
        for index in range.clone() {
            signed +=
                (points[previous].x - points[index].x) * (points[index].y + points[previous].y);
            previous = index;
        }
        let mut last = None;
        if counter_clockwise == (signed > 0.0) {
            for index in range {
                last = Some(self.insert_node(index as u32, points[index], last));
            }
        } else {
            for index in range.rev() {
                last = Some(self.insert_node(index as u32, points[index], last));
            }
        }
        let mut last = last?;
        if self.equals(last, self.next(last)) {
            let next = self.next(last);
            self.remove_node(last);
            last = next;
        }
        Some(last)
    }

    fn filter_points(&mut self, start: usize, end: Option<usize>) -> usize {
        let mut end = end.unwrap_or(start);
        let mut p = start;
        let mut guard = self.nodes.len() * 4 + 16;
        loop {
            guard -= 1;
            if guard == 0 {
                return end;
            }
            let mut again = false;
            let node = self.nodes[p];
            if !node.steiner
                && (self.equals(p, node.next) || self.node_area(node.prev, p, node.next) == 0.0)
            {
                self.remove_node(p);
                p = node.prev;
                end = node.prev;
                if p == self.next(p) {
                    break;
                }
                again = true;
            } else {
                p = node.next;
            }
            if !again && p == end {
                break;
            }
        }
        end
    }

    fn earcut_linked(&mut self, ear: Option<usize>, pass: u8) {
        let Some(mut ear) = ear else {
            return;
        };
        if pass == 0 && self.inverse_size > 0.0 {
            self.index_curve(ear);
        }
        let mut stop = ear;
        let mut guard = self.nodes.len() * self.nodes.len().min(1 << 16) + 64;
        while self.prev(ear) != self.next(ear) {
            guard -= 1;
            if guard == 0 {
                return;
            }
            let prev = self.prev(ear);
            let next = self.next(ear);
            let is_ear = if self.inverse_size > 0.0 {
                self.is_ear_hashed(ear)
            } else {
                self.is_ear(ear)
            };
            if is_ear {
                self.triangles.push([
                    self.nodes[prev].vertex,
                    self.nodes[ear].vertex,
                    self.nodes[next].vertex,
                ]);
                self.remove_node(ear);
                ear = self.next(next);
                stop = ear;
                continue;
            }
            ear = next;
            if ear == stop {
                match pass {
                    0 => {
                        let filtered = self.filter_points(ear, None);
                        self.earcut_linked(Some(filtered), 1);
                    }
                    1 => {
                        let filtered = self.filter_points(ear, None);
                        let cured = self.cure_local_intersections(filtered);
                        self.earcut_linked(Some(cured), 2);
                    }
                    _ => self.split_earcut(ear),
                }
                break;
            }
        }
    }

    fn is_ear(&self, ear: usize) -> bool {
        let (a, b, c) = (self.prev(ear), ear, self.next(ear));
        let (pa, pb, pc) = (self.p(a), self.p(b), self.p(c));
        if area(pa, pb, pc) >= 0.0 {
            return false;
        }
        let min = pa.min(pb).min(pc);
        let max = pa.max(pb).max(pc);
        let mut p = self.next(c);
        while p != a {
            let point = self.p(p);
            if point.x >= min.x
                && point.x <= max.x
                && point.y >= min.y
                && point.y <= max.y
                && point_in_triangle(pa, pb, pc, point)
                && self.node_area(self.prev(p), p, self.next(p)) >= 0.0
            {
                return false;
            }
            p = self.next(p);
        }
        true
    }

    fn is_ear_hashed(&self, ear: usize) -> bool {
        let (a, b, c) = (self.prev(ear), ear, self.next(ear));
        let (pa, pb, pc) = (self.p(a), self.p(b), self.p(c));
        if area(pa, pb, pc) >= 0.0 {
            return false;
        }
        let min = pa.min(pb).min(pc);
        let max = pa.max(pb).max(pc);
        let min_z = z_order(min, self.min, self.inverse_size);
        let max_z = z_order(max, self.min, self.inverse_size);
        let blocks = |node: usize| {
            let point = self.p(node);
            point.x >= min.x
                && point.x <= max.x
                && point.y >= min.y
                && point.y <= max.y
                && node != a
                && node != c
                && point_in_triangle(pa, pb, pc, point)
                && self.node_area(self.prev(node), node, self.next(node)) >= 0.0
        };
        let mut p = self.nodes[ear].prev_z;
        let mut n = self.nodes[ear].next_z;
        while let (Some(pp), Some(nn)) = (p, n) {
            if self.nodes[pp].z < min_z || self.nodes[nn].z > max_z {
                break;
            }
            if blocks(pp) {
                return false;
            }
            p = self.nodes[pp].prev_z;
            if blocks(nn) {
                return false;
            }
            n = self.nodes[nn].next_z;
        }
        while let Some(pp) = p {
            if self.nodes[pp].z < min_z {
                break;
            }
            if blocks(pp) {
                return false;
            }
            p = self.nodes[pp].prev_z;
        }
        while let Some(nn) = n {
            if self.nodes[nn].z > max_z {
                break;
            }
            if blocks(nn) {
                return false;
            }
            n = self.nodes[nn].next_z;
        }
        true
    }

    fn cure_local_intersections(&mut self, start: usize) -> usize {
        let mut start = start;
        let mut p = start;
        let mut guard = self.nodes.len() * 2 + 16;
        loop {
            guard -= 1;
            if guard == 0 {
                break;
            }
            let a = self.prev(p);
            let b = self.next(self.next(p));
            if !self.equals(a, b)
                && intersects(self.p(a), self.p(p), self.p(self.next(p)), self.p(b))
                && self.locally_inside(a, b)
                && self.locally_inside(b, a)
            {
                self.triangles.push([
                    self.nodes[a].vertex,
                    self.nodes[p].vertex,
                    self.nodes[b].vertex,
                ]);
                let next = self.next(p);
                self.remove_node(p);
                self.remove_node(next);
                p = b;
                start = b;
            }
            p = self.next(p);
            if p == start {
                break;
            }
        }
        self.filter_points(p, None)
    }

    fn split_earcut(&mut self, start: usize) {
        let mut a = start;
        loop {
            let mut b = self.next(self.next(a));
            while b != self.prev(a) {
                if self.nodes[a].vertex != self.nodes[b].vertex && self.is_valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let a_next = self.next(a);
                    let a = self.filter_points(a, Some(a_next));
                    let c_next = self.next(c);
                    let c = self.filter_points(c, Some(c_next));
                    self.earcut_linked(Some(a), 0);
                    self.earcut_linked(Some(c), 0);
                    return;
                }
                b = self.next(b);
            }
            a = self.next(a);
            if a == start {
                return;
            }
        }
    }

    fn eliminate_holes(
        &mut self,
        points: &[DVec2],
        holes: &[std::ops::Range<usize>],
        mut outer: usize,
    ) -> usize {
        let mut queue = Vec::new();
        for range in holes {
            if let Some(list) = self.linked_list(points, range.clone(), false) {
                if list == self.next(list) {
                    self.nodes[list].steiner = true;
                }
                queue.push(self.leftmost(list));
            }
        }
        queue.sort_by(|a, b| self.p(*a).x.total_cmp(&self.p(*b).x));
        for hole in queue {
            outer = self.eliminate_hole(hole, outer);
        }
        outer
    }

    fn eliminate_hole(&mut self, hole: usize, outer: usize) -> usize {
        let Some(bridge) = self.find_hole_bridge(hole, outer) else {
            return outer;
        };
        let reverse = self.split_polygon(bridge, hole);
        let reverse_next = self.next(reverse);
        self.filter_points(reverse, Some(reverse_next));
        let bridge_next = self.next(bridge);
        self.filter_points(bridge, Some(bridge_next))
    }

    fn find_hole_bridge(&self, hole: usize, outer: usize) -> Option<usize> {
        let h = self.p(hole);
        let mut p = outer;
        let mut qx = f64::NEG_INFINITY;
        let mut m = None;
        loop {
            let a = self.p(p);
            let b = self.p(self.next(p));
            if h.y <= a.y && h.y >= b.y && b.y != a.y {
                let x = a.x + (h.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if x <= h.x && x > qx {
                    qx = x;
                    m = Some(if a.x < b.x { p } else { self.next(p) });
                    if x == h.x {
                        return m;
                    }
                }
            }
            p = self.next(p);
            if p == outer {
                break;
            }
        }
        let mut m = m?;
        let stop = m;
        let mp = self.p(m);
        let mut tan_min = f64::INFINITY;
        p = m;
        loop {
            let point = self.p(p);
            let (first, last) = if h.y < mp.y { (h.x, qx) } else { (qx, h.x) };
            if h.x >= point.x
                && point.x >= mp.x
                && h.x != point.x
                && point_in_triangle(DVec2::new(first, h.y), mp, DVec2::new(last, h.y), point)
            {
                let tan = (h.y - point.y).abs() / (h.x - point.x);
                let current = self.p(m);
                if self.locally_inside(p, hole)
                    && (tan < tan_min
                        || (tan == tan_min
                            && (point.x > current.x
                                || (point.x == current.x && self.sector_contains_sector(m, p)))))
                {
                    m = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(m)
    }

    fn sector_contains_sector(&self, m: usize, p: usize) -> bool {
        self.node_area(self.prev(m), m, self.prev(p)) < 0.0
            && self.node_area(self.next(p), m, self.next(m)) < 0.0
    }

    fn index_curve(&mut self, start: usize) {
        let mut p = start;
        loop {
            if self.nodes[p].z == 0 {
                self.nodes[p].z = z_order(self.p(p), self.min, self.inverse_size);
            }
            self.nodes[p].prev_z = Some(self.prev(p));
            self.nodes[p].next_z = Some(self.next(p));
            p = self.next(p);
            if p == start {
                break;
            }
        }
        if let Some(prev_z) = self.nodes[p].prev_z {
            self.nodes[prev_z].next_z = None;
        }
        self.nodes[p].prev_z = None;
        self.sort_linked(p);
    }

    /// Merge sort of the z-order list.
    fn sort_linked(&mut self, list: usize) {
        let mut list = Some(list);
        let mut in_size = 1;
        loop {
            let mut p = list;
            list = None;
            let mut tail: Option<usize> = None;
            let mut merges = 0;
            while let Some(start) = p {
                merges += 1;
                let mut q = Some(start);
                let mut p_size = 0;
                for _ in 0..in_size {
                    p_size += 1;
                    q = q.and_then(|node| self.nodes[node].next_z);
                    if q.is_none() {
                        break;
                    }
                }
                let mut q_size = in_size;
                let mut p_cursor = Some(start);
                while p_size > 0 || (q_size > 0 && q.is_some()) {
                    let take_p = match (p_cursor, q) {
                        (Some(pn), Some(qn)) => {
                            p_size != 0 && (q_size == 0 || self.nodes[pn].z <= self.nodes[qn].z)
                        }
                        (Some(_), None) => true,
                        _ => false,
                    };
                    let e = if take_p {
                        let Some(e) = p_cursor else { break };
                        p_cursor = self.nodes[e].next_z;
                        p_size -= 1;
                        e
                    } else {
                        let Some(e) = q else { break };
                        q = self.nodes[e].next_z;
                        q_size -= 1;
                        e
                    };
                    match tail {
                        Some(tail) => self.nodes[tail].next_z = Some(e),
                        None => list = Some(e),
                    }
                    self.nodes[e].prev_z = tail;
                    tail = Some(e);
                }
                p = q;
            }
            if let Some(tail) = tail {
                self.nodes[tail].next_z = None;
            }
            in_size *= 2;
            if merges <= 1 {
                break;
            }
        }
    }

    fn leftmost(&self, start: usize) -> usize {
        let mut p = start;
        let mut leftmost = start;
        loop {
            let point = self.p(p);
            let best = self.p(leftmost);
            if point.x < best.x || (point.x == best.x && point.y < best.y) {
                leftmost = p;
            }
            p = self.next(p);
            if p == start {
                return leftmost;
            }
        }
    }

    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (an, ap) = (self.next(a), self.prev(a));
        let (bn, bp) = (self.next(b), self.prev(b));
        self.nodes[an].vertex != self.nodes[b].vertex
            && self.nodes[ap].vertex != self.nodes[b].vertex
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b)
                && self.locally_inside(b, a)
                && self.middle_inside(a, b)
                && (self.node_area(ap, a, bp) != 0.0 || self.node_area(a, bp, b) != 0.0))
                || (self.equals(a, b)
                    && self.node_area(ap, a, an) > 0.0
                    && self.node_area(bp, b, bn) > 0.0))
    }

    fn intersects_polygon(&self, a: usize, b: usize) -> bool {
        let (va, vb) = (self.nodes[a].vertex, self.nodes[b].vertex);
        let mut p = a;
        loop {
            let next = self.next(p);
            let (vp, vn) = (self.nodes[p].vertex, self.nodes[next].vertex);
            if vp != va
                && vn != va
                && vp != vb
                && vn != vb
                && intersects(self.p(p), self.p(next), self.p(a), self.p(b))
            {
                return true;
            }
            p = next;
            if p == a {
                return false;
            }
        }
    }

    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (ap, an) = (self.prev(a), self.next(a));
        if self.node_area(ap, a, an) < 0.0 {
            self.node_area(a, b, an) >= 0.0 && self.node_area(a, ap, b) >= 0.0
        } else {
            self.node_area(a, b, ap) < 0.0 || self.node_area(a, an, b) < 0.0
        }
    }

    fn middle_inside(&self, a: usize, b: usize) -> bool {
        let middle = (self.p(a) + self.p(b)) / 2.0;
        let mut inside = false;
        let mut p = a;
        loop {
            let point = self.p(p);
            let next = self.p(self.next(p));
            if (point.y > middle.y) != (next.y > middle.y)
                && next.y != point.y
                && middle.x
                    < (next.x - point.x) * (middle.y - point.y) / (next.y - point.y) + point.x
            {
                inside = !inside;
            }
            p = self.next(p);
            if p == a {
                return inside;
            }
        }
    }

    fn split_polygon(&mut self, a: usize, b: usize) -> usize {
        let a2 = self.nodes.len();
        let b2 = a2 + 1;
        let an = self.next(a);
        let bp = self.prev(b);
        let mut a_copy = self.nodes[a];
        let mut b_copy = self.nodes[b];
        a_copy.z = 0;
        b_copy.z = 0;
        a_copy.prev_z = None;
        a_copy.next_z = None;
        b_copy.prev_z = None;
        b_copy.next_z = None;
        a_copy.steiner = false;
        b_copy.steiner = false;
        a_copy.next = an;
        a_copy.prev = b2;
        b_copy.next = a2;
        b_copy.prev = bp;
        self.nodes.push(a_copy);
        self.nodes.push(b_copy);
        self.nodes[a].next = b;
        self.nodes[b].prev = a;
        self.nodes[an].prev = a2;
        self.nodes[bp].next = b2;
        b2
    }
}

/// Twice the signed area of `a b c`; positive when counter-clockwise.
pub(crate) fn orient(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b - a).perp_dot(c - a)
}

fn in_circle(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> (f64, f64) {
    let (ad, bd, cd) = (a - d, b - d, c - d);
    let (al, bl, cl) = (
        ad.length_squared(),
        bd.length_squared(),
        cd.length_squared(),
    );
    let determinant = al * bd.perp_dot(cd) + bl * cd.perp_dot(ad) + cl * ad.perp_dot(bd);
    let scale = al + bl + cl;
    (determinant, scale * scale)
}

/// A triangle mesh over 2D points with edge adjacency. Boundary edges (no
/// neighbour) are constraints.
pub(crate) struct TriangleMesh {
    pub points: Vec<DVec2>,
    pub triangles: Vec<[u32; 3]>,
    /// Neighbour across edge `k` (`triangles[t][k]` to `[k + 1]`).
    neighbours: Vec<[u32; 3]>,
}

impl TriangleMesh {
    /// Builds adjacency for counter-clockwise triangles; degenerate and
    /// clockwise triangles are dropped.
    pub(crate) fn new(points: Vec<DVec2>, triangles: Vec<[u32; 3]>) -> Self {
        let triangles: Vec<[u32; 3]> = triangles
            .into_iter()
            .filter(|triangle| {
                triangle
                    .iter()
                    .all(|corner| (*corner as usize) < points.len())
                    && orient(
                        points[triangle[0] as usize],
                        points[triangle[1] as usize],
                        points[triangle[2] as usize],
                    ) > 0.0
            })
            .collect();
        let mut edges = std::collections::HashMap::with_capacity(triangles.len() * 3);
        let mut duplicated = std::collections::HashSet::new();
        for (index, triangle) in triangles.iter().enumerate() {
            for k in 0..3 {
                let key = (triangle[k], triangle[(k + 1) % 3]);
                if edges.insert(key, (index as u32, k)).is_some() {
                    duplicated.insert(key);
                }
            }
        }
        let mut neighbours = vec![[NONE; 3]; triangles.len()];
        for (index, triangle) in triangles.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (triangle[k], triangle[(k + 1) % 3]);
                if duplicated.contains(&(a, b)) || duplicated.contains(&(b, a)) {
                    continue;
                }
                if let Some((other, _)) = edges.get(&(b, a)) {
                    neighbours[index][k] = *other;
                }
            }
        }
        Self {
            points,
            triangles,
            neighbours,
        }
    }

    fn point(&self, index: u32) -> DVec2 {
        self.points[index as usize]
    }

    /// Whether edge `k` of `triangle` has a neighbour (is not a boundary).
    pub(crate) fn is_interior(&self, triangle: u32, k: usize) -> bool {
        self.neighbours[triangle as usize][k] != NONE
    }

    fn edge_index(&self, triangle: u32, a: u32, b: u32) -> Option<usize> {
        let corners = self.triangles[triangle as usize];
        (0..3).find(|k| corners[*k] == a && corners[(k + 1) % 3] == b)
    }

    fn replace_neighbour(&mut self, triangle: u32, old: u32, new: u32) {
        if triangle == NONE {
            return;
        }
        for slot in &mut self.neighbours[triangle as usize] {
            if *slot == old {
                *slot = new;
                return;
            }
        }
    }

    /// Flips edge `k` of `triangle` when that makes the pair Delaunay and
    /// the quad is convex. Returns the triangles whose other edges changed.
    fn flip_if_illegal(&mut self, triangle: u32, k: usize) -> Option<[u32; 2]> {
        let t = triangle as usize;
        let n = self.neighbours[t][k];
        if n == NONE {
            return None;
        }
        let [a, b, c] = [
            self.triangles[t][k],
            self.triangles[t][(k + 1) % 3],
            self.triangles[t][(k + 2) % 3],
        ];
        let m = self.edge_index(n, b, a)?;
        let d = self.triangles[n as usize][(m + 2) % 3];
        let (pa, pb, pc, pd) = (self.point(a), self.point(b), self.point(c), self.point(d));
        let (determinant, scale) = in_circle(pa, pb, pc, pd);
        if determinant <= 1e-12 * scale {
            return None;
        }
        if orient(pc, pa, pd) <= 0.0 || orient(pd, pb, pc) <= 0.0 {
            return None;
        }
        let t_bc = self.neighbours[t][(k + 1) % 3];
        let t_ca = self.neighbours[t][(k + 2) % 3];
        let n_ad = self.neighbours[n as usize][(m + 1) % 3];
        let n_db = self.neighbours[n as usize][(m + 2) % 3];
        self.triangles[t] = [c, a, d];
        self.neighbours[t] = [t_ca, n_ad, n];
        self.triangles[n as usize] = [d, b, c];
        self.neighbours[n as usize] = [n_db, t_bc, triangle];
        self.replace_neighbour(n_ad, n, triangle);
        self.replace_neighbour(t_bc, triangle, n);
        Some([triangle, n])
    }

    /// Restores the Delaunay property from the given edges outward,
    /// recording every triangle touched.
    fn legalize(&mut self, mut stack: Vec<(u32, usize)>, touched: &mut Vec<u32>) {
        let mut budget = 64 * self.triangles.len() + 1024;
        while let Some((triangle, k)) = stack.pop() {
            if budget == 0 {
                return;
            }
            budget -= 1;
            if let Some([t, n]) = self.flip_if_illegal(triangle, k) {
                touched.extend([t, n]);
                stack.extend([(t, 0), (t, 1), (n, 0), (n, 1)]);
            }
        }
    }

    /// Splits boundary edges at the ring points the ear clipper dropped (as
    /// collinear, or in slivers too thin to keep), so every boundary point
    /// of `rings` is a mesh vertex. Points well off their edge (cut away as
    /// self-intersections) stay out.
    pub(crate) fn restore_boundary_points(&mut self, rings: &[std::ops::Range<usize>]) {
        let mut used = vec![false; self.points.len()];
        for triangle in &self.triangles {
            for corner in triangle {
                used[*corner as usize] = true;
            }
        }
        let mut boundary = std::collections::HashMap::new();
        for (index, triangle) in self.triangles.iter().enumerate() {
            for k in 0..3 {
                if self.neighbours[index][k] == NONE {
                    boundary.insert((triangle[k], triangle[(k + 1) % 3]), (index as u32, k));
                }
            }
        }
        for ring in rings {
            let kept: Vec<usize> = ring.clone().filter(|index| used[*index]).collect();
            if kept.len() < 2 || kept.len() == ring.len() {
                continue;
            }
            let length = ring.len();
            for (position, &from) in kept.iter().enumerate() {
                let to = kept[(position + 1) % kept.len()];
                let mut gap = Vec::new();
                let mut cursor = from;
                loop {
                    cursor = ring.start + (cursor - ring.start + 1) % length;
                    if cursor == to {
                        break;
                    }
                    gap.push(cursor as u32);
                }
                if gap.is_empty() {
                    continue;
                }
                let (from, to) = (from as u32, to as u32);
                let (mut start, end, entry) = if let Some(entry) = boundary.get(&(from, to)) {
                    (from, to, *entry)
                } else if let Some(entry) = boundary.get(&(to, from)) {
                    gap.reverse();
                    (to, from, *entry)
                } else {
                    continue;
                };
                let (mut triangle, mut k) = entry;
                for point in gap {
                    let (a, b, p) = (self.point(start), self.point(end), self.point(point));
                    let c = self.point(self.triangles[triangle as usize][(k + 2) % 3]);
                    let segment = b - a;
                    let along = (p - a).dot(segment) / segment.length_squared().max(1e-300);
                    let off = (p - a).perp_dot(segment).abs() / segment.length().max(1e-300);
                    // Nearly on the edge, and splitting there leaves both
                    // halves counter-clockwise.
                    if !(along > 0.0 && along < 1.0)
                        || off > 0.1 * segment.length()
                        || orient(a, p, c) <= 0.0
                        || orient(p, b, c) <= 0.0
                    {
                        continue;
                    }
                    boundary.remove(&(start, end));
                    let new = self.split_boundary_edge(triangle, k, point, &mut boundary);
                    start = point;
                    triangle = new;
                    k = 0;
                }
            }
        }
    }

    /// Splits boundary edge `k` of `triangle` at `point`; returns the
    /// triangle holding the second half (as its edge 0).
    fn split_boundary_edge(
        &mut self,
        triangle: u32,
        k: usize,
        point: u32,
        boundary: &mut std::collections::HashMap<(u32, u32), (u32, usize)>,
    ) -> u32 {
        let t = triangle as usize;
        let corners = self.triangles[t];
        let [a, b, c] = [corners[k], corners[(k + 1) % 3], corners[(k + 2) % 3]];
        let n_bc = self.neighbours[t][(k + 1) % 3];
        let n_ca = self.neighbours[t][(k + 2) % 3];
        let new = self.triangles.len() as u32;
        self.triangles[t] = [a, point, c];
        self.neighbours[t] = [NONE, new, n_ca];
        self.triangles.push([point, b, c]);
        self.neighbours.push([NONE, n_bc, triangle]);
        self.replace_neighbour(n_bc, triangle, new);
        boundary.insert((a, point), (triangle, 0));
        boundary.insert((point, b), (new, 0));
        if n_ca == NONE {
            boundary.insert((c, a), (triangle, 2));
        }
        if n_bc == NONE {
            boundary.insert((b, c), (new, 1));
        }
        new
    }

    /// Flips every interior edge until the mesh is constrained Delaunay.
    pub(crate) fn make_delaunay(&mut self) {
        let stack = (0..self.triangles.len() as u32)
            .flat_map(|triangle| (0..3).map(move |k| (triangle, k)))
            .collect();
        self.legalize(stack, &mut Vec::new());
    }

    /// Inserts `point`, which must lie strictly inside `triangle`, and
    /// restores the Delaunay property. Returns every triangle created or
    /// changed.
    pub(crate) fn insert_in_triangle(&mut self, triangle: u32, point: DVec2) -> Vec<u32> {
        let t = triangle as usize;
        let p = self.points.len() as u32;
        self.points.push(point);
        let [a, b, c] = self.triangles[t];
        let [n_ab, n_bc, n_ca] = self.neighbours[t];
        let t1 = self.triangles.len() as u32;
        let t2 = t1 + 1;
        self.triangles[t] = [a, b, p];
        self.neighbours[t] = [n_ab, t1, t2];
        self.triangles.push([b, c, p]);
        self.neighbours.push([n_bc, t2, triangle]);
        self.triangles.push([c, a, p]);
        self.neighbours.push([n_ca, triangle, t1]);
        self.replace_neighbour(n_bc, triangle, t1);
        self.replace_neighbour(n_ca, triangle, t2);
        let mut touched = vec![triangle, t1, t2];
        self.legalize(vec![(triangle, 0), (t1, 0), (t2, 0)], &mut touched);
        touched.sort_unstable();
        touched.dedup();
        touched
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use substancecraft_core::math;

    fn total_area(points: &[DVec2], triangles: &[[u32; 3]]) -> f64 {
        triangles
            .iter()
            .map(|t| {
                orient(
                    points[t[0] as usize],
                    points[t[1] as usize],
                    points[t[2] as usize],
                ) / 2.0
            })
            .sum()
    }

    #[test]
    fn ear_clips_a_square_with_a_hole_counter_clockwise() {
        let points = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(0.0, 4.0),
            DVec2::new(4.0, 4.0),
            DVec2::new(4.0, 0.0),
            DVec2::new(1.0, 1.0),
            DVec2::new(3.0, 1.0),
            DVec2::new(3.0, 3.0),
            DVec2::new(1.0, 3.0),
        ];
        let triangles = earcut(&points, &[0, 4]);
        assert_eq!(triangles.len(), 8);
        assert!((total_area(&points, &triangles) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn ear_clips_large_concave_rings_through_the_hashed_path() {
        // A 200-point star: hashing kicks in above 80 points.
        let points: Vec<DVec2> = (0..200)
            .map(|index| {
                let angle = index as f64 / 200.0 * std::f64::consts::TAU;
                let radius = if index % 2 == 0 { 2.0 } else { 1.0 };
                DVec2::new(math::cos(angle), math::sin(angle)) * radius
            })
            .collect();
        let triangles = earcut(&points, &[0]);
        assert_eq!(triangles.len(), 198);
        let polygon_area: f64 = (0..200)
            .map(|i| points[i].perp_dot(points[(i + 1) % 200]) / 2.0)
            .sum();
        assert!((total_area(&points, &triangles) - polygon_area).abs() < 1e-9);
    }

    #[test]
    fn delaunay_flips_and_insertion_keep_area_and_boundary() {
        let points = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(10.0, 1.0),
            DVec2::new(0.0, 1.0),
        ];
        let mut mesh = TriangleMesh::new(points.clone(), earcut(&points, &[0]));
        mesh.make_delaunay();
        let touched = mesh.insert_in_triangle(0, {
            let t = mesh.triangles[0];
            (mesh.points[t[0] as usize] + mesh.points[t[1] as usize] + mesh.points[t[2] as usize])
                / 3.0
        });
        assert!(touched.len() >= 3);
        assert_eq!(mesh.triangles.len(), 4);
        assert!((total_area(&mesh.points, &mesh.triangles) - 10.0).abs() < 1e-9);
        assert!(mesh.triangles.iter().all(|t| orient(
            mesh.points[t[0] as usize],
            mesh.points[t[1] as usize],
            mesh.points[t[2] as usize]
        ) > 0.0));
    }
}
