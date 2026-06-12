/// Rotated rectangle geometry and NMS.
///
/// Matches C++ FilterWithRotatedRect / rotated_rectangle_intersection logic.

use crate::matcher::{MatchResult, Point2d};

// ---------------------------------------------------------------------------
// RotatedRect
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct RotatedRect {
    pub cx:     f64,
    pub cy:     f64,
    pub width:  f64,
    pub height: f64,
    pub angle:  f64, // degrees
}

impl RotatedRect {
    pub fn area(&self) -> f64 { self.width * self.height }

    /// Returns the 4 corners in order: LT, RT, RB, LB (matching OpenCV boxPoints).
    pub fn corners(&self) -> [Point2d; 4] {
        let rad = self.angle * std::f64::consts::PI / 180.0;
        let (cos_a, sin_a) = (rad.cos(), rad.sin());
        let hw = self.width  / 2.0;
        let hh = self.height / 2.0;
        // OpenCV: axes are +x right, +y down. Rotation is counter-clockwise in image coords.
        let rot = |dx: f64, dy: f64| Point2d {
            x: self.cx + cos_a * dx - sin_a * dy,
            y: self.cy + sin_a * dx + cos_a * dy,
        };
        [rot(-hw, -hh), rot(hw, -hh), rot(hw, hh), rot(-hw, hh)]
    }
}

// ---------------------------------------------------------------------------
// Polygon clipping (Sutherland-Hodgman) and area
// ---------------------------------------------------------------------------

fn clip_by_edge(poly: &[Point2d], a: Point2d, b: Point2d) -> Vec<Point2d> {
    if poly.is_empty() { return vec![]; }
    let mut out = Vec::with_capacity(poly.len() + 1);
    let edge_x = b.x - a.x;
    let edge_y = b.y - a.y;
    let inside = |p: Point2d| edge_x * (p.y - a.y) - edge_y * (p.x - a.x) >= 0.0;
    let intersect = |p: Point2d, q: Point2d| -> Point2d {
        let dx = q.x - p.x; let dy = q.y - p.y;
        let t = ((a.x - p.x) * edge_y - (a.y - p.y) * edge_x) / (dx * edge_y - dy * edge_x);
        Point2d { x: p.x + t * dx, y: p.y + t * dy }
    };
    for i in 0..poly.len() {
        let cur = poly[i];
        let prev = poly[(i + poly.len() - 1) % poly.len()];
        if inside(cur) {
            if !inside(prev) { out.push(intersect(prev, cur)); }
            out.push(cur);
        } else if inside(prev) {
            out.push(intersect(prev, cur));
        }
    }
    out
}

fn polygon_area(pts: &[Point2d]) -> f64 {
    let n = pts.len();
    if n < 3 { return 0.0; }
    let mut s = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        s += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    s.abs() * 0.5
}

/// Intersection area of two rotated rectangles (Sutherland-Hodgman clipping).
fn intersection_area(a: &RotatedRect, b: &RotatedRect) -> f64 {
    let corners_b = b.corners();
    let mut poly: Vec<Point2d> = corners_b.to_vec();
    let corners_a = a.corners();
    for i in 0..4 {
        poly = clip_by_edge(&poly, corners_a[i], corners_a[(i+1) % 4]);
        if poly.is_empty() { return 0.0; }
    }
    polygon_area(&poly)
}

/// IoU for two rotated rectangles.
pub fn iou(a: &RotatedRect, b: &RotatedRect) -> f64 {
    let inter = intersection_area(a, b);
    if inter == 0.0 { return 0.0; }
    let union = a.area() + b.area() - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

/// Non-maximum suppression over MatchResults sorted by score descending.
/// Builds a RotatedRect for each result (using the bounding box of the 4 corners)
/// and suppresses any result whose IoU with a kept result exceeds `iou_thresh`.
pub fn nms(mut results: Vec<MatchResult>, iou_thresh: f64, max_count: usize) -> Vec<MatchResult> {
    // Sort descending by score.
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let rects: Vec<RotatedRect> = results.iter().map(|r| {
        let pts = [r.left_top, r.right_top, r.right_bottom, r.left_bottom];
        let dx = r.right_top.x - r.left_top.x;
        let dy = r.right_top.y - r.left_top.y;
        let w = (dx*dx + dy*dy).sqrt();
        let dx2 = r.left_bottom.x - r.left_top.x;
        let dy2 = r.left_bottom.y - r.left_top.y;
        let h = (dx2*dx2 + dy2*dy2).sqrt();
        let angle_rad = dy.atan2(dx);
        // Back-compute centre from min/max of corners to match stored center.
        let _pts = pts; // suppress unused
        RotatedRect {
            cx:     r.center.x,
            cy:     r.center.y,
            width:  w,
            height: h,
            angle:  angle_rad * 180.0 / std::f64::consts::PI,
        }
    }).collect();

    let n = results.len();
    let mut suppressed = vec![false; n];
    let mut kept: Vec<MatchResult> = Vec::new();

    for i in 0..n {
        if suppressed[i] { continue; }
        kept.push(results[i].clone());
        if kept.len() >= max_count { break; }
        for j in (i+1)..n {
            if suppressed[j] { continue; }
            if iou(&rects[i], &rects[j]) > iou_thresh {
                suppressed[j] = true;
            }
        }
    }
    kept
}
