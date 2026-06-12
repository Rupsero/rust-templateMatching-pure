/// Public API types (mirrors the C++ template_matching namespace).

#[derive(Clone, Debug)]
pub struct MatcherParam {
    pub max_count:       i32,
    pub score_threshold: f64,
    pub iou_threshold:   f64,
    pub angle:           f64,
    pub min_area:        f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Point2d { pub x: f64, pub y: f64 }

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub left_top:     Point2d,
    pub right_top:    Point2d,
    pub left_bottom:  Point2d,
    pub right_bottom: Point2d,
    pub center:       Point2d,
    pub angle:        f64,
    pub score:        f64,
}
