use std::f64::consts::PI;

use crate::image_ops::{
    Image, MatF32, build_pyramid, ccorr_direct, ccoeff_normalize,
    integral_image, rotation_matrix, warp_affine, mean_std_dev,
};
use crate::matcher::{MatcherParam, MatchResult, Point2d};
use crate::geometry::nms;

const MATCH_CANDIDATE_NUM: usize = 5;
const VISION_TOLERANCE: f64 = 1e-7;

// ---------------------------------------------------------------------------
// TemplData (pre-computed per-layer statistics)
// ---------------------------------------------------------------------------

struct TemplData {
    pyramid:      Vec<Image>,
    inv_area:     Vec<f64>,
    templ_mean:   Vec<f64>,
    templ_norm:   Vec<f64>,
    result_equal1: Vec<bool>,
    border_color: u8,
}

// ---------------------------------------------------------------------------
// Internal match candidate (before final result construction)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Candidate {
    pt_x: f64,
    pt_y: f64,
    score: f64,
    angle: f64,
    result_3x3: [[f32; 3]; 3],
    pos_on_border: bool,
}

impl Candidate {
    fn new(pt_x: f64, pt_y: f64, score: f64, angle: f64) -> Self {
        Self { pt_x, pt_y, score, angle, result_3x3: [[0.0; 3]; 3], pos_on_border: false }
    }
}

// ---------------------------------------------------------------------------
// Public matcher
// ---------------------------------------------------------------------------

pub struct PatternMatcher {
    param:      MatcherParam,
    templ_data: Option<TemplData>,
}

impl PatternMatcher {
    pub fn new(param: MatcherParam) -> Self {
        Self { param, templ_data: None }
    }

    pub fn set_template(&mut self, templ: &Image) -> i32 {
        self.templ_data = None;
        if templ.width == 0 || templ.height == 0 { return -1; }
        let min_area = self.param.min_area;
        self.templ_data = Some(learn_pattern(templ, min_area));
        0
    }

    pub fn match_image(&self, src: &Image) -> Vec<MatchResult> {
        let td = match &self.templ_data {
            Some(td) if !td.pyramid.is_empty() => td,
            _ => return vec![],
        };
        match_impl(src, td, &self.param)
    }
}

// ---------------------------------------------------------------------------
// Helper: ptRotatePt2f equivalent
// Positive angle_rad → same direction as OpenCV getRotationMatrix2D.
// ---------------------------------------------------------------------------

#[inline]
fn rotate_pt(x: f64, y: f64, cx: f64, cy: f64, angle_rad: f64) -> (f64, f64) {
    let (cos_a, sin_a) = (angle_rad.cos(), angle_rad.sin());
    let dx = x - cx;
    let dy = y - cy;
    (cx + cos_a * dx + sin_a * dy, cy - sin_a * dx + cos_a * dy)
}

// ---------------------------------------------------------------------------
// GetTopLayer: smallest pyramid level whose template area <= min_area
// ---------------------------------------------------------------------------

fn get_top_layer(w: usize, h: usize, min_side: usize) -> usize {
    let min_area = (min_side * min_side) as usize;
    let mut area = w * h;
    let mut layer = 0usize;
    while area > min_area {
        area /= 4;
        layer += 1;
    }
    layer
}

// ---------------------------------------------------------------------------
// LearnPattern
// ---------------------------------------------------------------------------

fn learn_pattern(templ: &Image, min_area: f64) -> TemplData {
    let min_side = min_area.sqrt() as usize;
    let top_layer = get_top_layer(templ.width, templ.height, min_side);
    let pyramid = build_pyramid(templ, top_layer);
    let n_layers = pyramid.len();

    let sum: f64 = templ.data.iter().map(|&p| p as f64).sum();
    let mean_val = sum / (templ.width * templ.height) as f64;
    let border_color: u8 = if mean_val < 128.0 { 255 } else { 0 };

    let mut inv_area      = vec![1.0f64; n_layers];
    let mut templ_mean    = vec![0.0f64; n_layers];
    let mut templ_norm    = vec![0.0f64; n_layers];
    let mut result_equal1 = vec![false;  n_layers];

    for i in 0..n_layers {
        let img = &pyramid[i];
        let n = (img.width * img.height) as f64;
        let ia = 1.0 / n;
        let (mean, std_dev) = mean_std_dev(img);
        let variance = std_dev * std_dev;
        result_equal1[i] = variance < f64::EPSILON;
        // templ_norm = std_dev * sqrt(n) — mirrors C++ templNorm calculation
        let tnorm = std_dev / ia.sqrt();
        inv_area[i]   = ia;
        templ_mean[i] = mean;
        templ_norm[i] = tnorm;
    }

    TemplData { pyramid, inv_area, templ_mean, templ_norm, result_equal1, border_color }
}

// ---------------------------------------------------------------------------
// Best bounding-box size for a rotated source image
// ---------------------------------------------------------------------------

fn best_rotation_size(src_w: usize, src_h: usize, angle_deg: f64) -> (usize, usize) {
    let cx = (src_w as f64 - 1.0) / 2.0;
    let cy = (src_h as f64 - 1.0) / 2.0;
    let corners = [
        (0.0, 0.0),
        (src_w as f64 - 1.0, 0.0),
        (src_w as f64 - 1.0, src_h as f64 - 1.0),
        (0.0, src_h as f64 - 1.0),
    ];
    let rad = angle_deg * PI / 180.0;
    let mut min_x = f64::MAX; let mut max_x = f64::MIN;
    let mut min_y = f64::MAX; let mut max_y = f64::MIN;
    for &(x, y) in &corners {
        let (rx, ry) = rotate_pt(x, y, cx, cy, rad);
        if rx < min_x { min_x = rx; }
        if rx > max_x { max_x = rx; }
        if ry < min_y { min_y = ry; }
        if ry > max_y { max_y = ry; }
    }
    (
        (max_x - min_x + 1.5) as usize,
        (max_y - min_y + 1.5) as usize,
    )
}

// ---------------------------------------------------------------------------
// GetRotatedROI: extract a rotated patch around pt_lt (with 3px padding)
// ---------------------------------------------------------------------------

fn get_rotated_roi(src: &Image, templ_w: usize, templ_h: usize,
                   pt_lt_x: f64, pt_lt_y: f64, angle_deg: f64, border: u8) -> Image {
    let cx = (src.width  as f64 - 1.0) / 2.0;
    let cy = (src.height as f64 - 1.0) / 2.0;

    // Where does ptLT land after the forward rotation?
    let rad = angle_deg * PI / 180.0;
    let (pt_lt_rot_x, pt_lt_rot_y) = rotate_pt(pt_lt_x, pt_lt_y, cx, cy, rad);

    let out_w = templ_w + 6;
    let out_h = templ_h + 6;

    // Fast path for angle ≈ 0: avoid warpAffine and just crop + pad the source.
    if angle_deg.abs() < VISION_TOLERANCE {
        let ox = (pt_lt_rot_x - 3.0).round() as i32;
        let oy = (pt_lt_rot_y - 3.0).round() as i32;
        return crop_padded(src, ox, oy, out_w, out_h, border);
    }

    let mut m = rotation_matrix(cx, cy, angle_deg);
    m[0][2] -= pt_lt_rot_x - 3.0;
    m[1][2] -= pt_lt_rot_y - 3.0;

    warp_affine(src, &m, out_w, out_h, border)
}

/// Crop a rectangular region starting at (ox, oy) from `src`, padding out-of-bounds with `border`.
fn crop_padded(src: &Image, ox: i32, oy: i32, out_w: usize, out_h: usize, border: u8) -> Image {
    let mut out = Image::new(out_w, out_h);
    let sw = src.width as i32;
    let sh = src.height as i32;
    for dy in 0..out_h as i32 {
        let sy = oy + dy;
        if sy < 0 || sy >= sh {
            let row = &mut out.data[dy as usize * out_w..(dy as usize + 1) * out_w];
            row.fill(border);
            continue;
        }
        let src_row = &src.data[sy as usize * src.width..(sy as usize + 1) * src.width];
        let dst_row = &mut out.data[dy as usize * out_w..(dy as usize + 1) * out_w];
        for dx in 0..out_w as i32 {
            let sx = ox + dx;
            dst_row[dx as usize] = if sx >= 0 && sx < sw { src_row[sx as usize] } else { border };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Match a rotated source against template at `layer` → CCOEFF_NORMED result
// ---------------------------------------------------------------------------

fn match_template_layer(src: &Image, td: &TemplData, layer: usize) -> MatF32 {
    let tw = td.pyramid[layer].width;
    let th = td.pyramid[layer].height;

    if td.result_equal1[layer] {
        let rw = src.width  - tw + 1;
        let rh = src.height - th + 1;
        return MatF32::filled(rw, rh, 1.0);
    }

    let mut result = ccorr_direct(src, &td.pyramid[layer]);
    let (sum, sqsum) = integral_image(src);
    ccoeff_normalize(
        &sum, &sqsum, &mut result,
        td.inv_area[layer], td.templ_mean[layer], td.templ_norm[layer],
        tw, th,
    );
    result
}

// ---------------------------------------------------------------------------
// "Paint black" around ptMaxLoc and find next maximum
// ---------------------------------------------------------------------------

fn get_next_max(result: &mut MatF32, peak_x: usize, peak_y: usize,
                templ_w: usize, templ_h: usize, iou_thresh: f64) -> ((usize, usize), f32) {
    let factor = 1.0 - iou_thresh;
    let sx = (peak_x as f64 - templ_w as f64 * factor).round() as i32;
    let sy = (peak_y as f64 - templ_h as f64 * factor).round() as i32;
    let bw = (2.0 * templ_w as f64 * factor).round() as i32;
    let bh = (2.0 * templ_h as f64 * factor).round() as i32;
    result.fill_rect(sx, sy, bw, bh, -1.0);
    let (_, _, mx, mx_loc) = result.min_max_loc();
    (mx_loc, mx)
}

// ---------------------------------------------------------------------------
// Sub-pixel estimation (10-parameter quadratic fit over 27 points)
// ---------------------------------------------------------------------------

fn sub_pix_estimation(candidates: &[Candidate], angle_step: f64, max_idx: usize) -> (f64, f64, f64) {
    let cx = candidates[max_idx].pt_x;
    let cy = candidates[max_idx].pt_y;
    let ct = candidates[max_idx].angle;

    let mut mat_a = [[0.0f64; 10]; 27];
    let mut mat_s = [0.0f64; 27];
    let mut row = 0usize;

    for theta in 0i32..3 {
        for y in -1i32..=1 {
            for x in -1i32..=1 {
                let dx = cx + x as f64;
                let dy = cy + y as f64;
                let dt = (ct + (theta - 1) as f64 * angle_step) * PI / 180.0;
                mat_a[row] = [dx*dx, dy*dy, dt*dt, dx*dy, dx*dt, dy*dt, dx, dy, dt, 1.0];
                // theta 0 → candidate max_idx-1, theta 1 → max_idx, theta 2 → max_idx+1
                mat_s[row] = candidates[max_idx + theta as usize - 1].result_3x3[x as usize + 1][y as usize + 1] as f64;
                row += 1;
            }
        }
    }

    let z = solve_normal_equations(&mat_a, &mat_s);
    // K1 * delta = K2,  where K1 = Hessian/2, K2 = -gradient
    let k1 = [
        [2.0*z[0],  z[3],      z[4]     ],
        [z[3],      2.0*z[1],  z[5]     ],
        [z[4],      z[5],      2.0*z[2] ],
    ];
    let k2 = [-z[6], -z[7], -z[8]];
    let delta = solve_3x3(k1, k2);
    (delta[0], delta[1], delta[2] * 180.0 / PI)
}

fn solve_normal_equations(a: &[[f64; 10]; 27], s: &[f64; 27]) -> [f64; 10] {
    let mut ata = [[0.0f64; 10]; 10];
    let mut ats = [0.0f64; 10];
    for k in 0..27 {
        for i in 0..10 {
            ats[i] += a[k][i] * s[k];
            for j in 0..10 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }
    // Gauss-Jordan with partial pivoting
    let mut aug = [[0.0f64; 11]; 10];
    for i in 0..10 {
        aug[i][..10].copy_from_slice(&ata[i]);
        aug[i][10] = ats[i];
    }
    for col in 0..10 {
        let mut max_row = col;
        for row in col+1..10 {
            if aug[row][col].abs() > aug[max_row][col].abs() { max_row = row; }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 { continue; }
        for j in col..11 { aug[col][j] /= pivot; }
        for row in 0..10 {
            if row == col { continue; }
            let f = aug[row][col];
            for j in col..11 { aug[row][j] -= f * aug[col][j]; }
        }
    }
    let mut z = [0.0f64; 10];
    for i in 0..10 { z[i] = aug[i][10]; }
    z
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
             -a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
             +a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
    if det.abs() < 1e-12 { return [0.0; 3]; }
    let d = 1.0 / det;
    let inv = [
        [ (a[1][1]*a[2][2]-a[1][2]*a[2][1])*d, -(a[0][1]*a[2][2]-a[0][2]*a[2][1])*d,  (a[0][1]*a[1][2]-a[0][2]*a[1][1])*d],
        [-(a[1][0]*a[2][2]-a[1][2]*a[2][0])*d,  (a[0][0]*a[2][2]-a[0][2]*a[2][0])*d, -(a[0][0]*a[1][2]-a[0][2]*a[1][0])*d],
        [ (a[1][0]*a[2][1]-a[1][1]*a[2][0])*d, -(a[0][0]*a[2][1]-a[0][1]*a[2][0])*d,  (a[0][0]*a[1][1]-a[0][1]*a[1][0])*d],
    ];
    [
        inv[0][0]*b[0] + inv[0][1]*b[1] + inv[0][2]*b[2],
        inv[1][0]*b[0] + inv[1][1]*b[1] + inv[1][2]*b[2],
        inv[2][0]*b[0] + inv[2][1]*b[1] + inv[2][2]*b[2],
    ]
}

// ---------------------------------------------------------------------------
// Per-candidate refinement — pure function, safe to run in parallel threads.
// Walks pyramid layers from (top_layer-1) down to 0, narrowing position and
// angle. Returns None if the candidate falls below the score threshold.
// ---------------------------------------------------------------------------

fn refine_candidate(
    tc:          &Candidate,
    src_pyr:     &[Image],
    td:          &TemplData,
    param:       &MatcherParam,
    top_layer:   usize,
    layer_score: &[f64],
    top_cx:      f64,
    top_cy:      f64,
) -> Option<Candidate> {
    let stop_layer = 0usize;

    let dra = -tc.angle * PI / 180.0;
    let (mut pt_lt_x, mut pt_lt_y) = rotate_pt(tc.pt_x, tc.pt_y, top_cx, top_cy, dra);
    let mut cur_angle = tc.angle;

    if top_layer <= stop_layer {
        let scale = if top_layer == 0 { 1.0 } else { 2.0 };
        return Some(Candidate::new(pt_lt_x * scale, pt_lt_y * scale, tc.score, cur_angle));
    }

    'layers: for layer in (stop_layer..top_layer).rev() {
        let src_layer = &src_pyr[layer];
        let src_cx = (src_layer.width  as f64 - 1.0) / 2.0;
        let src_cy = (src_layer.height as f64 - 1.0) / 2.0;

        let tw = td.pyramid[layer].width;
        let th = td.pyramid[layer].height;
        let templ_max_side = tw.max(th) as f64;
        let angle_step = (2.0 / templ_max_side).atan() * 180.0 / PI;

        let fine_angles: Vec<f64> = if param.angle < VISION_TOLERANCE {
            vec![0.0]
        } else {
            (-2i32..=2).map(|i| cur_angle + i as f64 * angle_step).collect()
        };

        let n_angles = fine_angles.len();
        let mut fine_candidates: Vec<Candidate> = Vec::with_capacity(n_angles);
        let mut best_score = -1.0f64;
        let mut best_idx   = 0usize;

        for (j, &ang) in fine_angles.iter().enumerate() {
            let roi = get_rotated_roi(src_layer, tw, th,
                                      pt_lt_x * 2.0, pt_lt_y * 2.0, ang, 0);

            if roi.width < tw || roi.height < th {
                fine_candidates.push(Candidate::new(0.0, 0.0, -1.0, ang));
                continue;
            }

            let result = match_template_layer(&roi, td, layer);
            let (_, _, max_val, max_loc) = result.min_max_loc();
            let (mx, my) = (max_loc.0, max_loc.1);

            let on_border = mx == 0 || my == 0
                || mx == result.width - 1 || my == result.height - 1;
            let mut cand = Candidate::new(mx as f64, my as f64, max_val as f64, ang);
            cand.pos_on_border = on_border;

            if !on_border {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let rx = mx as i32 + dx;
                        let ry = my as i32 + dy;
                        if rx >= 0 && ry >= 0
                            && rx < result.width  as i32
                            && ry < result.height as i32
                        {
                            cand.result_3x3[(dx+1) as usize][(dy+1) as usize] =
                                result.get(rx as usize, ry as usize);
                        }
                    }
                }
            }
            let _ = result;

            if cand.score > best_score {
                best_score = cand.score;
                best_idx   = j;
            }
            fine_candidates.push(cand);
        }

        if best_score < layer_score[layer] { break 'layers; }

        if layer == 0 && !fine_candidates[best_idx].pos_on_border
            && best_idx > 0 && best_idx < n_angles - 1
        {
            let (dx, dy, da) = sub_pix_estimation(&fine_candidates, angle_step, best_idx);
            fine_candidates[best_idx].pt_x    = dx;
            fine_candidates[best_idx].pt_y    = dy;
            fine_candidates[best_idx].angle   = da;
        }

        let best      = &fine_candidates[best_idx];
        let new_angle = best.angle;
        let rad       = new_angle * PI / 180.0;

        let (pt_lt_rot_x, pt_lt_rot_y) =
            rotate_pt(pt_lt_x * 2.0, pt_lt_y * 2.0, src_cx, src_cy, rad);
        let combined_x = best.pt_x + (pt_lt_rot_x - 3.0);
        let combined_y = best.pt_y + (pt_lt_rot_y - 3.0);
        let (back_x, back_y) = rotate_pt(combined_x, combined_y, src_cx, src_cy, -rad);

        if layer == stop_layer {
            let scale = if stop_layer == 0 { 1.0 } else { 2.0 };
            let mut c = best.clone();
            c.pt_x  = back_x * scale;
            c.pt_y  = back_y * scale;
            c.angle = new_angle;
            return Some(c);
        }
        cur_angle = new_angle;
        pt_lt_x   = back_x;
        pt_lt_y   = back_y;
    }
    None
}

// ---------------------------------------------------------------------------
// Main match implementation (mirrors PatternMatcher::match)
// ---------------------------------------------------------------------------

fn match_impl(src: &Image, td: &TemplData, param: &MatcherParam) -> Vec<MatchResult> {
    let top_layer = td.pyramid.len() - 1;

    // Build source pyramid
    let src_pyr = build_pyramid(src, top_layer);

    let top_pyr_w  = src_pyr[top_layer].width;
    let top_pyr_h  = src_pyr[top_layer].height;
    let top_cx = (top_pyr_w as f64 - 1.0) / 2.0;
    let top_cy = (top_pyr_h as f64 - 1.0) / 2.0;

    // Angle step at top layer
    let templ_top_max_side = td.pyramid[top_layer].width.max(td.pyramid[top_layer].height) as f64;
    let angle_step_top = (2.0 / templ_top_max_side).atan() * 180.0 / PI;

    // Build angle list for top layer
    let angles_top: Vec<f64> = if param.angle < VISION_TOLERANCE {
        vec![0.0]
    } else {
        let mut v = Vec::new();
        let mut a = 0.0f64;
        while a <= param.angle + angle_step_top {
            v.push(a);
            a += angle_step_top;
        }
        let mut a = -angle_step_top;
        while a >= -param.angle - angle_step_top {
            v.push(a);
            a -= angle_step_top;
        }
        v
    };

    // Per-layer score thresholds (decay by 0.9 per level upward)
    let mut layer_score: Vec<f64> = vec![param.score_threshold; top_layer + 1];
    for l in 1..=top_layer {
        layer_score[l] = layer_score[l-1] * 0.9;
    }

    // --- Stage 1: top layer rough match ---
    let templ_top_w = td.pyramid[top_layer].width;
    let templ_top_h = td.pyramid[top_layer].height;

    let mut top_candidates: Vec<Candidate> = Vec::new();

    for &angle in &angles_top {
        let (best_w, best_h) = best_rotation_size(top_pyr_w, top_pyr_h, angle);
        let tx = (best_w as f64 - 1.0) / 2.0 - top_cx;
        let ty = (best_h as f64 - 1.0) / 2.0 - top_cy;

        let mut m = rotation_matrix(top_cx, top_cy, angle);
        m[0][2] += tx;
        m[1][2] += ty;

        let rot_src = warp_affine(&src_pyr[top_layer], &m, best_w, best_h, td.border_color);

        // Skip if result map would be empty
        if best_w < templ_top_w || best_h < templ_top_h { continue; }

        let mut result = match_template_layer(&rot_src, td, top_layer);

        let (_, _, mut max_val, mut max_loc) = result.min_max_loc();
        if (max_val as f64) < layer_score[top_layer] { continue; }

        // Shift back by the translation applied to the rotation matrix
        top_candidates.push(Candidate::new(
            max_loc.0 as f64 - tx,
            max_loc.1 as f64 - ty,
            max_val as f64,
            angle,
        ));

        let search_limit = param.max_count as usize + MATCH_CANDIDATE_NUM - 1;
        for _ in 0..search_limit {
            let (loc, val) = get_next_max(&mut result, max_loc.0, max_loc.1,
                                          templ_top_w, templ_top_h, param.iou_threshold);
            if (val as f64) < layer_score[top_layer] { break; }
            max_loc = loc;
            max_val = val;
            top_candidates.push(Candidate::new(
                max_loc.0 as f64 - tx,
                max_loc.1 as f64 - ty,
                max_val as f64,
                angle,
            ));
        }
    }

    // Sort descending by score
    top_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // --- Stage 2: refine from top layer down to layer 0 (parallel per candidate) ---
    let mut all_results: Vec<Candidate> = std::thread::scope(|s| {
        let handles: Vec<_> = top_candidates.iter()
            .map(|tc| s.spawn(|| refine_candidate(
                tc, &src_pyr, td, param, top_layer, &layer_score, top_cx, top_cy,
            )))
            .collect();
        handles.into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect()
    });

    // Filter by score
    all_results.retain(|c| c.score >= param.score_threshold);
    all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Build MatchResult list (using full-resolution template dimensions)
    let iw = td.pyramid[0].width  as f64;
    let ih = td.pyramid[0].height as f64;

    let match_results: Vec<MatchResult> = all_results.iter().take(param.max_count as usize * 2 + MATCH_CANDIDATE_NUM)
        .map(|c| {
            let ra = -c.angle * PI / 180.0;
            let (cos_a, sin_a) = (ra.cos(), ra.sin());
            let lt = Point2d { x: c.pt_x, y: c.pt_y };
            let rt = Point2d { x: lt.x + iw * cos_a, y: lt.y - iw * sin_a };
            let lb = Point2d { x: lt.x + ih * sin_a, y: lt.y + ih * cos_a };
            let rb = Point2d { x: rt.x + ih * sin_a, y: rt.y + ih * cos_a };
            let center = Point2d {
                x: (lt.x + rt.x + rb.x + lb.x) / 4.0,
                y: (lt.y + rt.y + rb.y + lb.y) / 4.0,
            };
            let mut angle = -c.angle;
            if angle < -180.0 { angle += 360.0; }
            if angle >  180.0 { angle -= 360.0; }
            MatchResult { left_top: lt, right_top: rt, left_bottom: lb, right_bottom: rb,
                          center, angle, score: c.score }
        })
        .collect();

    // NMS
    nms(match_results, param.iou_threshold, param.max_count as usize)
}
