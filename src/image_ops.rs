/// Pure-Rust replacements for OpenCV image operations.
///
/// Image coordinate convention: x = column, y = row, row-major storage.

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Grayscale u8 image.
#[derive(Clone)]
pub struct Image {
    pub data:   Vec<u8>,
    pub width:  usize,
    pub height: usize,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0u8; width * height], width, height }
    }

    #[inline] pub fn get(&self, x: usize, y: usize) -> u8 { self.data[y * self.width + x] }
}

/// f32 matrix (template-match result maps).
#[derive(Clone)]
pub struct MatF32 {
    pub data:   Vec<f32>,
    pub width:  usize,
    pub height: usize,
}

impl MatF32 {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0f32; width * height], width, height }
    }

    pub fn filled(width: usize, height: usize, val: f32) -> Self {
        Self { data: vec![val; width * height], width, height }
    }

    #[inline] pub fn get(&self, x: usize, y: usize) -> f32 { self.data[y * self.width + x] }

    pub fn row_mut(&mut self, y: usize) -> &mut [f32] {
        &mut self.data[y*self.width..(y+1)*self.width]
    }

    /// Fill a rectangle with `val`, clamped to matrix bounds.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, val: f32) {
        let x0 = x.max(0) as usize;
        let y0 = y.max(0) as usize;
        let x1 = (x + w).min(self.width  as i32).max(0) as usize;
        let y1 = (y + h).min(self.height as i32).max(0) as usize;
        for ry in y0..y1 {
            self.data[ry*self.width+x0..ry*self.width+x1].fill(val);
        }
    }

    /// Returns (min_val, min_loc, max_val, max_loc).
    pub fn min_max_loc(&self) -> (f32, (usize, usize), f32, (usize, usize)) {
        let mut mn = f32::MAX; let mut mx = f32::MIN;
        let mut mn_loc = (0,0); let mut mx_loc = (0,0);
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.get(x, y);
                if v < mn { mn = v; mn_loc = (x, y); }
                if v > mx { mx = v; mx_loc = (x, y); }
            }
        }
        (mn, mn_loc, mx, mx_loc)
    }
}

/// f64 matrix (integral images).
pub struct MatF64 {
    pub data:  Vec<f64>,
    pub width: usize,
}

impl MatF64 {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0f64; width * height], width }
    }
    #[inline] pub fn get(&self, x: usize, y: usize) -> f64 { self.data[y * self.width + x] }
    #[inline] pub fn set(&mut self, x: usize, y: usize, v: f64) { self.data[y * self.width + x] = v; }
}

// ---------------------------------------------------------------------------
// Image statistics
// ---------------------------------------------------------------------------

/// Mean and standard deviation of a grayscale image.
pub fn mean_std_dev(img: &Image) -> (f64, f64) {
    let n = (img.width * img.height) as f64;
    let sum: f64 = img.data.iter().map(|&p| p as f64).sum();
    let mean = sum / n;
    let var: f64 = img.data.iter().map(|&p| { let d = p as f64 - mean; d * d }).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// Integral image (2D prefix sum, size (w+1) × (h+1))
// ---------------------------------------------------------------------------

pub fn integral_image(img: &Image) -> (MatF64, MatF64) {
    let (w, h) = (img.width, img.height);
    let (sw, sh) = (w + 1, h + 1);
    let mut sum   = MatF64::new(sw, sh);
    let mut sqsum = MatF64::new(sw, sh);

    for y in 1..sh {
        for x in 1..sw {
            let px = img.get(x-1, y-1) as f64;
            sum.set(x, y,
                px + sum.get(x-1, y) + sum.get(x, y-1) - sum.get(x-1, y-1));
            sqsum.set(x, y,
                px*px + sqsum.get(x-1, y) + sqsum.get(x, y-1) - sqsum.get(x-1, y-1));
        }
    }
    (sum, sqsum)
}

// ---------------------------------------------------------------------------
// Gaussian pyramid (mirrors OpenCV buildPyramid / pyrDown)
// ---------------------------------------------------------------------------

/// Horizontally blur+downsample one source row into i16 values × 16 (deferred divide).
/// Output length = out_w = (src_w + 1) / 2.  Max value = 255 * 16 = 4080, fits in i16.
///
/// The bulk loop uses unsafe pointer arithmetic (no clamping, 5 consecutive reads)
/// so LLVM can vectorize it as a u8 convolution.
#[inline]
fn horiz_blur_downsample_i16(src: &[u8], dst: &mut [i16]) {
    let out_w = dst.len();
    let src_w = src.len();
    let last = src_w as i32 - 1;

    // The safe bulk range: sx-2 >= 0 (ox >= 1) and sx+2 <= last (ox <= (last-2)/2)
    let bulk_start = 1usize;
    let bulk_end   = if last >= 2 { ((last - 2) / 2 + 1) as usize } else { 0 }.min(out_w);

    // Left border (with clamping)
    for ox in 0..bulk_start.min(out_w) {
        let sx = (ox * 2) as i32;
        let p = |d: i32| src[(sx + d).clamp(0, last) as usize] as i16;
        dst[ox] = p(-2) + p(-1) * 4 + p(0) * 6 + p(1) * 4 + p(2);
    }

    // Bulk (5 consecutive reads, no clamping — vectorizable)
    let src_ptr = src.as_ptr();
    for ox in bulk_start..bulk_end {
        let sx = ox * 2;
        unsafe {
            let s = src_ptr.add(sx - 2);
            dst[ox] = *s as i16 + (*s.add(1)) as i16 * 4 + (*s.add(2)) as i16 * 6
                    + (*s.add(3)) as i16 * 4 + (*s.add(4)) as i16;
        }
    }

    // Right border (with clamping)
    for ox in bulk_end..out_w {
        let sx = (ox * 2) as i32;
        let p = |d: i32| src[(sx + d).clamp(0, last) as usize] as i16;
        dst[ox] = p(-2) + p(-1) * 4 + p(0) * 6 + p(1) * 4 + p(2);
    }
}

/// Downsample by 2 with Gaussian pre-blur (OpenCV pyrDown equivalent).
///
/// H pass: u8 → i16 (integer [1,4,6,4,1]) with X-downsampling (rolling ring of 5 rows).
/// V pass: i16 [1,4,6,4,1] → u8 with Y-downsampling, dividing by 256 total.
///
/// Ring is a FLAT Vec<i16> of shape [5][out_w] (slot-major) to avoid pointer-chase
/// overhead from Vec<Vec<i16>>, making the V-pass inner loop SIMD-friendly.
pub fn pyr_down(img: &Image) -> Image {
    let (w, h) = (img.width, img.height);
    let out_w = (w + 1) / 2;
    let out_h = (h + 1) / 2;
    let hi    = h as i32;

    // Flat ring: ring[slot * out_w .. (slot+1) * out_w]
    let mut ring     = vec![0i16; 5 * out_w];
    let mut ring_tag = [-1000i32; 5];

    // Ensure source row `sy` is H-processed in the ring; return slot index.
    let ensure_row = |ring: &mut Vec<i16>, ring_tag: &mut [i32; 5], sy: i32| -> usize {
        let sy_c = sy.clamp(0, hi - 1);
        let slot = (sy_c as usize) % 5;
        if ring_tag[slot] != sy_c {
            let base = (sy_c as usize) * w;
            horiz_blur_downsample_i16(&img.data[base..base + w], &mut ring[slot * out_w..(slot + 1) * out_w]);
            ring_tag[slot] = sy_c;
        }
        slot
    };

    let mut out = Image::new(out_w, out_h);
    for oy in 0..out_h {
        let sy = (oy * 2) as i32;
        let s0 = ensure_row(&mut ring, &mut ring_tag, sy - 2);
        let s1 = ensure_row(&mut ring, &mut ring_tag, sy - 1);
        let s2 = ensure_row(&mut ring, &mut ring_tag, sy    );
        let s3 = ensure_row(&mut ring, &mut ring_tag, sy + 1);
        let s4 = ensure_row(&mut ring, &mut ring_tag, sy + 2);

        let out_row = &mut out.data[oy * out_w..(oy + 1) * out_w];
        // Unsafe pointers remove aliasing ambiguity so LLVM can auto-vectorize.
        let p0 = unsafe { ring.as_ptr().add(s0 * out_w) };
        let p1 = unsafe { ring.as_ptr().add(s1 * out_w) };
        let p2 = unsafe { ring.as_ptr().add(s2 * out_w) };
        let p3 = unsafe { ring.as_ptr().add(s3 * out_w) };
        let p4 = unsafe { ring.as_ptr().add(s4 * out_w) };
        let po = out_row.as_mut_ptr();

        unsafe {
            for ox in 0..out_w {
                let v: i32 = *p0.add(ox) as i32
                           + *p1.add(ox) as i32 * 4
                           + *p2.add(ox) as i32 * 6
                           + *p3.add(ox) as i32 * 4
                           + *p4.add(ox) as i32;
                *po.add(ox) = ((v + 128) >> 8).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// Build Gaussian pyramid with `levels` additional levels (index 0 = original).
pub fn build_pyramid(img: &Image, levels: usize) -> Vec<Image> {
    let mut pyr = Vec::with_capacity(levels + 1);
    pyr.push(img.clone());
    for _ in 0..levels {
        let next = pyr_down(pyr.last().unwrap());
        pyr.push(next);
    }
    pyr
}

// ---------------------------------------------------------------------------
// Affine warp (mirrors OpenCV warpAffine, inverse mapping + bilinear interp)
// ---------------------------------------------------------------------------

/// 2×3 rotation matrix around (cx, cy) for `angle_deg` degrees.
/// Equivalent to OpenCV's getRotationMatrix2D with scale=1.
pub fn rotation_matrix(cx: f64, cy: f64, angle_deg: f64) -> [[f64; 3]; 2] {
    let rad = angle_deg * std::f64::consts::PI / 180.0;
    let (cos_a, sin_a) = (rad.cos(), rad.sin());
    [
        [ cos_a, sin_a, (1.0 - cos_a)*cx - sin_a*cy],
        [-sin_a, cos_a,  sin_a*cx + (1.0 - cos_a)*cy],
    ]
}

/// Warp `src` with affine matrix `m` (forward mapping: dst_pt = M * src_pt).
/// Uses inverse mapping + bilinear interpolation; out-of-bounds pixels → `border`.
///
/// Uses f32 for the per-pixel coordinate computation (sufficient precision for
/// sub-pixel accuracy at image sizes < 16K×16K) and integer fixed-point bilinear
/// interpolation (Q15) to minimise float latency in the hot loop.
pub fn warp_affine(src: &Image, m: &[[f64; 3]; 2], out_w: usize, out_h: usize, border: u8) -> Image {
    // Invert 2×2 rotation + translation (keep f64 for the inversion itself).
    let (a, b, tx) = (m[0][0], m[0][1], m[0][2]);
    let (c, d, ty) = (m[1][0], m[1][1], m[1][2]);
    let det = a*d - b*c;
    if det.abs() < 1e-10 { return Image::new(out_w, out_h); }

    let (ia, ib) = ( d/det, -b/det);
    let (ic, id) = (-c/det,  a/det);
    let itx = -(ia*tx + ib*ty);
    let ity = -(ic*tx + id*ty);

    // Cast to f32 for the hot per-pixel path.
    let (ia, ib, ic, id) = (ia as f32, ib as f32, ic as f32, id as f32);
    let (itx, ity) = (itx as f32, ity as f32);
    let sw = src.width  as f32;
    let sh = src.height as f32;
    let src_stride = src.width;

    let mut out = Image::new(out_w, out_h);
    let src_ptr = src.data.as_ptr();
    let out_ptr = out.data.as_mut_ptr();

    for y in 0..out_h {
        // Row base for inverse mapping — amortise the y multiply.
        let sx_base = ib * y as f32 + itx;
        let sy_base = id * y as f32 + ity;
        let out_row = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(y * out_w), out_w) };

        for x in 0..out_w {
            let sx = ia * x as f32 + sx_base;
            let sy = ic * x as f32 + sy_base;

            if sx < 0.0 || sy < 0.0 || sx >= sw - 1.0 || sy >= sh - 1.0 {
                out_row[x] = if sx < 0.0 || sy < 0.0 || sx >= sw || sy >= sh {
                    border
                } else {
                    unsafe { *src_ptr.add(sy as usize * src_stride + sx as usize) }
                };
                continue;
            }
            // Integer bilinear: fixed-point with 8-bit fractional part (Q8).
            let x0 = sx as usize; let y0 = sy as usize;
            let fx = ((sx - x0 as f32) * 256.0 + 0.5) as u32;
            let fy = ((sy - y0 as f32) * 256.0 + 0.5) as u32;
            unsafe {
                let row0 = src_ptr.add(y0 * src_stride + x0);
                let row1 = row0.add(src_stride);
                let p00 = *row0       as u32;
                let p10 = *row0.add(1) as u32;
                let p01 = *row1       as u32;
                let p11 = *row1.add(1) as u32;
                let top = p00 * (256 - fx) + p10 * fx;
                let bot = p01 * (256 - fx) + p11 * fx;
                out_row[x] = ((top * (256 - fy) + bot * fy + (1 << 15)) >> 16) as u8;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Direct CCORR + CCOEFF normalisation (the hot path)
// ---------------------------------------------------------------------------

/// Direct cross-correlation: result[y][x] = Σ src[y+r][x+c] · templ[r][c].
/// Result size: (src.w - templ.w + 1) × (src.h - templ.h + 1).
///
/// Large result maps (rh*rw >= 1024) are split across CPU cores via
/// std::thread::scope — zero extra dependencies, cross-platform.
/// Small maps (Stage 2 fine refinement) stay single-threaded to avoid
/// thread-spawn overhead exceeding compute time.
pub fn ccorr_direct(src: &Image, templ: &Image) -> MatF32 {
    let rw = src.width  - templ.width  + 1;
    let rh = src.height - templ.height + 1;
    let tw = templ.width;
    let th = templ.height;
    let sw = src.width;
    let mut result = MatF32::new(rw, rh);

    // LLVM emits NEON udot / AVX2 vpmaddubsw for this pattern.
    // Outer-level parallelism comes from rayon (scale loop), not from here.
    let src_ptr   = src.data.as_ptr();
    let templ_ptr = templ.data.as_ptr();
    for r in 0..rh {
        let res_row = result.row_mut(r);
        for c in 0..rw {
            let mut acc: u64 = 0;
            let src_base = r * sw + c;
            unsafe {
                for tr in 0..th {
                    let sp = src_ptr.add(src_base + tr * sw);
                    let tp = templ_ptr.add(tr * tw);
                    let mut row_acc: u32 = 0;
                    for tc in 0..tw {
                        row_acc += *sp.add(tc) as u32 * *tp.add(tc) as u32;
                    }
                    acc += row_acc as u64;
                }
            }
            res_row[c] = acc as f32;
        }
    }
    result
}

/// Normalise a raw CCORR result in-place to CCOEFF_NORMED scores  (mirrors C++ CCOEFF_Denominator).
pub fn ccoeff_normalize(
    sum: &MatF64, sqsum: &MatF64,
    result: &mut MatF32,
    inv_area: f64, templ_mean: f64, templ_norm: f64,
    templ_w: usize, templ_h: usize,
) {
    let rw = result.width;
    let rh = result.height;

    for r in 0..rh {
        let res_row = result.row_mut(r);
        for c in 0..rw {
            // Summed-area table: window [c, r] → [c+tw, r+th]
            let t_sum   = sum.get(c, r) - sum.get(c+templ_w, r) - sum.get(c, r+templ_h) + sum.get(c+templ_w, r+templ_h);
            let t_sqsum = sqsum.get(c, r) - sqsum.get(c+templ_w, r) - sqsum.get(c, r+templ_h) + sqsum.get(c+templ_w, r+templ_h);

            let wnd_mean2 = t_sum * t_sum * inv_area;
            let diff2 = (t_sqsum - wnd_mean2).max(0.0);
            let t = if diff2 <= (0.5_f64).min(10.0 * f64::EPSILON * t_sqsum) {
                0.0
            } else {
                diff2.sqrt() * templ_norm
            };

            let num_raw = res_row[c] as f64 - t_sum * templ_mean;
            res_row[c] = if num_raw.abs() < t {
                (num_raw / t) as f32
            } else if num_raw.abs() < t * 1.125 {
                if num_raw > 0.0 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear resize
// ---------------------------------------------------------------------------

/// Resize `img` to (new_w × new_h) with bilinear interpolation (pixel-centre alignment).
pub fn resize_bilinear(img: &Image, new_w: usize, new_h: usize) -> Image {
    let mut out = Image::new(new_w, new_h);
    if new_w == 0 || new_h == 0 { return out; }
    let sx = img.width  as f32 / new_w as f32;
    let sy = img.height as f32 / new_h as f32;
    let sw = img.width  as i32;
    let sh = img.height as i32;
    for oy in 0..new_h {
        let src_y = (oy as f32 + 0.5) * sy - 0.5;
        let y0 = src_y.floor() as i32;
        let fy = (src_y - y0 as f32).max(0.0);
        for ox in 0..new_w {
            let src_x = (ox as f32 + 0.5) * sx - 0.5;
            let x0 = src_x.floor() as i32;
            let fx = (src_x - x0 as f32).max(0.0);
            let smp = |ix: i32, iy: i32| -> f32 {
                img.data[iy.clamp(0, sh-1) as usize * img.width
                        + ix.clamp(0, sw-1) as usize] as f32
            };
            let v = (1.0-fy) * ((1.0-fx) * smp(x0, y0)   + fx * smp(x0+1, y0))
                  +      fy  * ((1.0-fx) * smp(x0, y0+1) + fx * smp(x0+1, y0+1));
            out.data[oy * new_w + ox] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Drawing helpers (Bresenham — no extra dependency)
// ---------------------------------------------------------------------------

fn draw_pixel_rgb(buf: &mut [u8], w: usize, h: usize, x: i32, y: i32, rgb: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
        let i = (y as usize * w + x as usize) * 3;
        buf[i] = rgb[0]; buf[i+1] = rgb[1]; buf[i+2] = rgb[2];
    }
}

pub fn draw_line_rgb(buf: &mut [u8], w: usize, h: usize, mut x0: i32, mut y0: i32, mut x1: i32, mut y1: i32, rgb: [u8; 3]) {
    let steep = (y1-y0).abs() > (x1-x0).abs();
    if steep { std::mem::swap(&mut x0, &mut y0); std::mem::swap(&mut x1, &mut y1); }
    if x0 > x1 { std::mem::swap(&mut x0, &mut x1); std::mem::swap(&mut y0, &mut y1); }
    let dx = x1 - x0;
    let dy = (y1 - y0).abs();
    let ystep = if y0 < y1 { 1i32 } else { -1 };
    let mut err = dx / 2;
    let mut y = y0;
    for x in x0..=x1 {
        if steep { draw_pixel_rgb(buf, w, h, y, x, rgb); } else { draw_pixel_rgb(buf, w, h, x, y, rgb); }
        err -= dy;
        if err < 0 { y += ystep; err += dx; }
    }
}

pub fn draw_cross_rgb(buf: &mut [u8], w: usize, h: usize, cx: i32, cy: i32, r: i32, rgb: [u8; 3]) {
    draw_line_rgb(buf, w, h, cx-r, cy, cx+r, cy, rgb);
    draw_line_rgb(buf, w, h, cx, cy-r, cx, cy+r, rgb);
}
