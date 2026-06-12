mod image_ops;
mod matcher;
mod geometry;
mod pattern_matching;

use image::{GrayImage, RgbImage};
use image_ops::{Image, draw_line_rgb, draw_cross_rgb};
use matcher::MatcherParam;
use pattern_matching::PatternMatcher;

fn load_gray(path: &str) -> Option<Image> {
    let dyn_img = image::open(path).ok()?;
    let gray: GrayImage = dyn_img.into_luma8();
    let (w, h) = gray.dimensions();
    Some(Image { data: gray.into_raw(), width: w as usize, height: h as usize })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <template.png> <source.png> [score_thresh] [max_count] [angle] [min_area]", args[0]);
        std::process::exit(1);
    }

    let templ_path = &args[1];
    let src_path   = &args[2];
    let score_thresh = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.7f64);
    let max_count    = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(10i32);
    let angle        = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0f64);
    let min_area     = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(256.0f64);

    let templ = load_gray(templ_path).expect("Failed to load template image");
    let src   = load_gray(src_path).expect("Failed to load source image");

    let param = MatcherParam { max_count, score_threshold: score_thresh,
                               iou_threshold: 0.0, angle, min_area };

    let mut matcher = PatternMatcher::new(param);
    matcher.set_template(&templ);

    let results = matcher.match_image(&src);

    println!("Results ({} matches):", results.len());
    for (i, r) in results.iter().enumerate() {
        println!("  [{i}]");
        println!("    left_top:     ({:.2}, {:.2})", r.left_top.x,     r.left_top.y);
        println!("    right_top:    ({:.2}, {:.2})", r.right_top.x,    r.right_top.y);
        println!("    right_bottom: ({:.2}, {:.2})", r.right_bottom.x, r.right_bottom.y);
        println!("    left_bottom:  ({:.2}, {:.2})", r.left_bottom.x,  r.left_bottom.y);
        println!("    center:       ({:.2}, {:.2})", r.center.x,       r.center.y);
        println!("    angle:        {:.4}°",         r.angle);
        println!("    score:        {:.4}",           r.score);
    }

    // Draw results on RGB image and save
    let src_rgb_dyn = image::open(src_path).unwrap().into_rgb8();
    let (sw, sh) = src_rgb_dyn.dimensions();
    let mut rgb_buf: Vec<u8> = src_rgb_dyn.into_raw();

    let green = [0u8, 255, 0];
    let red   = [255u8, 0, 0];

    for r in &results {
        let pts = [r.left_top, r.right_top, r.right_bottom, r.left_bottom];
        for i in 0..4 {
            let a = pts[i];
            let b = pts[(i + 1) % 4];
            draw_line_rgb(&mut rgb_buf, sw as usize, sh as usize,
                a.x as i32, a.y as i32, b.x as i32, b.y as i32, green);
        }
        draw_cross_rgb(&mut rgb_buf, sw as usize, sh as usize,
            r.center.x as i32, r.center.y as i32, 8, red);
    }

    let out_img = RgbImage::from_raw(sw, sh, rgb_buf).unwrap();
    out_img.save("result_pure.png").expect("Failed to save result_pure.png");
    println!("\nSaved result_pure.png");
}
