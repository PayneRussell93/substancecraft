use std::sync::mpsc::sync_channel;
use substancecraft_core::{PrimitivePreset, ViewportCamera};
use substancecraft_filament::FilamentRuntime;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "snapshot.ppm".into());
    let (w, h) = (400u32, 300u32);
    let mut r = FilamentRuntime::new(w, h).unwrap();
    r.set_mesh(&PrimitivePreset::Cube.build_mesh()).unwrap();
    let n = 64u32;
    let mut tex = Vec::new();
    for y in 0..n { for x in 0..n {
        let c = if x < n / 2 && y < n / 2 { [220, 40, 40, 255] }
            else if (x / 8 + y / 8) % 2 == 0 { [230, 230, 230, 255] } else { [40, 90, 200, 255] };
        tex.extend_from_slice(&c);
    }}
    r.set_base_color_map(1, n, n, &tex).unwrap();
    let (tx, rx) = sync_channel(1);
    assert!(r.render(ViewportCamera::default(), w, h, 1.0, move |p| { let _ = tx.send(p); }).unwrap());
    let px = loop { r.poll(); if let Ok(p) = rx.try_recv() { break p; } std::thread::sleep(std::time::Duration::from_millis(5)); };
    let mut out = format!("P6 {w} {h} 255\n").into_bytes();
    for p in px.chunks_exact(4) { out.extend_from_slice(&p[..3]); }
    std::fs::write(path, out).unwrap();
}
