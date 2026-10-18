//! Format round-trips, parser edge cases and malformed-input handling.

use mm_core::{primitives, DVec3, Mesh};
use mm_io::*;
use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::PathBuf;

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mm-io-tests");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Closed test mesh with non-trivial coordinates.
fn test_mesh() -> Mesh {
    let mut m = primitives::icosphere(12.5, 3);
    m.transform(|p| p + DVec3::new(1.5, -2.25, 100.0));
    m
}

/// Compare triangle by triangle (all formats preserve triangle and corner order).
fn assert_same_triangles(orig: &Mesh, got: &Mesh, tol: f64) {
    assert_eq!(orig.triangle_count(), got.triangle_count());
    let a: Vec<u32> = orig.triangle_ids().collect();
    let b: Vec<u32> = got.triangle_ids().collect();
    for (&ta, &tb) in a.iter().zip(&b) {
        let pa = orig.tri_positions(ta);
        let pb = got.tri_positions(tb);
        for k in 0..3 {
            assert!(pa[k].distance(pb[k]) <= tol, "triangle {ta} corner {k}: {:?} vs {:?}", pa[k], pb[k]);
        }
    }
}

fn check_import(orig: &Mesh, got: &ImportedMesh, tol: f64) {
    got.mesh.check_validity().unwrap();
    assert_eq!(got.mesh.vertex_count(), orig.vertex_count(), "{:?}", got.format);
    assert_eq!(got.mesh.triangle_count(), orig.triangle_count());
    assert_eq!(got.mesh.is_closed(), orig.is_closed());
    assert_eq!(got.report.nonmanifold_split, 0);
    assert_same_triangles(orig, &got.mesh, tol);
}

#[test]
fn roundtrip_every_format() {
    let m = test_mesh();
    let cases: Vec<(&str, WriteOptions, f64)> = vec![
        ("bin.stl", WriteOptions::default(), 1e-4),
        ("ascii.stl", WriteOptions { stl_ascii: true, ..Default::default() }, 1e-4),
        ("ascii_f64.stl", WriteOptions { stl_ascii: true, double_precision: true, ..Default::default() }, 1e-12),
        ("mesh.obj", WriteOptions::default(), 1e-4),
        ("mesh_f64.obj", WriteOptions { double_precision: true, ..Default::default() }, 1e-12),
        ("mesh.ply", WriteOptions::default(), 1e-4),
        ("mesh_f64.ply", WriteOptions { double_precision: true, ..Default::default() }, 1e-12),
        ("mesh.off", WriteOptions::default(), 1e-4),
        ("mesh.3mf", WriteOptions::default(), 1e-4),
        ("mesh_f64.3mf", WriteOptions { double_precision: true, ..Default::default() }, 1e-12),
    ];
    for (name, opts, tol) in cases {
        let path = tmp_dir().join(format!("roundtrip_{name}"));
        write_mesh(&path, &m, &opts).unwrap();
        let got = read_mesh(&path).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(got.name, format!("roundtrip_{}", name.rsplit_once('.').unwrap().0));
        check_import(&m, &got, tol);
        assert!(got.warnings.is_empty(), "{name}: {:?}", got.warnings);
        // In-memory path agrees with the file path.
        let fmt = Format::from_path(&path).unwrap();
        let bytes = write_mesh_to_bytes(&m, fmt, &opts).unwrap();
        let got2 = read_mesh_from_bytes(&bytes, fmt).unwrap();
        check_import(&m, &got2, tol);
    }
}

#[test]
fn stl_cube_welds_closed() {
    let cube = primitives::cube(2.0, 3);
    let bytes = write_mesh_to_bytes(&cube, Format::Stl, &WriteOptions::default()).unwrap();
    assert_eq!(bytes.len(), 84 + 50 * cube.triangle_count());
    let got = read_mesh_from_bytes(&bytes, Format::Stl).unwrap();
    got.mesh.check_validity().unwrap();
    assert!(got.mesh.is_closed());
    assert!(got.mesh.is_consistently_oriented());
    assert_eq!(got.mesh.vertex_count(), cube.vertex_count());
    assert_eq!(got.welded_vertices, 3 * cube.triangle_count() - cube.vertex_count());
    assert!((mm_core::measure::volume(&got.mesh) - 8.0).abs() < 1e-5);

    // Without welding the STL is a triangle soup.
    let soup = read_mesh_from_bytes_with(&bytes, Format::Stl, &ReadOptions { weld: Weld::None }).unwrap();
    assert_eq!(soup.mesh.vertex_count(), 3 * cube.triangle_count());
    assert!(!soup.mesh.is_closed());
    assert_eq!(soup.welded_vertices, 0);
}

#[test]
fn stl_binary_header_starting_with_solid() {
    let cube = primitives::cube(1.0, 1);
    let mut bytes = write_mesh_to_bytes(&cube, Format::Stl, &WriteOptions::default()).unwrap();
    let hdr = b"solid this is really a binary file facet vertex endsolid";
    bytes[..hdr.len()].copy_from_slice(hdr);
    let got = read_mesh_from_bytes(&bytes, Format::Stl).unwrap();
    assert_eq!(got.mesh.triangle_count(), 12);
    assert!(got.mesh.is_closed());

    // Binary with trailing junk and a "solid" header still parses as binary.
    let mut junk = bytes.clone();
    junk.extend_from_slice(&[0u8; 7]);
    let got = read_mesh_from_bytes(&junk, Format::Stl).unwrap();
    assert_eq!(got.mesh.triangle_count(), 12);
    assert_eq!(got.warnings.len(), 1);

    // Header that parses as a complete (empty) ASCII solid, plus a size mismatch.
    let mut tricky = bytes.clone();
    let hdr = b"solid x\nendsolid x\nfacet\n";
    tricky[..80].fill(b' ');
    tricky[..hdr.len()].copy_from_slice(hdr);
    tricky.extend_from_slice(b"\n\n");
    let got = read_mesh_from_bytes(&tricky, Format::Stl).unwrap();
    assert_eq!(got.mesh.triangle_count(), 12);
    assert!(got.mesh.is_closed());

    // Header count of zero: fall back to the file size.
    let mut zero = bytes.clone();
    zero[80..84].copy_from_slice(&0u32.to_le_bytes());
    let got = read_mesh_from_bytes(&zero, Format::Stl).unwrap();
    assert_eq!(got.mesh.triangle_count(), 12);
}

#[test]
fn stl_ascii_parsing() {
    let text = "solid  part one \r\n\
        facet normal 0 0 1\r\n outer loop\r\n  vertex 0 0 0\r\n  vertex 1 0 0\r\n  vertex 0 1 0\r\n endloop\r\nendfacet\r\n\
        FACET NORMAL 0 0 -1\r\n OUTER LOOP\r\n  VERTEX 1 0 0\r\n  VERTEX -0.0 0 0\r\n  VERTEX 0 -1 0e0\r\n ENDLOOP\r\nENDFACET\r\n\
        endsolid part one\r\n\
        solid second\n\
        facet normal 0 0 1\n outer loop\n vertex 5 5 5\n vertex 6 5 5\n vertex 6 6 5\n vertex 5 6 5\n endloop\n endfacet\n\
        endsolid\n";
    let got = read_mesh_from_bytes(text.as_bytes(), Format::Stl).unwrap();
    got.mesh.check_validity().unwrap();
    // 2 triangles + a quad fanned into 2.
    assert_eq!(got.mesh.triangle_count(), 4);
    // -0.0 welds with 0.0: first solid has 4 distinct points, second 4.
    assert_eq!(got.mesh.vertex_count(), 8);
    let groups: Vec<i32> = got.mesh.triangle_ids().map(|t| got.mesh.tri_group(t)).collect();
    assert_eq!(groups, vec![0, 0, 1, 1]);
    assert!(got.warnings.is_empty(), "{:?}", got.warnings);

    // Junk after the final endsolid is ignored with a warning.
    let mut junk = text.as_bytes().to_vec();
    junk.extend_from_slice(b"\0\0\0 exporter padding \x01\x02");
    let got = read_mesh_from_bytes(&junk, Format::Stl).unwrap();
    assert_eq!(got.mesh.triangle_count(), 4);
    assert_eq!(got.warnings.len(), 1);
}

#[test]
fn obj_features() {
    let text = "# comment line\r\n\
        mtllib foo.mtl\r\n\
        o thing\r\n\
        v 0 0 0 1 0 0   \r\n\
        v 1 0 0 0 1 0\r\n\
        v 1 1 0 0 0 1\r\n\
        v 0 1 0 1 1 1 # trailing comment\r\n\
        vt 0 0\r\nvn 0 0 1\r\n\
        usemtl red\r\n\
        s 1\r\n\
        g first\r\n\
        f 1/1/1 2/1/1 3/1/1 4/1/1\r\n\
        v 0 0 1 0.5 0.5 0.5\r\n\
        g second\r\n\
        f -5//1 -1//1 -4//1\r\n\
        g first\r\n\
        f 2/1 -1/1 3/1\r\n\
        l 1 2\r\n\
        p 1\r\n";
    let got = read_mesh_from_bytes(text.as_bytes(), Format::Obj).unwrap();
    let m = &got.mesh;
    m.check_validity().unwrap();
    assert_eq!(m.vertex_count(), 5);
    assert_eq!(m.triangle_count(), 4);
    let tris: Vec<[u32; 3]> = m.triangle_ids().map(|t| m.triangle(t)).collect();
    assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 2]]);
    let groups: Vec<i32> = m.triangle_ids().map(|t| m.tri_group(t)).collect();
    assert_eq!(groups, vec![0, 0, 1, 0]);
    let colors = got.colors.as_ref().unwrap();
    assert_eq!(colors.len(), m.max_vid());
    assert_eq!(colors[0], [1.0, 0.0, 0.0]);
    assert_eq!(colors[2], [0.0, 0.0, 1.0]);
    assert_eq!(colors[4], [0.5, 0.5, 0.5]);
    // OBJ without colors has none.
    let plain = read_mesh_from_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", Format::Obj).unwrap();
    assert!(plain.colors.is_none());
    assert_eq!(plain.mesh.tri_group(0), 0);
}

#[test]
fn obj_groups_roundtrip() {
    let mut m = test_mesh();
    let ids: Vec<u32> = m.triangle_ids().collect();
    for &t in &ids {
        let c = m.tri_centroid(t);
        let g = if c.x > 1.5 { 7 } else if c.y > -2.25 { 3 } else { 0 };
        m.set_tri_group(t, g);
    }
    let opts = WriteOptions { obj_groups: true, ..Default::default() };
    let bytes = write_mesh_to_bytes(&m, Format::Obj, &opts).unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("g group_7"));
    let got = read_mesh_from_bytes(&bytes, Format::Obj).unwrap();
    check_import(&m, &got, 1e-4);
    // Group partition is preserved (ids may be renumbered).
    let mut map: HashMap<i32, i32> = HashMap::new();
    let got_ids: Vec<u32> = got.mesh.triangle_ids().collect();
    for (&a, &b) in ids.iter().zip(&got_ids) {
        let e = *map.entry(m.tri_group(a)).or_insert(got.mesh.tri_group(b));
        assert_eq!(e, got.mesh.tri_group(b));
    }
    let mut vals: Vec<i32> = map.values().copied().collect();
    vals.sort();
    vals.dedup();
    assert_eq!(vals.len(), 3);
}

/// Encode values for hand-built binary PLY files.
enum V {
    U8(u8),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

fn enc(vals: &[V], be: bool, out: &mut Vec<u8>) {
    for v in vals {
        macro_rules! put {
            ($x:expr) => {
                if be {
                    out.extend_from_slice(&$x.to_be_bytes())
                } else {
                    out.extend_from_slice(&$x.to_le_bytes())
                }
            };
        }
        match v {
            V::U8(x) => out.push(*x),
            V::I16(x) => put!(x),
            V::U32(x) => put!(x),
            V::I32(x) => put!(x),
            V::F32(x) => put!(x),
            V::F64(x) => put!(x),
        }
    }
}

fn ply_with_extras(be: bool) -> Vec<u8> {
    let fmt = if be { "binary_big_endian" } else { "binary_little_endian" };
    let mut out = format!(
        "ply\r\nformat {fmt} 1.0\r\ncomment made by hand\r\nobj_info test\r\n\
         element vertex 5\r\n\
         property float32 x\r\nproperty float y\r\nproperty double z\r\n\
         property float nx\r\nproperty list uchar short extra\r\n\
         property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\nproperty uchar alpha\r\n\
         property double confidence\r\n\
         element face 2\r\n\
         property uchar flags\r\nproperty list uchar uint vertex_indices\r\nproperty list uchar float texcoord\r\n\
         element edge 1\r\nproperty int vertex1\r\nproperty int vertex2\r\n\
         end_header\r\n"
    )
    .into_bytes();
    let verts = [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]];
    for (i, p) in verts.iter().enumerate() {
        let mut vals = vec![V::F32(p[0]), V::F32(p[1]), V::F64(p[2] as f64), V::F32(0.5)];
        // Variable-length list in the vertex element.
        vals.push(V::U8(i as u8));
        for k in 0..i {
            vals.push(V::I16(-(k as i16)));
        }
        vals.extend([V::U8(255), V::U8(0), V::U8((i * 50) as u8), V::U8(9), V::F64(0.25)]);
        enc(&vals, be, &mut out);
    }
    // A quad and a triangle.
    enc(&[V::U8(1), V::U8(4), V::U32(0), V::U32(1), V::U32(2), V::U32(3), V::U8(2), V::F32(0.0), V::F32(1.0)], be, &mut out);
    enc(&[V::U8(0), V::U8(3), V::U32(0), V::U32(4), V::U32(1), V::U8(0)], be, &mut out);
    enc(&[V::I32(0), V::I32(1)], be, &mut out);
    out
}

#[test]
fn ply_binary_with_extra_properties() {
    for be in [true, false] {
        let data = ply_with_extras(be);
        let got = read_mesh_from_bytes(&data, Format::Ply).unwrap();
        let m = &got.mesh;
        m.check_validity().unwrap();
        assert_eq!(m.vertex_count(), 5);
        assert_eq!(m.triangle_count(), 3);
        let tris: Vec<[u32; 3]> = m.triangle_ids().map(|t| m.triangle(t)).collect();
        assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3], [0, 4, 1]]);
        assert_eq!(m.vertex(4), DVec3::new(0.5, 0.5, 1.0));
        let c = got.colors.as_ref().unwrap();
        assert_eq!(&c[2][..2], &[1.0, 0.0]);
        assert!((c[2][2] - 100.0 / 255.0).abs() < 1e-6);
        // Truncating anywhere in the body must fail cleanly.
        let hdr_len = data.windows(10).position(|w| w == b"end_header").unwrap() + 12;
        for cut in hdr_len..data.len() {
            assert!(read_mesh_from_bytes(&data[..cut], Format::Ply).is_err(), "cut at {cut}");
        }
    }
}

#[test]
fn ply_ascii_variants() {
    let text = "ply\nformat ascii 1.0\nelement vertex 4\nproperty double x\nproperty double y\nproperty double z\n\
        property float red\nproperty float green\nproperty float blue\n\
        element face 1\nproperty list uint8 int32 vertex_index\nend_header\n\
        0 0 0 1 0 0\n1 0 0 0 1 0\n1 1 0 0 0 1\n0 1 0 0.5 0.5 0.5\n4 0 1 2 3\n";
    let got = read_mesh_from_bytes(text.as_bytes(), Format::Ply).unwrap();
    assert_eq!(got.mesh.triangle_count(), 2);
    assert_eq!(got.colors.as_ref().unwrap()[3], [0.5, 0.5, 0.5]);

    // Triangle strips (old Stanford scans) with a restart marker.
    let strips = "ply\nformat ascii 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\n\
        element tristrips 1\nproperty list int int vertex_indices\nend_header\n\
        0 0 0\n1 0 0\n0 1 0\n1 1 0\n2 1 0\n6 0 1 2 3 -1 4\n";
    let got = read_mesh_from_bytes(strips.as_bytes(), Format::Ply).unwrap();
    let tris: Vec<[u32; 3]> = got.mesh.triangle_ids().map(|t| got.mesh.triangle(t)).collect();
    assert_eq!(tris, vec![[0, 1, 2], [2, 1, 3]]);
    assert!(got.mesh.is_consistently_oriented());
}

#[test]
fn off_variants() {
    let off = "# leading comment\nOFF 5 3 0 # counts on the keyword line\n\
        0 0 0\n1 0 0\n\n1 1 0 # vertex comment\n0 1 0\n0.5 0.5 1\n\
        4 0 1 2 3 255 0 0\n3 0 4 1\n3 1 4 2 0.1 0.2 0.3 1.0\n";
    let got = read_mesh_from_bytes(off.as_bytes(), Format::Off).unwrap();
    got.mesh.check_validity().unwrap();
    assert_eq!(got.mesh.triangle_count(), 4);
    assert!(got.colors.is_none());

    let coff = "COFF\n3 1 0\n0 0 0 255 0 0 255\n1 0 0 0 255 0 255\n0 1 0 0 0 255 255\n3 0 1 2\n";
    let got = read_mesh_from_bytes(coff.as_bytes(), Format::Off).unwrap();
    let c = got.colors.unwrap();
    assert_eq!(c[0], [1.0, 0.0, 0.0]);
    assert_eq!(c[2], [0.0, 0.0, 1.0]);

    let noff = "NOFF\n3 1 3\n0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n3 0 1 2\n";
    assert_eq!(read_mesh_from_bytes(noff.as_bytes(), Format::Off).unwrap().mesh.triangle_count(), 1);
}

fn make_zip(parts: &[(&str, &str)]) -> Vec<u8> {
    let mut zw = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let opts = zip::write::SimpleFileOptions::default();
    for (name, body) in parts {
        zw.start_file(*name, opts).unwrap();
        zw.write_all(body.as_bytes()).unwrap();
    }
    zw.finish().unwrap().into_inner()
}

const RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/main.model" Id="r1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"#;

const TETRA: &str = r#"<mesh><vertices>
<vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/>
</vertices><triangles>
<triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3" pid="1" p1="0"/>
</triangles></mesh>"#;

#[test]
fn threemf_units_transforms_components() {
    let model = format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<metadata name="Title">t</metadata>
<resources>
<basematerials id="9"><base name="red" displaycolor="#FF0000"/></basematerials>
<object id="1" type="model">{TETRA}</object>
<object id="2" type="model"><components>
  <component objectid="1"/>
  <component objectid="1" transform="-1 0 0 0 1 0 0 0 1 10 0 0"/>
</components></object>
</resources>
<build>
<item objectid="2" transform="2 0 0 0 2 0 0 0 2 0 0 5"/>
</build>
</model>"##
    );
    let data = make_zip(&[("_rels/.rels", RELS), ("3D/main.model", &model)]);
    let got = read_mesh_from_bytes(&data, Format::ThreeMf).unwrap();
    let m = &got.mesh;
    m.check_validity().unwrap();
    assert_eq!(m.triangle_count(), 8);
    assert_eq!(m.vertex_count(), 8);
    assert!(m.is_closed());
    // Both instances stay outward-facing (the mirrored one is re-wound).
    assert!(m.is_consistently_oriented());
    assert!(mm_core::measure::volume(m) > 0.0);
    // unit cm -> mm (x10), item scale x2, translate z by 5 (in cm).
    let b = m.bounds();
    assert!((b.min - DVec3::new(0.0, 0.0, 50.0)).length() < 1e-9, "{b:?}");
    assert!((b.max - DVec3::new(200.0, 20.0, 70.0)).length() < 1e-9, "{b:?}");
    // One face group per placed mesh.
    let groups: std::collections::BTreeSet<i32> = m.triangle_ids().map(|t| m.tri_group(t)).collect();
    assert_eq!(groups.len(), 2);
}

#[test]
fn threemf_production_paths() {
    let root = r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">
<resources>
<object id="5" p:UUID="x" type="model"><components>
<component p:path="/3D/Objects/object_1.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 3"/>
</components></object>
</resources>
<build><item objectid="5"/></build>
</model>"#;
    let part = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<resources><object id="1" type="model">{TETRA}</object></resources><build/></model>"#
    );
    let data = make_zip(&[("_rels/.rels", RELS), ("3D/main.model", root), ("3D/Objects/object_1.model", &part)]);
    let got = read_mesh_from_bytes(&data, Format::ThreeMf).unwrap();
    assert_eq!(got.mesh.triangle_count(), 4);
    assert!((got.mesh.bounds().min.z - 3.0).abs() < 1e-12);

    // Root found without _rels via the conventional name.
    let plain = make_zip(&[(
        "3D/3dmodel.model",
        &format!(r#"<model unit="inch"><resources><object id="1">{TETRA}</object></resources><build><item objectid="1"/></build></model>"#),
    )]);
    let got = read_mesh_from_bytes(&plain, Format::ThreeMf).unwrap();
    assert!((got.mesh.bounds().max.x - 25.4).abs() < 1e-12);
}

#[test]
fn tolerance_weld_option() {
    // Perturb the triangle soup of a cube slightly so exact welding cannot close it.
    let cube = primitives::cube(10.0, 2);
    let (pos, tris, _) = cube.to_arrays();
    let mut text = String::from("solid jitter\n");
    for (i, t) in tris.iter().enumerate() {
        text += "facet normal 0 0 0\nouter loop\n";
        for (k, &v) in t.iter().enumerate() {
            let p = pos[v as usize] + DVec3::splat(((i * 3 + k) % 7) as f64 * 1e-5);
            text += &format!("vertex {} {} {}\n", p.x, p.y, p.z);
        }
        text += "endloop\nendfacet\n";
    }
    text += "endsolid jitter\n";
    let exact = read_mesh_from_bytes(text.as_bytes(), Format::Stl).unwrap();
    assert!(!exact.mesh.is_closed());
    let opts = ReadOptions { weld: Weld::Tolerance(1e-3) };
    let tol = read_mesh_from_bytes_with(text.as_bytes(), Format::Stl, &opts).unwrap();
    tol.mesh.check_validity().unwrap();
    assert!(tol.mesh.is_closed());
    assert_eq!(tol.mesh.vertex_count(), cube.vertex_count());

    // Exact weld of an indexed format with duplicated vertices.
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 4 6 5\n";
    let raw = read_mesh_from_bytes(obj.as_bytes(), Format::Obj).unwrap();
    assert_eq!(raw.mesh.boundary_edge_ids().count(), 6);
    let welded = read_mesh_from_bytes_with(obj.as_bytes(), Format::Obj, &ReadOptions { weld: Weld::Exact }).unwrap();
    assert_eq!(welded.mesh.vertex_count(), 4);
    assert_eq!(welded.welded_vertices, 2);
    assert_eq!(welded.mesh.boundary_edge_ids().count(), 4);
}

#[test]
fn colors_follow_nonmanifold_splits() {
    // Three triangles on one edge (0,1): the third gets private vertex copies.
    let text = "ply\nformat ascii 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\n\
        property uchar red\nproperty uchar green\nproperty uchar blue\n\
        element face 3\nproperty list uchar int vertex_indices\nend_header\n\
        0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n0 -1 0 10 10 10\n0 0 1 20 20 20\n\
        3 0 1 2\n3 1 0 3\n3 0 1 4\n";
    let got = read_mesh_from_bytes(text.as_bytes(), Format::Ply).unwrap();
    assert_eq!(got.report.nonmanifold_split, 1);
    let c = got.colors.as_ref().unwrap();
    assert_eq!(c.len(), got.mesh.max_vid());
    for v in got.mesh.vertex_ids() {
        let p = got.mesh.vertex(v);
        let expect = if p == DVec3::ZERO {
            [1.0, 0.0, 0.0]
        } else if p == DVec3::X {
            [0.0, 1.0, 0.0]
        } else {
            continue;
        };
        assert_eq!(c[v as usize], expect, "vertex {v}");
    }
}

#[test]
fn extension_dispatch_and_atomic_write() {
    let m = primitives::cube(1.0, 1);
    let path = tmp_dir().join("UPPER.STL");
    write_mesh(&path, &m, &WriteOptions::default()).unwrap();
    // Overwrite in place.
    write_mesh(&path, &m, &WriteOptions { stl_ascii: true, ..Default::default() }).unwrap();
    let got = read_mesh(&path).unwrap();
    assert_eq!(got.name, "UPPER");
    assert_eq!(got.format, Format::Stl);
    assert!(got.mesh.is_closed());
    assert!(std::fs::read(&path).unwrap().starts_with(b"solid UPPER"));

    assert!(write_mesh(&tmp_dir().join("x.xyz"), &m, &WriteOptions::default()).is_err());
    assert!(read_mesh(&tmp_dir().join("x.xyz")).is_err());
    assert!(read_mesh(&tmp_dir().join("missing.stl")).is_err());
    let bad_dir = tmp_dir().join("no_such_dir").join("a.stl");
    assert!(write_mesh(&bad_dir, &m, &WriteOptions::default()).is_err());
    assert!(write_mesh(&tmp_dir().join("empty.stl"), &Mesh::new(), &WriteOptions::default()).is_err());
    // No temp files left behind.
    let leftovers: Vec<_> = std::fs::read_dir(tmp_dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.ends_with(".tmp") && (n.starts_with(".empty.stl") || n.starts_with(".UPPER.STL"))
        })
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
    assert_eq!(Format::from_extension("3MF"), Some(Format::ThreeMf));
}

#[test]
fn malformed_inputs_are_errors() {
    let cube = primitives::cube(1.0, 1);
    let stl = write_mesh_to_bytes(&cube, Format::Stl, &WriteOptions::default()).unwrap();
    // Truncated binary STL (header count larger than the data).
    assert!(read_mesh_from_bytes(&stl[..stl.len() - 1], Format::Stl).is_err());
    assert!(read_mesh_from_bytes(&stl[..100], Format::Stl).is_err());
    assert!(read_mesh_from_bytes(&stl[..83], Format::Stl).is_err());
    // Binary STL whose header claims billions of triangles.
    let mut huge = stl.clone();
    huge[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(read_mesh_from_bytes(&huge, Format::Stl).is_err());

    for f in Format::ALL {
        assert!(read_mesh_from_bytes(b"", f).is_err(), "{f} empty");
        assert!(read_mesh_from_bytes(b"\x00\x01garbage\xff\xfe not a mesh", f).is_err(), "{f} garbage");
    }

    let cases: &[(Format, &str)] = &[
        (Format::Stl, "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"),
        (Format::Stl, "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 zero\n"),
        (Format::Stl, "solid x\nendsolid x\n"),
        (Format::Obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n"),
        (Format::Obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -4\n"),
        (Format::Obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"),
        (Format::Obj, "v 0 0\nf 1 1 1\n"),
        (Format::Obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n"),
        (Format::Obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\n"),
        (Format::Off, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 3\n"),
        (Format::Off, "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 -1\n"),
        (Format::Off, "OFF\n3 1 0\n0 0 0\n1 0 0\n"),
        (Format::Off, "OFF\n3 2 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"),
        (Format::Off, "OFF\n999999999 1 0\n0 0 0\n"),
        (Format::Off, "OFF BINARY\n"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 3\n"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1\n"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nend_header\n0 0\n1 0\n0 1\n"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"),
        (Format::Ply, "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n\x00\x00"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n200 0 1 2\n"),
        (Format::Ply, "ply\nformat utf8 1.0\nend_header\n"),
        (Format::Ply, "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n"),
    ];
    for (f, text) in cases {
        assert!(read_mesh_from_bytes(text.as_bytes(), *f).is_err(), "{f}: {text:?}");
    }

    // Degenerate-only input.
    assert!(read_mesh_from_bytes(b"v 0 0 0\nv 0 0 0\nf 1 1 2\n", Format::Obj).is_err());

    // 3MF: out-of-range index, missing object, bad unit, broken XML, not a zip.
    let bad_index = r#"<model><resources><object id="1"><mesh><vertices><vertex x="0" y="0" z="0"/></vertices>
        <triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object></resources><build><item objectid="1"/></build></model>"#;
    let missing = format!(r#"<model><resources><object id="1">{TETRA}</object></resources><build><item objectid="2"/></build></model>"#);
    let bad_unit = format!(r#"<model unit="furlong"><resources><object id="1">{TETRA}</object></resources><build><item objectid="1"/></build></model>"#);
    let broken = format!(r#"<model><resources><object id="1">{TETRA}</resources></model>"#);
    let cyclic = r#"<model><resources><object id="1"><components><component objectid="2"/></components></object>
        <object id="2"><components><component objectid="1"/></components></object></resources><build><item objectid="1"/></build></model>"#;
    for model in [bad_index, &missing, &bad_unit, &broken, cyclic] {
        let data = make_zip(&[("3D/3dmodel.model", model)]);
        assert!(read_mesh_from_bytes(&data, Format::ThreeMf).is_err(), "{model}");
    }
    assert!(read_mesh_from_bytes(&make_zip(&[("readme.txt", "hi")]), Format::ThreeMf).is_err());
}

#[test]
fn fuzz_no_panics() {
    let m = primitives::icosphere(1.0, 1);
    let mut seeds: Vec<(Format, Vec<u8>)> = Vec::new();
    for f in Format::ALL {
        seeds.push((f, write_mesh_to_bytes(&m, f, &WriteOptions::default()).unwrap()));
    }
    seeds.push((Format::Stl, write_mesh_to_bytes(&m, Format::Stl, &WriteOptions { stl_ascii: true, ..Default::default() }).unwrap()));
    seeds.push((Format::Ply, ply_with_extras(true)));
    seeds.push((Format::Ply, {
        let mut t = String::from("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n");
        t += "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        t.into_bytes()
    }));
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for (f, data) in &seeds {
        // Every truncation (sampled for long files).
        let step = (data.len() / 400).max(1);
        for cut in (0..data.len()).step_by(step) {
            let _ = read_mesh_from_bytes(&data[..cut], *f);
        }
        // Random byte mutations.
        for _ in 0..300 {
            let mut d = data.clone();
            for _ in 0..1 + next() % 8 {
                let i = (next() as usize) % d.len();
                d[i] = next() as u8;
            }
            if let Ok(r) = read_mesh_from_bytes(&d, *f) {
                r.mesh.check_validity().unwrap();
            }
        }
    }
    // Pure noise.
    for _ in 0..200 {
        let len = (next() % 2000) as usize;
        let d: Vec<u8> = (0..len).map(|_| next() as u8).collect();
        for f in Format::ALL {
            let _ = read_mesh_from_bytes(&d, f);
        }
    }
}

#[test]
fn vertex_colors_round_trip() {
    let mut m = mm_core::primitives::icosphere(1.0, 2);
    for v in m.vertex_ids().collect::<Vec<_>>() {
        let p = m.vertex(v);
        m.set_vertex_color(v, [(p.x as f32 + 1.0) / 2.0, (p.y as f32 + 1.0) / 2.0, 0.25]);
    }
    for format in [mm_io::Format::Ply, mm_io::Format::Obj] {
        let bytes = mm_io::write_mesh_to_bytes(&m, format, &Default::default()).unwrap();
        let back = mm_io::read_mesh_from_bytes(&bytes, format).unwrap().mesh;
        assert!(back.has_vertex_colors(), "{format:?}: colours lost");
        // Same vertex order after a round trip of a compact mesh.
        for (v, w) in m.vertex_ids().zip(back.vertex_ids()) {
            let (a, b) = (m.vertex_color(v).unwrap(), back.vertex_color(w).unwrap());
            assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1.0 / 255.0 + 1e-4), "{format:?}: {a:?} vs {b:?}");
        }
    }
    // A mesh without colours writes none.
    let plain = mm_core::primitives::cube(1.0, 1);
    let bytes = mm_io::write_mesh_to_bytes(&plain, mm_io::Format::Ply, &Default::default()).unwrap();
    assert!(!mm_io::read_mesh_from_bytes(&bytes, mm_io::Format::Ply).unwrap().mesh.has_vertex_colors());
}

#[test]
fn amf_and_export_only_formats() {
    let mut m = primitives::icosphere(1.0, 2);
    for t in m.triangle_ids().collect::<Vec<_>>() {
        let g = (m.tri_centroid(t).z > 0.0) as i32;
        m.set_tri_group(t, g);
    }
    m.set_vertex_color(0, [0.25, 0.5, 0.75]);
    // AMF round trip: positions, volumes as groups, colours.
    let bytes = write_mesh_to_bytes(&m, Format::Amf, &WriteOptions { double_precision: true, ..Default::default() }).unwrap();
    let back = read_mesh_from_bytes(&bytes, Format::Amf).unwrap().mesh;
    assert_eq!(back.triangle_count(), m.triangle_count());
    let groups: std::collections::BTreeSet<i32> = back.triangle_ids().map(|t| back.tri_group(t)).collect();
    assert_eq!(groups.len(), 2);
    assert!(back.has_vertex_colors());
    let vol = |x: &Mesh| mm_core::measure::volume(x);
    assert!((vol(&back) - vol(&m)).abs() < 1e-9);
    // Zipped AMF in inches.
    let xml = String::from_utf8(bytes).unwrap().replace("unit=\"millimeter\"", "unit=\"inch\"");
    let mut zipped = Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut zipped);
        z.start_file("model.amf", zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(xml.as_bytes()).unwrap();
        z.finish().unwrap();
    }
    let inch = read_mesh_from_bytes(&zipped.into_inner(), Format::Amf).unwrap().mesh;
    assert!((inch.bounds().extents().x - 25.4 * m.bounds().extents().x).abs() < 1e-6);
    // Export-only formats write and refuse to read.
    for f in [Format::Dae, Format::Wrl, Format::Smesh] {
        assert!(!f.can_read());
        let out = String::from_utf8(write_mesh_to_bytes(&m, f, &WriteOptions::default()).unwrap()).unwrap();
        assert!(!out.is_empty());
        assert!(read_mesh_from_bytes(out.as_bytes(), f).is_err());
    }
    let smesh = String::from_utf8(write_mesh_to_bytes(&m, Format::Smesh, &WriteOptions::default()).unwrap()).unwrap();
    // Facets carry the face group as boundary marker.
    assert!(smesh.lines().any(|l| l.starts_with("3 ") && l.ends_with(" 1")));
    assert_eq!(Format::from_extension("SMESH"), Some(Format::Smesh));
}
