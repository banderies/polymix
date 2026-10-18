//! Scan folders and print what was found, without patient identifiers:
//! `cargo run --release -p mm-volume --example scan_summary -- <folder>…`

use mm_volume::model::Kind;
use mm_volume::scan::{scan, Progress, ScanOptions};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let roots: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let t = std::time::Instant::now();
    let c = scan(&roots, &ScanOptions::default(), None, &|p| {
        if let Progress::Reading { done, total } = p {
            if done % 1000 == 0 {
                eprintln!("reading {done}/{total}");
            }
        }
        true
    })?;
    println!("{} files in {:.2?}: {} volumes, {} patients, {} recognised, {} problems, {} DICOMDIR", c.files_seen, t.elapsed(), c.volumes.len(), c.patients.len(), c.recognised.len(), c.problems.len(), c.dicomdirs.len());
    let show = |k: usize| {
        let v = &c.volumes[k];
        let g = v.geometry.as_ref().map(|g| format!("{:?} {:.2}x{:.2}x{:.2} {}", g.dims, g.spacing[0], g.spacing[1], g.spacing[2], g.orientation())).unwrap_or_default();
        format!("{} [{} {}] {g}{}", v.name, v.format.label(), v.kind.label(), if v.warnings.is_empty() { String::new() } else { format!(" ⚠ {}", v.warnings.join("; ")) })
    };
    for (n, p) in c.patients.iter().enumerate() {
        println!("Patient {} ({} studies)", n + 1, p.studies.len());
        for s in &p.studies {
            println!("  Study {} ({} series)", s.date, s.series.len());
            for se in &s.series {
                println!("    {:?} {} — {} files, {} volumes, {} other objects{}", se.number, se.description, se.instances, se.volumes.len(), se.objects.len(), if se.objects.is_empty() { String::new() } else { format!(" ({})", se.objects.iter().map(|o| o.what.as_str()).collect::<Vec<_>>().join(", ")) });
                for &k in &se.volumes {
                    println!("      {}", show(k));
                }
            }
        }
        for &k in &p.other_volumes {
            println!("  (matched) {}", show(k));
        }
    }
    println!("Other files:");
    let mut by_kind: std::collections::BTreeMap<Kind, usize> = Default::default();
    for &k in &c.files {
        *by_kind.entry(c.volumes[k].kind).or_default() += 1;
        println!("  {}", show(k));
    }
    println!("by kind: {by_kind:?}");
    for r in &c.recognised {
        println!("recognised {}: {}", r.format, r.path.file_name().unwrap().to_string_lossy());
    }
    for p in &c.problems {
        println!("problem {}: {}", p.path.file_name().unwrap().to_string_lossy(), p.message);
    }
    println!("skipped: {} files {:?}; folders {}; not downloaded {}; no permission {}", c.skipped.file_count(), c.skipped.by_extension.iter().take(12).collect::<Vec<_>>(), c.skipped.folders.len(), c.skipped.not_downloaded.len(), c.skipped.no_permission.len());
    Ok(())
}
