//! Hide identifiers: mask patient names, IDs, birth dates and similar fields, and the parts of
//! patients' names wherever they appear in file and folder names.

use mm_volume::model::{Catalog, Field};

pub const MASK: &str = "•••";

#[derive(Clone, Debug, Default)]
pub struct Privacy {
    pub hide: bool,
    /// Lower-case name parts and IDs (3 or more characters) to mask in paths.
    tokens: Vec<Vec<char>>,
}

impl Privacy {
    pub fn new(hide: bool, catalog: Option<&Catalog>) -> Self {
        let mut tokens: Vec<Vec<char>> = Vec::new();
        if let Some(c) = catalog {
            for p in &c.patients {
                let words = p.key.name.split(|ch: char| !ch.is_alphanumeric()).chain(std::iter::once(p.key.id.as_str()));
                for w in words {
                    if w.chars().count() >= 3 {
                        let t: Vec<char> = w.chars().flat_map(|c| c.to_lowercase()).collect();
                        if !tokens.contains(&t) {
                            tokens.push(t);
                        }
                    }
                }
            }
        }
        // Longest first, so a full ID is masked before a shorter part of it.
        tokens.sort_by_key(|t| std::cmp::Reverse(t.len()));
        Self { hide, tokens }
    }

    /// An identifying value: masked when hiding (empty stays empty).
    pub fn id(&self, s: &str) -> String {
        if self.hide && !s.trim().is_empty() { MASK.into() } else { s.to_string() }
    }

    /// A file or folder name: patients' name parts and IDs masked when hiding.
    pub fn path(&self, s: &str) -> String {
        if !self.hide || self.tokens.is_empty() {
            return s.to_string();
        }
        let chars: Vec<char> = s.chars().collect();
        let lower: Vec<char> = chars.iter().map(|c| c.to_lowercase().next().unwrap_or(*c)).collect();
        let mut out = String::with_capacity(s.len());
        let mut i = 0;
        'outer: while i < chars.len() {
            for t in &self.tokens {
                if lower[i..].starts_with(t) {
                    out.push_str(MASK);
                    i += t.len();
                    continue 'outer;
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }

    /// A header field's value, masked when it identifies someone.
    pub fn field(&self, f: &Field) -> String {
        let identifying = f.group == "Patient" || matches!(f.name.as_str(), "Accession" | "Institution" | "Study ID" | "Patient ID" | "Patient name");
        if identifying { self.id(&f.value) } else { self.path(&f.value) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::model::{Patient, PatientKey};

    #[test]
    fn masks_names_in_paths_and_fields() {
        let mut c = Catalog::default();
        c.patients.push(Patient { key: PatientKey { id: "MRN12345".into(), name: "DOE^JANE^Q".into() }, name: "DOE JANE Q".into(), id: "MRN12345".into(), birth_date: "19800101".into(), sex: "F".into(), studies: vec![], other_volumes: vec![] });
        let p = Privacy::new(true, Some(&c));
        assert_eq!(p.path("/scans/Doe_Jane_J.MR.0002.IMA"), "/scans/•••_•••_J.MR.0002.IMA");
        assert_eq!(p.path("export_mrn12345.nii.gz"), "export_•••.nii.gz");
        assert_eq!(p.id("DOE JANE"), MASK);
        assert_eq!(p.id(""), "");
        assert_eq!(p.field(&Field::new("Patient", "Sex", "F")), MASK);
        assert_eq!(p.field(&Field::new("Series", "Description", "T1 AX")), "T1 AX");
        let shown = Privacy::new(false, Some(&c));
        assert_eq!(shown.path("Doe_Jane.IMA"), "Doe_Jane.IMA");
    }
}
