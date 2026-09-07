//! Probe: parse PyIceberg 0.12.0-written manifests with iceberg-rust `main`.
use iceberg::spec::{FormatVersion, Manifest, ManifestList};

const DIR: &str = "/tmp/claude-0/-home-user-iceberg-python/303389d4-f700-5bf2-8762-3863be27bc75/scratchpad/probe_files";

fn read(name: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}/{name}")).unwrap()
}

#[test]
fn probe_manifests() {
    for name in ["int.avro", "uuid.avro", "timestamp_ns.avro", "timestamptz_ns.avro", "bounds.avro"] {
        match Manifest::parse_avro(&read(name)) {
            Ok(m) => {
                let df = m.entries()[0].data_file();
                let partition: Vec<String> = df
                    .partition()
                    .iter()
                    .map(|l| l.map(|l| format!("{l:?}")).unwrap_or_else(|| "null".to_string()))
                    .collect();
                let mut bounds: Vec<(i32, Vec<u8>)> = df
                    .lower_bounds()
                    .iter()
                    .map(|(k, v)| (*k, v.to_bytes().unwrap().to_vec()))
                    .collect();
                bounds.sort();
                println!("PROBE {name:20} OK   partition={partition:?} lower_bounds={bounds:?}");
            }
            Err(e) => println!("PROBE {name:20} ERR  {e}"),
        }
    }
}

#[test]
fn probe_manifest_lists() {
    for name in ["list.avro", "list_partitions_empty_list.avro", "list_partitions_null.avro"] {
        match ManifestList::parse_with_version(&read(name), FormatVersion::V2) {
            Ok(ml) => {
                let mf = &ml.entries()[0];
                println!(
                    "PROBE {name:32} OK   partitions={:?}",
                    mf.partitions.as_ref().map(|p| p.len())
                );
            }
            Err(e) => println!("PROBE {name:32} ERR  {e}"),
        }
    }
}
