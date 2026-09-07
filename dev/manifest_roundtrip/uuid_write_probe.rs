//! Probe: can iceberg-rust `main` write a manifest with a uuid partition value, and what lands on disk?
use std::sync::Arc;

use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Literal, Manifest, ManifestWriterBuilder, NestedField, PartitionSpec, PrimitiveType, Schema,
    Struct, Transform, Type,
};

const OUT: &str = "/tmp/claude-0/-home-user-iceberg-python/303389d4-f700-5bf2-8762-3863be27bc75/scratchpad/probe_files/rust_uuid.avro";

#[tokio::test]
async fn uuid_write_probe() {
    let schema = Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "u", Type::Primitive(PrimitiveType::Uuid)).into(),
            ])
            .build()
            .unwrap(),
    );
    let spec = PartitionSpec::builder(schema.clone())
        .with_spec_id(0)
        .add_partition_field("u", "u", Transform::Identity)
        .unwrap()
        .build()
        .unwrap();
    let df = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("s3://b/f.parquet".to_string())
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::from_iter([Some(Literal::uuid(uuid::Uuid::from_u128(
            0x12345678_1234_5678_1234_567812345678,
        )))]))
        .record_count(1)
        .file_size_in_bytes(1)
        .partition_spec_id(0)
        .build()
        .unwrap();
    let _ = std::fs::remove_file(OUT);
    let io = FileIO::new_with_fs();
    let out = io.new_output(OUT).unwrap();
    let mut w = ManifestWriterBuilder::new(out, Some(1), schema, spec).build_v2_data();
    w.add_file(df, 1).unwrap();
    match w.write_manifest_file().await {
        Ok(_) => {
            let bs = std::fs::read(OUT).unwrap();
            println!("PROBE write OK, {} bytes", bs.len());
            match Manifest::parse_avro(&bs) {
                Ok(m) => println!("PROBE read-back OK partition={:?}", m.entries()[0].data_file().partition()),
                Err(e) => println!("PROBE read-back ERR {e}"),
            }
        }
        Err(e) => println!("PROBE write ERR {e}"),
    }
}
