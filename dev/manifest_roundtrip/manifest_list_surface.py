"""Companion check: manifest-list surface, FieldSummary, multi-field spec with bucket, null partition value, delete manifest."""
import os, struct, tempfile
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (DataFile, DataFileContent, FileFormat, ManifestContent, ManifestEntry,
    ManifestEntryStatus, ManifestFile, PartitionFieldSummary, write_manifest, write_manifest_list)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform, BucketTransform
from pyiceberg.typedef import Record
from pyiceberg.types import IntegerType, NestedField, StringType
from pyiceberg_core import manifest as rm

IO = PyArrowFileIO(); TMP = tempfile.mkdtemp(); SNAP = 123
schema = Schema(NestedField(1, "i", IntegerType(), required=False), NestedField(2, "s", StringType(), required=False), schema_id=0)
spec = PartitionSpec(PartitionField(1, 1000, IdentityTransform(), "i"), PartitionField(2, 1001, BucketTransform(4), "s_bucket"), spec_id=0)
pub = lambda o: [a for a in dir(o) if not a.startswith("_")]

def mk(content, partition):
    df = DataFile.from_args(content=content, file_path="s3://b/f.parquet", file_format=FileFormat.PARQUET, partition=partition,
        record_count=1, file_size_in_bytes=1, column_sizes=None, value_counts=None, null_value_counts=None, nan_value_counts=None,
        lower_bounds=None, upper_bounds=None, key_metadata=None, split_offsets=None, equality_ids=[1] if content == DataFileContent.EQUALITY_DELETES else None, sort_order_id=None)
    return ManifestEntry.from_args(status=ManifestEntryStatus.ADDED, snapshot_id=SNAP, sequence_number=None, file_sequence_number=None, data_file=df)

# data manifest, 2-field spec (identity int + bucket[4]), second entry has a null partition value
p = os.path.join(TMP, "data.avro")
with write_manifest(2, spec, schema, IO.new_output(p), SNAP, "deflate") as w:
    w.add_entry(mk(DataFileContent.DATA, Record(7, 2)))
    w.add_entry(mk(DataFileContent.DATA, Record(None, 3)))
    mf_data = w.to_manifest_file()
es = rm.read_manifest_entries(open(p, "rb").read()).entries()
print("multi-field spec + bucket:", [[x.value() if x is not None else None for x in e.data_file.partition] for e in es], "content:", [e.data_file.content for e in es])

# delete manifest (content=1)
p2 = os.path.join(TMP, "deletes.avro")
with write_manifest(2, spec, schema, IO.new_output(p2), SNAP, "deflate") as w:
    w.add_entry(mk(DataFileContent.EQUALITY_DELETES, Record(1, 0)))
    mf_del = w.to_manifest_file()
mf_del = ManifestFile.from_args(**{**{k: getattr(mf_del, k) for k in ("manifest_path","manifest_length","partition_spec_id","sequence_number","min_sequence_number","added_snapshot_id","added_files_count","existing_files_count","deleted_files_count","added_rows_count","existing_rows_count","deleted_rows_count","partitions","key_metadata")}, "content": ManifestContent.DELETES})
es2 = rm.read_manifest_entries(open(p2, "rb").read()).entries()
print("delete manifest entry:", "content=", es2[0].data_file.content, "equality_ids=", es2[0].data_file.equality_ids, "partition=", [x.value() for x in es2[0].data_file.partition])

# manifest list with both, sequence numbers assigned
lp = os.path.join(TMP, "list.avro")
with write_manifest_list(2, IO.new_output(lp), SNAP, None, 5, "deflate") as w:
    w.add_manifests([mf_data, mf_del])
ml = rm.read_manifest_list(open(lp, "rb").read()).entries()
print("ManifestFile surface:", pub(ml[0]))
for m in ml:
    print(f"  content={m.content} spec_id={m.partition_spec_id} seq={m.sequence_number} min_seq={m.min_sequence_number} added_snap={m.added_snapshot_id} "
          f"counts=({m.added_files_count},{m.existing_files_count},{m.deleted_files_count},{m.added_rows_count},{m.existing_rows_count},{m.deleted_rows_count}) "
          f"path={m.manifest_path.rsplit('/',1)[-1]} key_metadata={m.key_metadata} len={m.manifest_length}")
    for fs in m.partitions:
        print("   FieldSummary:", pub(fs), "->", fs.contains_null, fs.contains_nan, fs.lower_bound, fs.upper_bound)
print("pyiceberg wrote partitions summaries:", [[(s.contains_null, s.contains_nan, s.lower_bound, s.upper_bound) for s in m.partitions] for m in (mf_data, mf_del)])
