import os, tempfile
from decimal import Decimal
from pyiceberg.conversions import to_bytes, from_bytes
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat, ManifestEntry, ManifestEntryStatus, write_manifest
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.typedef import Record
from pyiceberg.types import DecimalType, IntegerType, NestedField, StringType, LongType
from pyiceberg_core import manifest as rm

IO = PyArrowFileIO(); TMP = tempfile.mkdtemp()
schema = Schema(NestedField(1, "dec", DecimalType(9, 2), required=False), NestedField(2, "i", IntegerType(), required=False),
                NestedField(3, "s", StringType(), required=False), NestedField(4, "big", DecimalType(38, 10), required=False), schema_id=0)

print("== what PyIceberg's own encoder emits for decimal bounds ==")
for t, v in ((DecimalType(9, 2), Decimal("0.01")), (DecimalType(9, 2), Decimal("1.28")), (DecimalType(9, 2), Decimal("-0.01")), (DecimalType(38, 10), Decimal("1.0000000000"))):
    print(f"  to_bytes({t}, {v}) = {to_bytes(t, v)!r}")

def run(label, lower, split_offsets=None, equality_ids=None):
    p = os.path.join(TMP, label.replace(" ", "_") + ".avro")
    df = DataFile.from_args(content=DataFileContent.DATA, file_path="s3://b/f.parquet", file_format=FileFormat.PARQUET, partition=Record(),
        record_count=1, file_size_in_bytes=1, column_sizes=None, value_counts=None, null_value_counts=None, nan_value_counts=None,
        lower_bounds=lower, upper_bounds=None, key_metadata=None, split_offsets=split_offsets, equality_ids=equality_ids, sort_order_id=None)
    e = ManifestEntry.from_args(status=ManifestEntryStatus.ADDED, snapshot_id=1, sequence_number=None, file_sequence_number=None, data_file=df)
    with write_manifest(2, PartitionSpec(spec_id=0), schema, IO.new_output(p), 1, "deflate") as w:
        w.add_entry(e)
    bs = open(p, "rb").read()
    try:
        d = rm.read_manifest_entries(bs).entries()[0].data_file
        print(f"{label:45} lower_bounds={d.lower_bounds!r}  split_offsets={d.split_offsets!r} equality_ids={d.equality_ids!r}")
    except BaseException as ex:
        print(f"{label:45} PANIC {type(ex).__name__}: {str(ex).splitlines()[0][:150]}")

print("== bounds edge cases through the reader ==")
run("padded decimal (4B @ p=9), value 1", {1: b"\x00\x00\x00\x01"})
run("minimal decimal, value 1", {1: b"\x01"})
run("minimal decimal, value 128 (sign byte)", {1: b"\x00\x80"})
run("minimal decimal, value -1", {1: b"\xff"})
run("padded decimal, value -1", {1: b"\xff\xff\xff\xff"})
run("over-long decimal (5B @ p=9)", {1: b"\x00\x00\x00\x00\x01"})
run("decimal(38,10) padded 16B, value 1", {4: b"\x00" * 15 + b"\x01"})
run("int bound wrong length (1B)", {2: b"\x01"})
run("bound for field id 99 not in schema", {1: b"\x01", 99: b"\x01"})
run("None split_offsets / equality_ids", {1: b"\x01"}, None, None)
run("empty split_offsets / equality_ids", {1: b"\x01"}, [], [])
