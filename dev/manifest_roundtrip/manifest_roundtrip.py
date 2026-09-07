"""Type round-trip audit: PyIceberg write_manifest -> pyiceberg_core.manifest.read_manifest_entries.

pip install 'pyiceberg==0.12.0' 'pyiceberg-core==0.10.1' pyarrow fastavro
python manifest_roundtrip.py
"""
from __future__ import annotations

import datetime as dt
import io as _io
import os
import struct
import sys
import tempfile
import traceback
import uuid
from decimal import Decimal

import fastavro
import pyiceberg
import pyiceberg_core
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    DataFile, DataFileContent, FileFormat, ManifestEntry, ManifestEntryStatus, write_manifest,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import (
    BinaryType, BooleanType, DateType, DecimalType, DoubleType, FixedType, FloatType, IntegerType, LongType,
    NestedField, StringType, TimestampNanoType, TimestampType, TimestamptzNanoType, TimestamptzType, TimeType,
    UUIDType,
)
from pyiceberg_core import manifest as rm

SNAPSHOT_ID = 8744736658442914487
IO = PyArrowFileIO()
TMP = tempfile.mkdtemp(prefix="manifest_roundtrip_")

# (label, iceberg type, python partition value as PyIceberg's writer expects it, expected read-back value)
# PyIceberg stores date/time/timestamp partition values as int (days / micros) in the Record.
CASES = [
    ("boolean",        BooleanType(),          True,                       True),
    ("int",            IntegerType(),          42,                         42),
    ("long",           LongType(),             1 << 40,                    1 << 40),
    ("float",          FloatType(),            1.5,                        1.5),
    ("double",         DoubleType(),           2.25,                       2.25),
    ("date",           DateType(),             19000,                      19000),
    ("time",           TimeType(),             12_345_678_901,             12_345_678_901),
    ("timestamp",      TimestampType(),        1_600_000_000_000_000,      1_600_000_000_000_000),
    ("timestamptz",    TimestamptzType(),      1_600_000_000_000_000,      1_600_000_000_000_000),
    ("string",         StringType(),           "hello",                    "hello"),
    ("uuid",           UUIDType(),             uuid.UUID("12345678-1234-5678-1234-567812345678"), uuid.UUID("12345678-1234-5678-1234-567812345678")),
    ("fixed[16]",      FixedType(16),          b"0123456789abcdef",        b"0123456789abcdef"),
    ("binary",         BinaryType(),           b"\x00\x01\x02",            b"\x00\x01\x02"),
    ("decimal(5,2)",   DecimalType(5, 2),      Decimal("123.45"),          Decimal("123.45")),
    ("decimal(9,2)",   DecimalType(9, 2),      Decimal("1234567.89"),      Decimal("1234567.89")),
    ("decimal(18,6)",  DecimalType(18, 6),     Decimal("123456789012.345678"), Decimal("123456789012.345678")),
    ("decimal(38,10)", DecimalType(38, 10),    Decimal("1234567890123456789012345678.1234567890"), Decimal("1234567890123456789012345678.1234567890")),
    ("timestamp_ns",   TimestampNanoType(),    1_600_000_000_000_000_000,  1_600_000_000_000_000_000),
    ("timestamptz_ns", TimestamptzNanoType(),  1_600_000_000_000_000_000,  1_600_000_000_000_000_000),
]


def write_one(label: str, typ, value, lower_bounds=None) -> tuple[str, bytes]:
    schema = Schema(NestedField(1, "col", typ, required=False), schema_id=0)
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="col"), spec_id=0)
    path = os.path.join(TMP, f"{label.replace('/', '_').replace('[', '_').replace(']', '')}.avro")
    df = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=f"s3://bucket/{label}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(value),
        record_count=1,
        file_size_in_bytes=1,
        column_sizes=None, value_counts=None, null_value_counts=None, nan_value_counts=None,
        lower_bounds=lower_bounds, upper_bounds=None, key_metadata=None, split_offsets=None, equality_ids=None,
        sort_order_id=None,
    )
    entry = ManifestEntry.from_args(status=ManifestEntryStatus.ADDED, snapshot_id=SNAPSHOT_ID,
                                    sequence_number=None, file_sequence_number=None, data_file=df)
    with write_manifest(2, spec, schema, IO.new_output(path), SNAPSHOT_ID, "deflate") as w:
        w.add_entry(entry)
    with open(path, "rb") as f:
        return path, f.read()


def fastavro_control(path: str):
    with open(path, "rb") as f:
        r = fastavro.reader(f)
        writer_schema = r.writer_schema
        recs = list(r)
    return writer_schema, recs


def main() -> None:
    core_version = __import__("importlib.metadata").metadata.version("pyiceberg-core")
    print(f"python {sys.version.split()[0]}  pyiceberg {pyiceberg.__version__}  pyiceberg_core {core_version}  fastavro {fastavro.__version__}")
    print()
    print("== partition value round-trip (one identity-partitioned column per type) ==")
    results = []
    panics = {}
    partition_avro_schemas = {}
    for label, typ, value, expected in CASES:
        path, bs = write_one(label, typ, value)
        ws, recs = fastavro_control(path)
        # writer-side Avro schema for the partition field
        df_fields = next(f for f in ws["fields"] if f["name"] == "data_file")["type"]["fields"]
        part = next(f for f in df_fields if f["name"] == "partition")["type"]["fields"][0]["type"]
        partition_avro_schemas[label] = part
        on_disk = recs[0]["data_file"]["partition"]["col"]
        try:
            m = rm.read_manifest_entries(bs)
            d = m.entries()[0].data_file
            lits = d.partition
            got = lits[0].value() if lits[0] is not None else None
            status = "OK  " if got == expected and type(got) == type(expected) else "DIFF"
            results.append((status, label, expected, got, type(got).__name__))
        except BaseException as e:  # noqa: BLE001  -- Rust panics are PanicException(BaseException)
            results.append(("PANIC", label, expected, f"{type(e).__module__}.{type(e).__name__}", ""))
            panics[label] = str(e).split("\n\nBacktrace")[0]
            if label == "uuid":
                bs_uuid = bs
    bs_uuid = bs_uuid  # noqa: F821  (set in the loop above)
    for status, label, expected, got, tname in results:
        print(f"{status:5} {label:15} wrote {str(expected):45} read {str(got):45} {tname}")
    print()
    print("== exact panic strings ==")
    for k, v in panics.items():
        print(f"{k:15} {v}")
    print()
    print("== writer-side Avro schema for the partition field (from the Avro header, via fastavro) ==")
    for k in ("uuid", "decimal(9,2)", "timestamp_ns", "timestamptz_ns", "fixed[16]", "time", "timestamptz"):
        print(f"{k:15} {partition_avro_schemas[k]}")
    print()

    # ---------------- lower_bounds byte preservation
    print("== lower_bounds byte preservation ==")
    schema = Schema(
        NestedField(1, "dec", DecimalType(9, 2), required=False),
        NestedField(2, "i", IntegerType(), required=False),
        NestedField(3, "s", StringType(), required=False),
        schema_id=0,
    )
    spec = PartitionSpec(spec_id=0)
    path = os.path.join(TMP, "bounds.avro")
    lower = {1: b"\x00\x00\x00\x01", 2: b"\x01\x00\x00\x00", 3: b"abc"}
    df = DataFile.from_args(
        content=DataFileContent.DATA, file_path="s3://bucket/bounds.parquet", file_format=FileFormat.PARQUET,
        partition=Record(), record_count=1, file_size_in_bytes=1,
        column_sizes=None, value_counts=None, null_value_counts=None, nan_value_counts=None,
        lower_bounds=lower, upper_bounds=None, key_metadata=None, split_offsets=None, equality_ids=None,
        sort_order_id=None,
    )
    entry = ManifestEntry.from_args(status=ManifestEntryStatus.ADDED, snapshot_id=SNAPSHOT_ID,
                                    sequence_number=None, file_sequence_number=None, data_file=df)
    with write_manifest(2, spec, schema, IO.new_output(path), SNAPSHOT_ID, "deflate") as w:
        w.add_entry(entry)
    bs = open(path, "rb").read()
    _, recs = fastavro_control(path)
    disk = {kv["key"]: kv["value"] for kv in recs[0]["data_file"]["lower_bounds"]}
    d = rm.read_manifest_entries(bs).entries()[0].data_file
    read = d.lower_bounds
    names = {1: "decimal(9,2)", 2: "int", 3: "string"}
    for fid in (1, 2, 3):
        st = "OK" if read[fid] == lower[fid] else "ALTERED"
        print(f"field {fid}  {names[fid]:13} wrote {lower[fid]!r:22} disk(fastavro) {disk[fid]!r:22} read {read[fid]!r:22} {st}")
    print()

    # ---------------- exposed surface / v2 inheritance fields
    print("== exposed surface (pyiceberg_core 0.10.1) ==")
    m = rm.read_manifest_entries(bs)
    e = m.entries()[0]
    pub = lambda o: [a for a in dir(o) if not a.startswith("_")]
    print("Manifest:        ", pub(m))
    print("ManifestEntry:   ", pub(e))
    print("DataFile:        ", pub(e.data_file))
    print("sequence_number / file_sequence_number on v2 entry:", e.sequence_number, e.file_sequence_number)
    path_u, bs_u = write_one("uuid_lit", IntegerType(), 1)
    lit = rm.read_manifest_entries(bs_u).entries()[0].data_file.partition[0]
    print("PrimitiveLiteral:", pub(lit))
    try:
        rm.read_manifest_entries(bs_uuid)
    except BaseException as e:  # noqa: BLE001
        print("Panic exception type:", f"{type(e).__module__}.{type(e).__name__}", "MRO:", [c.__name__ for c in type(e).__mro__])
        print("caught by `except Exception`:", isinstance(e, Exception))
    try:
        rm.read_manifest_entries(b"not an avro file")
    except BaseException as e:  # noqa: BLE001
        print("corrupt input exception type:", f"{type(e).__module__}.{type(e).__name__}", "-", str(e).splitlines()[0])

    # ---------------- unpartitioned spec -> partitions on entry side and manifest-list side
    print()
    print("== unpartitioned spec: partitions ==")
    print("entry side d.partition:", list(rm.read_manifest_entries(bs).entries()[0].data_file.partition))
    from pyiceberg.manifest import ManifestContent, ManifestFile, write_manifest_list
    mf = ManifestFile.from_args(
        manifest_path=path, manifest_length=len(bs), partition_spec_id=0, content=ManifestContent.DATA,
        sequence_number=1, min_sequence_number=1, added_snapshot_id=SNAPSHOT_ID, added_files_count=1,
        existing_files_count=0, deleted_files_count=0, added_rows_count=1, existing_rows_count=0,
        deleted_rows_count=0, partitions=[], key_metadata=None,
    )
    lpath = os.path.join(TMP, "list.avro")
    with write_manifest_list(2, IO.new_output(lpath), SNAPSHOT_ID, None, 1, "deflate") as w:
        w.add_manifests([mf])
    ml = rm.read_manifest_list(open(lpath, "rb").read()).entries()[0]
    print("list side ManifestFile.partitions:", ml.partitions)
    print("ManifestFile surface:", pub(ml))
    _, lrecs = fastavro_control(lpath)
    print("on-disk manifest-list partitions field:", lrecs[0]["partitions"])
    print()
    print(f"files in {TMP}")


if __name__ == "__main__":
    main()
