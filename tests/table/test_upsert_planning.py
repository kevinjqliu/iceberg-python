# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for the metadata-only planning phase of an upsert."""

import threading
from pathlib import PosixPath

import numpy as np
import pyarrow as pa
import pytest

from pyiceberg.conversions import from_bytes
from pyiceberg.io.pyarrow import schema_to_pyarrow
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import FileScanTask, Table
from pyiceberg.table.snapshots import Operation
from pyiceberg.table.upsert import (
    SourceIndex,
    UpsertPlan,
    UpsertPlanner,
    _merge_ranges,
    rows_that_differ,
)
from pyiceberg.transforms import BucketTransform, IdentityTransform
from pyiceberg.types import IntegerType, NestedField, StringType, StructType
from tests.catalog.test_base import InMemoryCatalog

SCHEMA = Schema(
    NestedField(1, "id", IntegerType(), required=True),
    NestedField(2, "payload", StringType(), required=True),
    identifier_field_ids=[1],
)
ARROW_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32(), nullable=False),
        pa.field("payload", pa.string(), nullable=False),
    ]
)


@pytest.fixture
def catalog(tmp_path: PosixPath) -> InMemoryCatalog:
    catalog = InMemoryCatalog("test.in_memory.catalog", warehouse=tmp_path.absolute().as_posix())
    catalog.create_namespace("default")
    return catalog


def rows(ids: range | list[int], payload: str = "a") -> pa.Table:
    return pa.Table.from_pylist([{"id": i, "payload": payload} for i in ids], schema=ARROW_SCHEMA)


def plan_for(table: Table, df: pa.Table, join_cols: list[str]) -> UpsertPlan:
    index = SourceIndex(df, join_cols, table.schema())
    snapshot = table.metadata.current_snapshot()
    assert snapshot is not None
    return UpsertPlanner(table.metadata, table.io, index).plan(snapshot.manifests(table.io))


def id_ranges(plan: UpsertPlan) -> list[tuple[int, int]]:
    """The ``id`` range each planned task's data file covers."""
    return sorted(
        (
            from_bytes(IntegerType(), task.file.lower_bounds[1]),
            from_bytes(IntegerType(), task.file.upper_bounds[1]),
        )
        for task in plan.tasks
    )


def three_files(catalog: InMemoryCatalog, identifier: str) -> Table:
    """A table of three data files covering ids 1-10, 11-20 and 21-30."""
    table = catalog.create_table(identifier, schema=SCHEMA)
    for start in (1, 11, 21):
        table.append(rows(range(start, start + 10)))
    return table


def test_file_is_skipped_when_no_key_falls_in_its_range(catalog: InMemoryCatalog) -> None:
    """Keys 5 and 25 span the whole table, so bounds alone cannot rule anything out.

    The metrics evaluator only sees ``5 <= id <= 25``, which overlaps all three
    files. Testing the sorted key set against each file's range is what proves the
    middle file holds nothing to match.
    """
    table = three_files(catalog, "default.skip_middle_file")

    plan = plan_for(table, rows([5, 25]), ["id"])

    assert id_ranges(plan) == [(1, 10), (21, 30)]


def test_rows_that_match_nothing_need_no_task(catalog: InMemoryCatalog) -> None:
    """Keys outside every file's range are provably new: no data file is read."""
    table = three_files(catalog, "default.all_new_keys")

    plan = plan_for(table, rows([100, 101]), ["id"])

    assert plan.tasks == []
    assert plan.definitely_new["id"].to_pylist() == [100, 101]
    assert len(plan.candidate_source) == 0


def test_source_is_split_into_new_and_candidate_rows(catalog: InMemoryCatalog) -> None:
    """Only the rows a file could match stay candidates; the rest go straight to insert."""
    table = three_files(catalog, "default.split_source")

    plan = plan_for(table, rows([3, 40, 41]), ["id"])

    assert plan.definitely_new["id"].to_pylist() == [40, 41]
    assert plan.candidate_source["id"].to_pylist() == [3]


def test_each_task_carries_only_the_rows_its_file_could_match(catalog: InMemoryCatalog) -> None:
    """The source slice is the residual: narrowed by metadata, finished by the executor."""
    table = three_files(catalog, "default.slice_per_file")

    plan = plan_for(table, rows([2, 15, 27]), ["id"])

    assert {
        id_range: task.source["id"].to_pylist()
        for id_range, task in zip(id_ranges(plan), sorted(plan.tasks, key=lambda t: t.file.lower_bounds[1]), strict=True)
    } == {(1, 10): [2], (11, 20): [15], (21, 30): [27]}


def test_partition_values_prune_manifests(catalog: InMemoryCatalog) -> None:
    """Source keys are pushed through the spec's transforms to name exact partitions.

    A range predicate does not project through ``bucket`` at all, so without this
    every partition would be a candidate.
    """
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=BucketTransform(8), name="id_bucket"))
    table = catalog.create_table("default.bucketed", schema=SCHEMA, partition_spec=spec)
    table.append(rows(range(1, 100)))

    plan = plan_for(table, rows([42]), ["id"])

    assert len(plan.tasks) == 1
    partition_of = {task.file.partition for task in plan.tasks}
    assert len(partition_of) == 1


def test_identity_partition_values_prune_manifests(catalog: InMemoryCatalog) -> None:
    schema = Schema(
        NestedField(1, "region", StringType(), required=True),
        NestedField(2, "id", IntegerType(), required=True),
        identifier_field_ids=[2],
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="region"))
    table = catalog.create_table("default.by_region", schema=schema, partition_spec=spec)
    arrow_schema = schema_to_pyarrow(schema)
    table.append(
        pa.Table.from_pylist([{"region": region, "id": i} for i, region in enumerate(["eu", "us", "apac"])], schema=arrow_schema)
    )

    source = pa.Table.from_pylist([{"region": "us", "id": 1}], schema=arrow_schema)
    plan = plan_for(table, source, ["region", "id"])

    assert [task.file.partition[0] for task in plan.tasks] == ["us"]


def test_task_reads_through_the_ordinary_scan_path(catalog: InMemoryCatalog) -> None:
    """``UpsertFileTask`` is a ``FileScanTask`` plus the source slice."""
    table = three_files(catalog, "default.task_shape")

    task = plan_for(table, rows([5]), ["id"]).tasks[0]
    scan_task = task.as_scan_task()

    assert isinstance(scan_task, FileScanTask)
    assert scan_task.file == task.file
    assert scan_task.delete_files == task.delete_files
    assert scan_task.residual == task.residual


def test_upsert_commits_exactly_one_snapshot(catalog: InMemoryCatalog) -> None:
    table = three_files(catalog, "default.single_snapshot")
    before = len(table.snapshots())

    result = table.upsert(rows([5, 15, 99], payload="b"), join_cols=["id"])

    assert (result.rows_updated, result.rows_inserted) == (2, 1)
    assert len(table.snapshots()) == before + 1
    assert table.snapshots()[-1].summary.operation == Operation.OVERWRITE


def test_upsert_of_only_new_rows_is_an_append(catalog: InMemoryCatalog) -> None:
    """Nothing is replaced, so there is nothing to overwrite."""
    table = three_files(catalog, "default.append_only_upsert")

    result = table.upsert(rows([100, 101]), join_cols=["id"])

    assert (result.rows_updated, result.rows_inserted) == (0, 2)
    assert table.snapshots()[-1].summary.operation == Operation.APPEND


def test_large_composite_key_upsert(catalog: InMemoryCatalog) -> None:
    """A composite key used to build one ``Or`` disjunct per row.

    At this size that predicate overflowed PyArrow's expression canonicaliser and
    took the process down with it (apache/iceberg-python#3508, #2675). Nothing here
    builds a per-row expression, so size is not a factor.
    """
    schema = Schema(
        NestedField(1, "group_id", IntegerType(), required=True),
        NestedField(2, "id", IntegerType(), required=True),
        NestedField(3, "payload", StringType(), required=True),
        identifier_field_ids=[1, 2],
    )
    table = catalog.create_table("default.composite_at_scale", schema=schema)
    arrow_schema = schema_to_pyarrow(schema)

    count = 20_000
    initial = pa.Table.from_pylist([{"group_id": i % 50, "id": i, "payload": "a"} for i in range(count)], schema=arrow_schema)
    table.append(initial)

    source = pa.Table.from_pylist(
        [{"group_id": i % 50, "id": i, "payload": "b"} for i in range(count // 2, count + count // 2)],
        schema=arrow_schema,
    )
    result = table.upsert(source, join_cols=["group_id", "id"])

    assert (result.rows_updated, result.rows_inserted) == (count // 2, count // 2)
    assert len(table.scan().to_arrow()) == count + count // 2


def test_merge_ranges_collapses_overlaps() -> None:
    starts, ends = _merge_ranges([(1, 5), (4, 8), (20, 25), (9, 10)])

    assert list(starts) == [1, 9, 20]
    assert list(ends) == [8, 10, 25]


def test_rows_that_differ_treats_null_as_a_change() -> None:
    fields = [NestedField(2, "value", IntegerType(), required=False)]
    source = pa.table({"value": pa.array([1, None, 3, None], type=pa.int32())})
    target = pa.table({"value": pa.array([1, 2, None, None], type=pa.int32())})

    assert rows_that_differ(source, target, fields).to_pylist() == [False, True, True, False]


def test_rows_that_differ_falls_back_to_python_for_nested_columns() -> None:
    nested = StructType(NestedField(3, "sub", StringType(), required=True))
    fields = [
        NestedField(2, "value", IntegerType(), required=False),
        NestedField(4, "nested", nested, required=False),
    ]
    struct_type = pa.struct([pa.field("sub", pa.string(), nullable=False)])
    source = pa.table(
        {
            "value": pa.array([1, 2, 3], type=pa.int32()),
            "nested": pa.array([{"sub": "x"}, {"sub": "y"}, {"sub": "z"}], type=struct_type),
        }
    )
    target = pa.table(
        {
            "value": pa.array([1, 9, 3], type=pa.int32()),
            "nested": pa.array([{"sub": "x"}, {"sub": "y"}, {"sub": "CHANGED"}], type=struct_type),
        }
    )

    # Row 1 is caught by the primitive column, row 2 only by the nested one.
    assert rows_that_differ(source, target, fields).to_pylist() == [False, True, True]


def test_source_index_sorts_on_the_most_selective_column() -> None:
    schema = Schema(
        NestedField(1, "group_id", IntegerType(), required=True),
        NestedField(2, "id", IntegerType(), required=True),
    )
    source = pa.table(
        {
            "group_id": pa.array([1, 1, 2, 2], type=pa.int32()),
            "id": pa.array([40, 10, 30, 20], type=pa.int32()),
        }
    )

    index = SourceIndex(source, ["group_id", "id"], schema)

    assert index.slice_col == "id"
    assert index.table["id"].to_pylist() == [10, 20, 30, 40]
    assert np.array_equal(index.values["id"], np.array([10, 20, 30, 40]))


def test_file_tasks_never_run_on_the_shared_executor(catalog: InMemoryCatalog, monkeypatch: pytest.MonkeyPatch) -> None:
    """File tasks must not occupy ``ExecutorFactory``'s pool.

    Each task reads its file through ``ArrowScan``, which submits to that shared
    pool and blocks on the result. Running the tasks there too lets them fill it
    and wait forever on nested work that can never be scheduled -- a deadlock that
    only appears once there are at least as many tasks as pool threads, so it
    hides completely in small tests.
    """
    from pyiceberg.table import upsert as upsert_module
    from pyiceberg.utils.concurrent import ExecutorFactory

    table = catalog.create_table("default.shared_pool", schema=SCHEMA)
    for start in range(1, 61, 10):
        table.append(rows(range(start, start + 10)))

    task_threads: set[threading.Thread] = set()
    run_task = upsert_module.execute_upsert_task

    def record(*args: object, **kwargs: object) -> object:
        task_threads.add(threading.current_thread())
        return run_task(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(upsert_module, "execute_upsert_task", record)

    shared_pool = ExecutorFactory.get_or_create()
    result = table.upsert(rows([2, 12, 22, 32, 42, 52], payload="b"), join_cols=["id"])

    assert result.rows_updated == 6
    assert len(task_threads) > 1, "tasks did not run in parallel"
    assert task_threads.isdisjoint(shared_pool._threads)  # type: ignore[attr-defined]
