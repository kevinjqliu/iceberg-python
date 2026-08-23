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
"""Planning and execution of upserts.

An upsert is "update the row if its key exists, insert it if it does not". Both
halves are decided per data file, so the operation is decomposed the same way a
read is:

1. **Plan** -- :class:`UpsertPlanner` reads table metadata only and emits one
   :class:`UpsertFileTask` per data file that could hold a matching row, plus the
   source rows that provably match nothing and can be inserted without any read.
2. **Execute** -- each task is independent: read the file, match it against its
   slice of the incoming dataframe, and write the rows that survive. Tasks run on
   the shared :class:`~pyiceberg.utils.concurrent.ExecutorFactory` pool.
3. **Commit** -- one snapshot referencing files that are already durable. Data
   files are written before the commit, exactly like ``append`` and
   ``dynamic_partition_overwrite`` do, because nothing is visible until commit.

``UpsertFileTask`` deliberately mirrors :class:`~pyiceberg.table.FileScanTask`:
the data file, the delete files that apply to it, and the residual predicate are
the same three fields. The only addition is the slice of the incoming dataframe
that could match that file -- the residual of the *source*, narrowed as far as
metadata allows and handed to the executor to finish.
"""

from __future__ import annotations

import itertools
import os
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from pyiceberg.conversions import from_bytes
from pyiceberg.exceptions import NotInstalledError
from pyiceberg.expressions import (
    And,
    BooleanExpression,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThanOrEqual,
    Or,
)
from pyiceberg.expressions.visitors import IN_PREDICATE_LIMIT
from pyiceberg.manifest import DataFile, ManifestEntry, ManifestFile
from pyiceberg.schema import Schema
from pyiceberg.table import ALWAYS_TRUE, FileScanTask, ManifestGroupPlanner, ScanTask
from pyiceberg.table.metadata import TableMetadata
from pyiceberg.types import (
    DateType,
    NestedField,
    PrimitiveType,
    TimestampNanoType,
    TimestampType,
    TimestamptzNanoType,
    TimestamptzType,
    TimeType,
)
from pyiceberg.utils.concurrent import ExecutorFactory

if TYPE_CHECKING:
    from pyiceberg.io import FileIO

# Marker columns used to carry row positions through the Arrow joins. The joins
# only ever see the key columns plus these, so payload columns of any type -- a
# struct, a list -- never reach the join. See https://github.com/apache/arrow/issues/35785
_TARGET_POS = "__target_pos"
_SOURCE_POS = "__source_pos"

# Above this many rows the sorted-key index is skipped for key types that need
# boxed Python objects (strings, binary). Bounds-based pruning still applies.
_MAX_BOXED_KEYS = 5_000_000

# Cap on the distinct source values fed through a scalar (non-vectorised) transform
# when building the exact partition predicate.
_MAX_SCALAR_TRANSFORM_VALUES = 10 * IN_PREDICATE_LIMIT

# Iceberg stores dates, times and timestamps as integers, and both the file
# bounds and the expression literals use that representation. Casting the Arrow
# key columns the same way keeps bounds, literals and key values comparable.
_INTEGER_VALUE_SPACE: dict[type, pa.DataType] = {
    DateType: pa.int32(),
    TimeType: pa.int64(),
    TimestampType: pa.int64(),
    TimestamptzType: pa.int64(),
    TimestampNanoType: pa.int64(),
    TimestamptzNanoType: pa.int64(),
}


def _conjunction(predicates: list[BooleanExpression]) -> BooleanExpression:
    """``And`` over any number of predicates, including zero and one."""
    if not predicates:
        return ALWAYS_TRUE
    if len(predicates) == 1:
        return predicates[0]
    return And(*predicates)


def _to_value_space(array: pa.Array | pa.ChunkedArray, field_type: PrimitiveType) -> pa.Array | pa.ChunkedArray:
    """Cast an Arrow array into the value space Iceberg uses for bounds and literals."""
    target = _INTEGER_VALUE_SPACE.get(type(field_type))
    return array.cast(target) if target is not None else array


class ThreadSafeCounter:
    """A counter shared by the threads that write data files.

    ``itertools.count`` is atomic under the GIL today, but that is an interpreter
    detail rather than a guarantee, and free-threaded builds remove it. One
    uncontended lock acquisition per written file is not measurable.
    """

    def __init__(self, start: int = 0) -> None:
        self._counter = itertools.count(start)
        self._lock = threading.Lock()

    def __next__(self) -> int:
        with self._lock:
            return next(self._counter)

    def __iter__(self) -> ThreadSafeCounter:
        return self


def _numpy_key_values(array: pa.ChunkedArray, field_type: PrimitiveType) -> np.ndarray | None:
    """Return the non-null key values, sorted, in Iceberg's value space.

    ``None`` means this column cannot back the interval tests, in which case
    planning falls back to bounds-only pruning. That is always sound: the
    interval tests only ever *remove* files that cannot match.
    """
    try:
        casted = _to_value_space(array.combine_chunks(), field_type).drop_null()
        values = casted.to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, NotImplementedError, ValueError, TypeError):
        return None
    if values.dtype == object and len(values) > _MAX_BOXED_KEYS:
        return None
    return np.sort(values)


class SourceIndex:
    """The incoming dataframe, prepared once and shared by planning and execution.

    Preparing it costs one sort. In exchange every later question -- can this file
    hold a matching key, which source rows can this file possibly match, which
    source rows match nothing at all -- is a binary search rather than a scan.
    """

    def __init__(self, source: pa.Table, join_cols: list[str], schema: Schema, case_sensitive: bool = True) -> None:
        self.join_cols = join_cols
        self.schema = schema
        self.fields: dict[str, NestedField] = {col: schema.find_field(col, case_sensitive=case_sensitive) for col in join_cols}
        keys = source.select(join_cols)
        self.null_counts: dict[str, int] = {col: keys[col].null_count for col in join_cols}

        # Columns whose values can back the interval tests.
        indexable = [
            col
            for col in join_cols
            if isinstance(self.fields[col].field_type, PrimitiveType)
            and _numpy_key_values(keys[col], self.fields[col].field_type) is not None
        ]
        # The source is sorted on one column so that each file's candidate rows
        # are a contiguous range. Null keys have no place in that ordering, so a
        # nullable column is indexed but never used to slice.
        self.slice_col = self._pick_slice_col(keys, [col for col in indexable if not self.null_counts[col]])

        if self.slice_col is not None:
            order = [self.slice_col] + [col for col in join_cols if col != self.slice_col]
            indices = pc.sort_indices(source.select(order), sort_keys=[(col, "ascending") for col in order])
            self.table = source.take(indices)
        else:
            self.table = source

        table_keys = self.table.select(join_cols)
        self.values: dict[str, np.ndarray | None] = {
            col: (_numpy_key_values(table_keys[col], self.fields[col].field_type) if col in indexable else None)
            for col in join_cols
        }

    @staticmethod
    def _pick_slice_col(keys: pa.Table, candidates: list[str]) -> str | None:
        """The most selective join column, which slices the source finest."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda col: pc.count_distinct(keys[col]).as_py() or 0)

    def file_range(self, data_file: DataFile, col: str) -> tuple[Any, Any] | None:
        """The ``[lower, upper]`` this file holds for ``col``, in Iceberg's value space."""
        field_id = self.fields[col].field_id
        lower = data_file.lower_bounds.get(field_id) if data_file.lower_bounds else None
        upper = data_file.upper_bounds.get(field_id) if data_file.upper_bounds else None
        if lower is None or upper is None:
            return None
        field_type = self.fields[col].field_type
        try:
            return from_bytes(field_type, lower), from_bytes(field_type, upper)
        except (ValueError, TypeError):
            return None

    def _file_may_hold_null(self, data_file: DataFile, col: str) -> bool:
        counts = data_file.null_value_counts
        if not counts:
            return True
        return counts.get(self.fields[col].field_id, 0) > 0

    def file_may_match(self, entry: ManifestEntry) -> bool:
        """Can this data file hold a row whose key is in the source?

        Runs after the partition and metrics evaluators, as the extra per-entry
        predicate :meth:`ManifestGroupPlanner.plan_files` accepts. Iceberg
        truncates lower bounds downwards and upper bounds upwards, so
        ``[lower, upper]`` always contains the file's real range and a negative
        answer here can never drop a file that could match.
        """
        data_file = entry.data_file
        for col in self.join_cols:
            values = self.values[col]
            if values is None or len(values) == 0:
                continue
            if self.null_counts[col] and self._file_may_hold_null(data_file, col):
                continue
            file_range = self.file_range(data_file, col)
            if file_range is None:
                continue
            lower, upper = file_range
            position = np.searchsorted(values, lower, side="left")
            if position >= len(values) or values[position] > upper:
                return False
        return True

    def bounds_predicate(self) -> BooleanExpression:
        """A predicate over the join columns that every matching row satisfies.

        Two literals per column, so unlike a per-key ``In`` or a per-row ``Or`` it
        stays under :data:`IN_PREDICATE_LIMIT` and the manifest and metrics
        evaluators actually prune with it.
        """
        predicates: list[BooleanExpression] = []
        for col in self.join_cols:
            field_type = self.fields[col].field_type
            if not isinstance(field_type, PrimitiveType):
                continue
            null_count = self.null_counts[col]
            bounds = self._column_bounds(col, field_type)
            if bounds is None:
                if null_count:
                    predicates.append(IsNull(col))
                continue
            lower, upper = bounds
            in_range = And(GreaterThanOrEqual(col, lower), LessThanOrEqual(col, upper))
            predicates.append(Or(in_range, IsNull(col)) if null_count else in_range)
        return _conjunction(predicates)

    def _column_bounds(self, col: str, field_type: PrimitiveType) -> tuple[Any, Any] | None:
        try:
            casted = _to_value_space(self.table[col].combine_chunks(), field_type)
            min_max = pc.min_max(casted).as_py()
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, NotImplementedError, ValueError, TypeError):
            return None
        lower, upper = min_max["min"], min_max["max"]
        return None if lower is None or upper is None else (lower, upper)

    def partition_predicate(self, spec: Any) -> BooleanExpression:
        """An exact ``In`` over the partition values the source keys actually land in.

        The read path projects a *predicate* into partition space. An upsert can
        do better, because it holds the values: pushing them through the spec's
        transforms names the only partitions that can hold a match. A range
        predicate does not project through ``bucket`` at all, and bucket counts
        are small, so this is where most of the pruning comes from on a table
        partitioned by the key.
        """
        predicates: list[BooleanExpression] = []
        for partition_field in spec.fields:
            source = self.schema.find_field(partition_field.source_id)
            if source.name not in self.join_cols or self.null_counts[source.name]:
                continue
            values = self._partition_values(partition_field, source)
            if values:
                predicates.append(In(partition_field.name, values))
        return _conjunction(predicates)

    def _partition_values(self, partition_field: Any, source: NestedField) -> list[Any] | None:
        transform = partition_field.transform
        result_type = transform.result_type(source.field_type)
        column = self.table[source.name].combine_chunks()
        try:
            transformed = transform.pyarrow_transform(source.field_type)(column)
            unique = pc.unique(_to_value_space(transformed, result_type)).drop_null()
            if not 0 < len(unique) <= IN_PREDICATE_LIMIT:
                return None
            return unique.to_pylist()
        except (NotInstalledError, NotImplementedError, AttributeError, pa.ArrowNotImplementedError, ValueError):
            # `pyarrow_transform` needs the optional `pyiceberg-core` extra for
            # every transform but identity. Fall back to the scalar transform over
            # the distinct source values, which are already in Iceberg's value
            # space. Only worth it while that set is small.
            try:
                distinct = pc.unique(_to_value_space(column, source.field_type)).drop_null()
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, NotImplementedError, ValueError, TypeError):
                return None
            if len(distinct) > _MAX_SCALAR_TRANSFORM_VALUES:
                return None
            apply_transform = transform.transform(source.field_type)
            values = {apply_transform(value) for value in distinct.to_pylist()}
            values.discard(None)
            return sorted(values) if 0 < len(values) <= IN_PREDICATE_LIMIT else None

    def split(self, data_files: list[DataFile]) -> tuple[pa.Table, pa.Table]:
        """Split the source into rows that provably match nothing, and the rest.

        A key outside the union of every candidate file's range cannot be in the
        table, so those rows are inserts that no data file has to be read to
        discover. On an append-heavy upsert that is most of the dataframe.
        """
        empty = self.table.schema.empty_table()
        if not data_files:
            return self.table, empty
        col = self.slice_col
        if col is None:
            return empty, self.table

        ranges = [self.file_range(data_file, col) for data_file in data_files]
        if any(file_range is None for file_range in ranges):
            # A file without bounds could hold anything, so nothing is provably new.
            return empty, self.table

        starts, ends = _merge_ranges([file_range for file_range in ranges if file_range is not None])
        values = self.slice_values(self.table)
        if values is None:
            return empty, self.table

        position = np.searchsorted(starts, values, side="right") - 1
        covered = (position >= 0) & (values <= ends[np.clip(position, 0, len(ends) - 1)])
        return self.table.filter(pa.array(~covered)), self.table.filter(pa.array(covered))

    def slice_values(self, table: pa.Table) -> np.ndarray | None:
        """The slice column of ``table`` as a numpy array in Iceberg's value space."""
        if self.slice_col is None:
            return None
        try:
            casted = _to_value_space(table[self.slice_col].combine_chunks(), self.fields[self.slice_col].field_type)
            return casted.to_numpy(zero_copy_only=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, NotImplementedError, ValueError, TypeError):
            return None

    def slice_for(self, data_file: DataFile, candidate: pa.Table, candidate_values: np.ndarray | None) -> pa.Table:
        """The rows of ``candidate`` this file could match -- the source's residual.

        ``candidate`` is sorted on the slice column, so this is a contiguous
        zero-copy view. Loose for a composite key; the executor's join drops the
        false positives.
        """
        if self.slice_col is None or candidate_values is None:
            return candidate
        file_range = self.file_range(data_file, self.slice_col)
        if file_range is None:
            return candidate
        lower, upper = file_range
        start = int(np.searchsorted(candidate_values, lower, side="left"))
        stop = int(np.searchsorted(candidate_values, upper, side="right"))
        return candidate.slice(start, max(stop - start, 0))


def _merge_ranges(ranges: list[tuple[Any, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``[lower, upper]`` pairs into a sorted set of disjoint ranges."""
    starts: list[Any] = []
    ends: list[Any] = []
    for lower, upper in sorted(ranges, key=lambda file_range: file_range[0]):
        if starts and lower <= ends[-1]:
            if upper > ends[-1]:
                ends[-1] = upper
        else:
            starts.append(lower)
            ends.append(upper)
    return np.asarray(starts), np.asarray(ends)


@dataclass(init=False)
class UpsertFileTask(ScanTask):
    """Everything needed to apply an upsert to a single data file.

    The first three fields are exactly :class:`~pyiceberg.table.FileScanTask`.
    ``source`` is the one thing a read does not have: the rows of the incoming
    dataframe that could match this file.
    """

    file: DataFile
    delete_files: set[DataFile]
    residual: BooleanExpression
    source: pa.Table
    join_cols: list[str]

    def __init__(
        self,
        data_file: DataFile,
        source: pa.Table,
        join_cols: list[str],
        delete_files: set[DataFile] | None = None,
        residual: BooleanExpression = ALWAYS_TRUE,
    ) -> None:
        self.file = data_file
        self.delete_files = delete_files or set()
        self.residual = residual
        self.source = source
        self.join_cols = join_cols

    def as_scan_task(self) -> FileScanTask:
        """Read this file through the ordinary scan path, deletes included."""
        return FileScanTask(self.file, self.delete_files, self.residual)


@dataclass
class UpsertFileResult:
    """What executing one :class:`UpsertFileTask` produced."""

    file: DataFile
    matched_keys: pa.Table
    updated_rows: pa.Table
    added_files: list[DataFile] = field(default_factory=list)
    replaced: bool = False


@dataclass
class UpsertPlan:
    """The instruction set for an upsert, derived from table metadata alone."""

    join_cols: list[str]
    tasks: list[UpsertFileTask]
    definitely_new: pa.Table
    candidate_source: pa.Table


class UpsertPlanner(ManifestGroupPlanner):
    """Plan an upsert down to one task per data file, using metadata only.

    Inherits the read path's pruning wholesale -- manifest evaluator, partition
    evaluator, metrics evaluator, delete-file index, residuals -- and changes only
    how the partition filter is derived, because an upsert holds the key values
    and can name the partitions exactly instead of projecting a predicate.
    """

    def __init__(
        self,
        table_metadata: TableMetadata,
        io: FileIO,
        source_index: SourceIndex,
        case_sensitive: bool = True,
    ) -> None:
        self.source_index = source_index
        super().__init__(
            table_metadata=table_metadata,
            io=io,
            row_filter=source_index.bounds_predicate(),
            case_sensitive=case_sensitive,
        )

    def _build_partition_projection(self, spec_id: int) -> BooleanExpression:
        projected = super()._build_partition_projection(spec_id)
        exact = self.source_index.partition_predicate(self.table_metadata.specs()[spec_id])
        return projected if exact == ALWAYS_TRUE else And(projected, exact)

    def plan(self, manifests: list[ManifestFile]) -> UpsertPlan:
        scan_tasks = list(self.plan_files(manifests, manifest_entry_filter=self.source_index.file_may_match))
        definitely_new, candidate = self.source_index.split([scan_task.file for scan_task in scan_tasks])
        candidate_values = self.source_index.slice_values(candidate)

        tasks = []
        for scan_task in scan_tasks:
            source = self.source_index.slice_for(scan_task.file, candidate, candidate_values)
            if len(source) > 0:
                tasks.append(
                    UpsertFileTask(
                        data_file=scan_task.file,
                        source=source,
                        join_cols=self.source_index.join_cols,
                        delete_files=scan_task.delete_files,
                        residual=scan_task.residual,
                    )
                )
        return UpsertPlan(
            join_cols=self.source_index.join_cols,
            tasks=tasks,
            definitely_new=definitely_new,
            candidate_source=candidate,
        )


def _positions(length: int) -> pa.Array:
    return pa.array(np.arange(length, dtype=np.int64), type=pa.int64())


def match_rows(target: pa.Table, source: pa.Table, join_cols: list[str]) -> pa.Table:
    """Join target rows to source rows on the key columns.

    Only the key columns and two position markers take part, so payload columns
    of any type stay out of the join. See https://github.com/apache/arrow/issues/35785
    """
    target_keys = target.select(join_cols)
    source_keys = source.select(join_cols).cast(target_keys.schema)
    return target_keys.append_column(_TARGET_POS, _positions(len(target))).join(
        source_keys.append_column(_SOURCE_POS, _positions(len(source))),
        keys=join_cols,
        join_type="inner",
    )


def rows_that_differ(source_rows: pa.Table, target_rows: pa.Table, compare_fields: list[NestedField]) -> pa.Array:
    """A boolean mask over matched row pairs: does the source row change the target row?

    Primitive columns are compared with Arrow kernels, which release the GIL and
    so let the per-file tasks actually run in parallel. Nested columns still need
    Python -- Arrow cannot compare them -- but only for the rows no primitive
    column has already told apart.
    """
    differs: Any = pa.array(np.zeros(len(source_rows), dtype=bool))
    nested: list[str] = []
    for compare_field in compare_fields:
        name = compare_field.name
        if not isinstance(compare_field.field_type, PrimitiveType):
            nested.append(name)
            continue
        source_column, target_column = source_rows[name], target_rows[name]
        null_differs = pc.not_equal(pc.is_null(source_column), pc.is_null(target_column))
        value_differs = pc.fill_null(pc.not_equal(source_column, target_column), False)
        differs = pc.or_(differs, pc.or_(null_differs, value_differs))

    if nested:
        undecided = pc.indices_nonzero(pc.invert(differs)).to_pylist()
        also_differs = [
            row for row in undecided if any(source_rows[name][row].as_py() != target_rows[name][row].as_py() for name in nested)
        ]
        if also_differs:
            mask = np.asarray(differs.to_numpy(zero_copy_only=False), dtype=bool).copy()
            mask[also_differs] = True
            differs = pa.array(mask)
    return differs if isinstance(differs, pa.Array) else differs.combine_chunks()


def execute_upsert_task(
    task: UpsertFileTask,
    table_metadata: TableMetadata,
    io: FileIO,
    write_uuid: uuid.UUID,
    counter: ThreadSafeCounter,
    when_matched_update_all: bool,
) -> UpsertFileResult:
    """Apply one task and write its output.

    Data files are written here, before any commit. Nothing is visible until the
    commit references them, so the worst case of a failed upsert is unreferenced
    files -- the same trade ``append`` already makes -- and in exchange the file's
    rows are released as soon as the remainder is on storage.
    """
    from pyiceberg.io.pyarrow import ArrowScan, _dataframe_to_data_files

    schema = table_metadata.schema()
    no_keys = task.source.select(task.join_cols).schema.empty_table()
    no_rows = task.source.schema.empty_table()
    unchanged = UpsertFileResult(file=task.file, matched_keys=no_keys, updated_rows=no_rows)

    # Only the key columns are needed to decide what matched. Reading the rest is
    # pure waste unless the matched rows are going to be compared and rewritten.
    projection = schema if when_matched_update_all else schema.select(*task.join_cols)
    target = ArrowScan(table_metadata, io, projection, ALWAYS_TRUE).to_table([task.as_scan_task()])
    if len(target) == 0:
        return unchanged

    matched = match_rows(target, task.source, task.join_cols)
    if len(matched) == 0:
        return unchanged

    source_positions = matched[_SOURCE_POS]
    matched_keys = task.source.select(task.join_cols).take(source_positions)
    if not when_matched_update_all:
        return UpsertFileResult(file=task.file, matched_keys=matched_keys, updated_rows=no_rows)

    ordered_source = task.source.select([column.name for column in target.schema]).cast(target.schema)
    compare_fields = [column for column in schema.fields if column.name not in task.join_cols]
    differs = rows_that_differ(
        ordered_source.take(source_positions),
        target.take(matched[_TARGET_POS]),
        compare_fields,
    )

    changed = pc.indices_nonzero(differs)
    if len(changed) == 0:
        # Matched, but the values are identical: rewriting the file would churn
        # storage for nothing.
        return UpsertFileResult(file=task.file, matched_keys=matched_keys, updated_rows=no_rows)

    updated_rows = task.source.take(source_positions.take(changed))
    superseded = matched[_TARGET_POS].take(changed).combine_chunks().to_numpy(zero_copy_only=False)
    keep = np.ones(len(target), dtype=bool)
    keep[superseded] = False
    remainder = target.filter(pa.array(keep))

    added_files = (
        list(
            _dataframe_to_data_files(
                table_metadata=table_metadata,
                df=remainder,
                io=io,
                write_uuid=write_uuid,
                counter=counter,
            )
        )
        if len(remainder) > 0
        else []
    )
    return UpsertFileResult(
        file=task.file,
        matched_keys=matched_keys,
        updated_rows=updated_rows,
        added_files=added_files,
        replaced=True,
    )


def _worker_count() -> int:
    workers = ExecutorFactory.max_workers()
    if workers is None:
        workers = min(32, (os.cpu_count() or 1) + 4)
    return max(workers, 1)


def execute_upsert_plan(
    plan: UpsertPlan,
    table_metadata: TableMetadata,
    io: FileIO,
    when_matched_update_all: bool,
    write_uuid: uuid.UUID,
    counter: ThreadSafeCounter,
) -> list[UpsertFileResult]:
    """Run every task in parallel, with bounded submission.

    The tasks get their own pool rather than ``ExecutorFactory``'s shared one.
    Each task reads its file through ``ArrowScan``, which submits to the shared
    pool itself and blocks on the result, so running the tasks there too lets
    them fill the pool and then wait forever on nested work that can never be
    scheduled. Keeping the tasks off it leaves the shared pool free to serve
    those reads.

    Submission is bounded because ``Executor.map`` would queue every task at
    once. That is fine when batches stream out of a read, but each in-flight task
    here holds a data file's worth of Arrow data, so peak memory is roughly
    ``in-flight x (file + remainder)`` -- which ``max-workers`` controls.
    """
    if not plan.tasks:
        return []

    workers = _worker_count()
    limit = max(2 * workers, 4)
    results: list[UpsertFileResult] = []
    pending: set[Future[UpsertFileResult]] = set()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="upsert") as executor:
        for task in plan.tasks:
            if len(pending) >= limit:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                results.extend(future.result() for future in done)
            pending.add(
                executor.submit(execute_upsert_task, task, table_metadata, io, write_uuid, counter, when_matched_update_all)
            )
        results.extend(future.result() for future in pending)
    return results


def rows_to_insert(candidate: pa.Table, matched_keys: pa.Table, join_cols: list[str]) -> pa.Table:
    """The candidate rows that no data file matched, in one anti-join."""
    if len(candidate) == 0:
        return candidate
    if len(matched_keys) == 0:
        return candidate
    positions = candidate.select(join_cols).append_column(_SOURCE_POS, _positions(len(candidate)))
    unmatched = positions.join(
        matched_keys.cast(positions.select(join_cols).schema),
        keys=join_cols,
        join_type="left anti",
    )
    return candidate.take(unmatched[_SOURCE_POS])
