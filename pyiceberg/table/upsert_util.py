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
import functools
import operator

import pyarrow as pa
from pyarrow import Table as pyarrow_table
from pyarrow import compute as pc

from pyiceberg.expressions import (
    AlwaysFalse,
    BooleanExpression,
    EqualTo,
    GreaterThanOrEqual,
    In,
    LessThanOrEqual,
)
from pyiceberg.expressions.visitors import IN_PREDICATE_LIMIT

# Reserved for joining the source and the target table on their row order
SOURCE_INDEX_COLUMN_NAME = "__source_index"
TARGET_INDEX_COLUMN_NAME = "__target_index"


def _validate_index_column_names(join_cols: list[str]) -> None:
    if SOURCE_INDEX_COLUMN_NAME in join_cols or TARGET_INDEX_COLUMN_NAME in join_cols:
        raise ValueError(
            f"{SOURCE_INDEX_COLUMN_NAME} and {TARGET_INDEX_COLUMN_NAME} are reserved for joining "
            f"DataFrames, and cannot be used as column names"
        ) from None


def create_scan_filter(df: pyarrow_table, join_cols: list[str]) -> BooleanExpression:
    """Build the filter that narrows the target table down to the files that can hold a matching key.

    The filter only prunes; the exact matching is done in PyArrow once the rows have been read, so it
    is free to match a superset of the rows that the source can match. Which superset is worth asking
    for depends on how many keys the source has:

    - Up to IN_PREDICATE_LIMIT keys the filter lists them, which lets the evaluators test every key
      against the bounds of each file and skip the files that fall in a gap between the keys.
    - Above that limit both the manifest and the metrics evaluator ignore a listed set and report that
      every file might match, so the filter falls back to a bounded predicate per join column. That
      keeps the pruning a low-cardinality column or a batch of consecutive keys allows, without the
      filter growing with the source.
    """
    unique_keys = df.select(join_cols).group_by(join_cols).aggregate([])

    if len(unique_keys) == 0:
        return AlwaysFalse()

    for col in join_cols:
        if df.column(col).null_count > 0:
            raise ValueError(f"Join columns cannot contain null values, but {col} does. No upsert executed")

    if len(unique_keys) <= IN_PREDICATE_LIMIT:
        if len(join_cols) == 1:
            return In(join_cols[0], unique_keys[0].to_pylist())

        return functools.reduce(
            operator.or_,
            [functools.reduce(operator.and_, [EqualTo(col, row[col]) for col in join_cols]) for row in unique_keys.to_pylist()],
        )

    return functools.reduce(operator.and_, [_create_column_filter(unique_keys, col) for col in join_cols])


def _create_column_filter(unique_keys: pyarrow_table, col: str) -> BooleanExpression:
    """Build the widest useful predicate for a single join column that does not grow with the source."""
    keys = unique_keys.column(col)
    values = keys.unique()

    if len(values) <= IN_PREDICATE_LIMIT:
        return In(col, values.to_pylist())

    bounds = pc.min_max(keys).as_py()
    return GreaterThanOrEqual(col, bounds["min"]) & LessThanOrEqual(col, bounds["max"])


def has_duplicate_rows(df: pyarrow_table, join_cols: list[str]) -> bool:
    """Check for duplicate rows in a PyArrow table based on the join columns."""
    return len(df.select(join_cols).group_by(join_cols).aggregate([([], "count_all")]).filter(pc.field("count_all") > 1)) > 0


def get_rows_to_update(source_table: pa.Table, target_table: pa.Table, join_cols: list[str]) -> pa.Table:
    """
    Return a table with rows that need to be updated in the target table based on the join columns.

    The table is joined on the identifier columns, and then checked if there are any updated rows.
    Those are selected and everything is renamed correctly.
    """
    all_columns = set(source_table.column_names)
    join_cols_set = set(join_cols)

    non_key_cols = list(all_columns - join_cols_set)

    if has_duplicate_rows(target_table, join_cols):
        raise ValueError("Target table has duplicate rows, aborting upsert")

    if len(target_table) == 0:
        # When the target table is empty, there is nothing to update :)
        return source_table.schema.empty_table()

    # We need to join on the identifier columns only, as PyArrow cannot do a join when non-join
    # columns have complex types
    # See: https://github.com/apache/arrow/issues/35785
    _validate_index_column_names(join_cols)

    # Step 1: Prepare source index with join keys and a marker index
    # Cast to target table schema, so we can do the join
    # See: https://github.com/apache/arrow/issues/37542
    source_index = (
        source_table.cast(target_table.schema)
        .select(join_cols_set)
        .append_column(SOURCE_INDEX_COLUMN_NAME, pa.array(range(len(source_table))))
    )

    # Step 2: Prepare target index with join keys and a marker
    target_index = target_table.select(join_cols_set).append_column(TARGET_INDEX_COLUMN_NAME, pa.array(range(len(target_table))))

    # Step 3: Perform an inner join to find which rows from source exist in target
    matching_indices = source_index.join(target_index, keys=list(join_cols_set), join_type="inner")

    if len(matching_indices) == 0 or len(non_key_cols) == 0:
        return source_table.schema.empty_table()

    # Step 4: Compare the non-key columns of the matched rows, one column at a time
    matched_target = target_table.take(matching_indices[TARGET_INDEX_COLUMN_NAME])
    # Cast to the target types, so the values are compared the way the join keys are matched
    matched_source = source_table.take(matching_indices[SOURCE_INDEX_COLUMN_NAME]).cast(matched_target.schema)

    differs = functools.reduce(
        pc.or_, [_column_differs(matched_source.column(col), matched_target.column(col)) for col in non_key_cols]
    )

    # Step 5: Take the rows of the source table that hold a new value, in the order of the source
    to_update = matching_indices.filter(differs).sort_by(SOURCE_INDEX_COLUMN_NAME)
    return source_table.take(to_update[SOURCE_INDEX_COLUMN_NAME])


def _column_differs(source_column: pa.ChunkedArray, target_column: pa.ChunkedArray) -> pa.ChunkedArray:
    """Return a mask that is set for every row where the two columns hold a different value."""
    if pa.types.is_nested(source_column.type):
        # PyArrow cannot compare columns with complex types, so those are compared in Python
        # See: https://github.com/apache/arrow/issues/35785
        return pa.chunked_array(
            [pa.array([s != t for s, t in zip(source_column.to_pylist(), target_column.to_pylist(), strict=True)], pa.bool_())]
        )

    # not_equal yields null wherever either side is null, so decide those rows on their nullness alone
    both_null = pc.and_(pc.is_null(source_column), pc.is_null(target_column))
    either_null = pc.or_(pc.is_null(source_column), pc.is_null(target_column))
    return pc.if_else(either_null, pc.invert(both_null), pc.not_equal(source_column, target_column))


def get_rows_to_insert(source_table: pa.Table, target_table: pa.Table, join_cols: list[str]) -> pa.Table:
    """
    Return the rows of the source table that do not match any row in the target table.

    The tables are joined on the identifier columns only, which keeps the row matching in
    PyArrow instead of expanding the keys into a boolean expression over the source table.
    """
    if len(source_table) == 0 or len(target_table) == 0:
        return source_table

    _validate_index_column_names(join_cols)

    # Cast the keys to the target types, so we can do the join
    # See: https://github.com/apache/arrow/issues/37542
    target_keys = target_table.select(join_cols)
    source_keys = (
        source_table.select(join_cols)
        .cast(target_keys.schema)
        .append_column(SOURCE_INDEX_COLUMN_NAME, pa.array(range(len(source_table))))
    )

    unmatched = source_keys.join(target_keys, keys=join_cols, join_type="left anti")

    # The join does not preserve the order of the source table, so sort the indices back
    return source_table.take(unmatched.sort_by(SOURCE_INDEX_COLUMN_NAME)[SOURCE_INDEX_COLUMN_NAME])


def replace_rows(target_table: pa.Table, replacement_table: pa.Table, join_cols: list[str]) -> pa.Table:
    """
    Return the target table with every row that matches the replacement table swapped for its new version.

    The replacements keep the position of the rows they replace, so rewriting a data file does not
    reshuffle it and the clustering of a sorted table survives the rewrite.
    """
    if len(replacement_table) == 0:
        return target_table

    _validate_index_column_names(join_cols)

    target_keys = target_table.select(join_cols)
    target_index = target_keys.append_column(TARGET_INDEX_COLUMN_NAME, pa.array(range(len(target_table))))
    # Cast the keys to the target types, so we can do the join
    # See: https://github.com/apache/arrow/issues/37542
    replacement_index = (
        replacement_table.select(join_cols)
        .cast(target_keys.schema)
        .append_column(SOURCE_INDEX_COLUMN_NAME, pa.array(range(len(replacement_table))))
    )

    matches = target_index.join(replacement_index, keys=join_cols, join_type="inner")

    # Address the replacements by their offset in the concatenated table, so both the rows that stay
    # and the rows that are replaced can be gathered in a single take
    combined = pa.concat_tables([target_table, replacement_table.cast(target_table.schema)])
    indices = list(range(len(target_table)))
    for target_idx, replacement_idx in zip(
        matches[TARGET_INDEX_COLUMN_NAME].to_pylist(),
        matches[SOURCE_INDEX_COLUMN_NAME].to_pylist(),
        strict=True,
    ):
        indices[target_idx] = len(target_table) + replacement_idx

    return combined.take(indices)
