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
    LessThanOrEqual,
)

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

    The filter is a range over the source keys per join column, so its size is bound by the number
    of join columns instead of the number of source rows. It matches a superset of the rows that the
    source can match; the exact matching is done in PyArrow once the rows have been read.
    """
    if len(df) == 0:
        return AlwaysFalse()

    predicates: list[BooleanExpression] = []
    for col in join_cols:
        keys = df.column(col)
        if keys.null_count > 0:
            raise ValueError(f"Join columns cannot contain null values, but {col} does. No upsert executed")

        bounds = pc.min_max(keys).as_py()
        if bounds["min"] == bounds["max"]:
            predicates.append(EqualTo(col, bounds["min"]))
        else:
            predicates.append(GreaterThanOrEqual(col, bounds["min"]))
            predicates.append(LessThanOrEqual(col, bounds["max"]))

    return functools.reduce(operator.and_, predicates)


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

    # We need to compare non_key_cols in Python as PyArrow
    # 1. Cannot do a join when non-join columns have complex types
    # 2. Cannot compare columns with complex types
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

    # Step 4: Compare all rows using Python
    to_update_indices = []
    for source_idx, target_idx in zip(
        matching_indices[SOURCE_INDEX_COLUMN_NAME].to_pylist(),
        matching_indices[TARGET_INDEX_COLUMN_NAME].to_pylist(),
        strict=True,
    ):
        source_row = source_table.slice(source_idx, 1)
        target_row = target_table.slice(target_idx, 1)

        for key in non_key_cols:
            source_val = source_row.column(key)[0].as_py()
            target_val = target_row.column(key)[0].as_py()
            if source_val != target_val:
                to_update_indices.append(source_idx)
                break

    # Step 5: Take rows from source table using the indices and cast to target schema
    if to_update_indices:
        return source_table.take(to_update_indices)
    else:
        return source_table.schema.empty_table()


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
