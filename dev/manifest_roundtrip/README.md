# Manifest type round-trip audit

Probes for the "delegate manifest reading to iceberg-rust via pyiceberg-core" epic (issue 45 on this fork).
They write manifests with PyIceberg's own writer and read them back through `pyiceberg_core.manifest`,
using `fastavro` as the on-disk control. Panics are caught with `except BaseException`.

```
pip install 'pyiceberg==0.12.0' 'pyiceberg-core==0.10.1' pyarrow fastavro
python manifest_roundtrip.py        # one identity-partitioned column per primitive type, bounds, surface, null partitions
python manifest_list_surface.py     # manifest-list surface, FieldSummary, bucket transform, null partition value, delete manifest
python bounds_edges.py              # decimal/int bound edge cases, unknown field ids, None vs empty lists
```

To audit `main` instead of the releases: `pip install -e <iceberg-python checkout>[pyarrow]`, then build the binding
from an iceberg-rust checkout with `maturin build --profile dev -o dist` in `bindings/python` and `pip install dist/*.whl`.

Rust probes for `crates/iceberg/tests/` in an iceberg-rust checkout (set `DIR` / `OUT` to the `.avro` files the
Python scripts leave in `/tmp/manifest_roundtrip_*`):

```
cargo test -p iceberg --test manifest_roundtrip_probe -- --nocapture   # parse the PyIceberg-written files with core
cargo test -p iceberg --test uuid_write_probe -- --nocapture           # try to write a uuid partition with core
```
