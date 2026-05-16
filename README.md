# PyPharao

Python library for **3D pharmacophore** representation and **Gaussian volume alignment** matching the [Pharao](https://github.com/silicos-it/pharao) C++ implementation, with optional perception from 3D molecules (via RDKit).

The whole public API lives at the top of the package, so a single import is enough:

```python
from pypharao import *
```

Use this in scripts/notebooks; in a library it is usually nicer to import the names you actually use.

## 1. Installation

PyPharao needs **NumPy** and (optionally) **RDKit** for molecule perception and SDF output, plus **tqdm** for progress bars.

### pip

```bash
pip install -e ".[dev,rdkit,examples]"
```

(`examples` pulls in `tqdm`. To add it later: `pip install tqdm`.)

If `pip install rdkit` is unavailable on your platform, install RDKit from `conda-forge` first:

```bash
conda install -c conda-forge rdkit numpy
pip install -e .
```

### conda

```bash
conda env create -f environment.yml
conda activate pypharao
```

A local conda recipe is provided in `conda-recipe/`:

```bash
conda build conda-recipe
```

## 2. Pharmacophores and PharmacophorePoints

A **pharmacophore search** consists of one **query pharmacophore** and one or more **database (molecule) pharmacophores** that are matched against it. PyPharao models that explicitly:

- `Pharmacophore` — base class. Holds an ordered list of `PharmacophorePoint`s and validates inserts against its `allowed_types`.
- `QueryPharmacophore(Pharmacophore)` — the search query. Allowed types: **all 11**. Carries a `name`.
- `MoleculePharmacophore(Pharmacophore)` — perceived from a database molecule on the fly. Allowed types: the **7 elementary + compound types**. No `EXCL`, no `UNDEF`.

### `PharmacophorePoint`

A point is one Gaussian feature site. It has four public attributes:

| Attribute | Type                                  | Meaning                                                          |
| --------- | ------------------------------------- | ---------------------------------------------------------------- |
| `type`    | `PointType`                           | Feature type (also exposed as `PharmacophorePoint.Type`).        |
| `center`  | `(x, y, z)` floats in ångström        | Location.                                                        |
| `sigma`   | `float` (ångström)                    | Gaussian width.                                                  |
| `normal`  | `(nx, ny, nz)` floats or `None`       | Absolute tip coordinates of the feature normal (Pharao convention) for types that carry one; `None` otherwise. |

Points are immutable; use `point.replace(...)` to derive a modified copy.

```python
p = PharmacophorePoint(type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1))
q = p.replace(center=(1, 0, 0))   # new sigma defaults are filled in for you
```

### `PointType` table

| Type             | Default sigma | Normal | Description                                                  |
| ---------------- | ------------- | ------ | ------------------------------------------------------------ |
| `AROM`           | 0.7           | yes    | Aromatic ring centroids with plane normals                   |
| `LIPO`           | 0.7           | no     | Lipophilic regions (from molecular surface), no aromatics    |
| `AROM|LIPO`      | 0.7           | yes    | Either an AROM or a LIPO group, or both (query only)          |
| `HDON`           | 1.0           | no     | H-bond donors (N/O with H, not negatively charged)           |
| `HACC`           | 1.0           | no     | H-bond acceptors (N/O, with Pharao-style filters)            |
| `HACC&HDON`      | 1.0           | no     | Both a HACC and an HDON at the same site                     |
| `HACC|HDON`      | 1.0           | no     | Either an HACC or an HDON group (query only)                  |
| `POSC`           | 1.0           | no     | Positively charged atoms                                     |
| `NEGC`           | 1.0           | no     | Negatively charged atoms                                     |
| `EXCL`           | 1.6           | no     | Exclusion sphere (query only; penalises overlap)             |
| `UNDEF`          | 1.0           | no     | Undefined placeholder (matches any molecule feature type)    |

`PointType` enum members use underscores (`PointType.AROM_OR_LIPO`, `PointType.HACC_AND_HDON`, `PointType.HACC_OR_HDON`) but the `.value` strings keep the Pharao-style `|` / `&` notation.

The defaults are available as `DEFAULT_SIGMA[PointType.X]` and `TYPE_HAS_NORMAL[PointType.X]`.

### Allowed types per subclass

|                          | Allowed `PointType`s                                                          |
| ------------------------ | ----------------------------------------------------------------------------- |
| `QueryPharmacophore`     | every `PointType` (including `EXCL`, `UNDEF`, `AROM|LIPO`, `HACC|HDON`, `HACC&HDON`) |
| `MoleculePharmacophore`  | `AROM, LIPO, HDON, HACC, HACC&HDON, POSC, NEGC`                               |

Adding a point of a disallowed type raises `ValueError`.

### Editing a pharmacophore

```python
q = QueryPharmacophore(name="example")
q.add_point(PharmacophorePoint(type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1)))
q.add_point(PharmacophorePoint(type=PointType.HDON, center=(1, 0, 0)))

len(q)                     # 2
for p in q: ...            # iterate
q[0]                       # index access
q.set_point(0, q[0].replace(center=(0.5, 0, 0)))
q.remove_point(1)
q.clear()
q.copy()                   # shallow copy preserving subclass + name
q.set_name("query_v2")
q.get_name()
```

`q.points` returns a snapshot list; mutate the collection through `add_point` / `set_point` / `remove_point` / `clear` so the type rules are enforced.

### Generating query pharmacophores

#### Manually

```python
from pypharao import *

q = QueryPharmacophore(name="manual")
q.add_point(PharmacophorePoint(type=PointType.AROM_OR_LIPO, center=(0, 0, 0), normal=(0, 0, 1)))
q.add_point(PharmacophorePoint(type=PointType.HACC_AND_HDON, center=(3, 0, 0)))
q.add_point(PharmacophorePoint(type=PointType.EXCL, center=(0, 0, 2.5)))
```

#### From JSON / Pharao `.phar`

```python
q = Pharmacophore.from_json_file("query.json")
q = Pharmacophore.read_phar("query.phar")
```

The reader dispatches to `QueryPharmacophore` or `MoleculePharmacophore` based on the `"kind"` field; `.phar` files are read as `QueryPharmacophore` (with the header line as the name).

#### From a 3D molecule (requires RDKit)

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from pypharao import *

mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

q = query_pharmacophore_from_molecule(mol, name="phenol")
```

`query_pharmacophore_from_molecule` returns a `QueryPharmacophore`. Compound query types (`AROM|LIPO`, `HACC|HDON`) are **not** auto-perceived — refine them by hand if desired. `HACC&HDON` is created automatically whenever an `HDON` and an `HACC` sit at the same atom (the two originals are removed).

```python
query_pharmacophore_from_protein(...)  # raises NotImplementedError (placeholder)
```

## 3. Pharmacophore perception

A `PharmacophorePerception` instance controls **which feature types** are emitted when a pharmacophore is built from a 3D molecule. There are two subclasses:

- `QueryPharmacophorePerception` — flags driving `query_pharmacophore_from_molecule`.
- `MoleculePharmacophorePerception` — flags driving the molecule pharmacophore perception used inside `PharmacophoreSearch.screen`.

Both subclasses cover the same seven auto-perceivable types and default to *all enabled*:

```
AROM, LIPO, HDON, HACC, HACC&HDON, POSC, NEGC
```

`EXCL`, `UNDEF`, `AROM|LIPO` and `HACC|HDON` are never auto-perceived; add them manually if you need them. The base `PharmacophorePerception` is abstract — instantiate one of the two subclasses instead.

### API

```python
perception = QueryPharmacophorePerception()
perception.print_features()                  # one line per feature type
perception.is_enabled(PointType.LIPO)         # True
perception.disable(PointType.LIPO)
perception.enable("HACC&HDON")
perception.types_enabled()                    # list of currently enabled types
for t in perception: ...                      # iterate over allowed types
```

When `query_pharmacophore_from_molecule` (or the internal molecule perception used by `screen()`) is called with no perception argument, the relevant default is used.

## 4. Performing a pharmacophore search

A search needs:

1. A `QueryPharmacophore`.
2. A database of 3D molecules (the caller is responsible for embedding/optimising them).
3. A `PharmacophoreSearch` instance bound to the query, optionally configured with a `MoleculePharmacophorePerception`.

```python
searcher = PharmacophoreSearch(query)                   # all molecule types perceived
searcher = PharmacophoreSearch(query, perception=opts)  # custom molecule perception
```

### `screen()` parameters

```python
hits = searcher.screen(
    mols,                       # one Chem.Mol or a list of mols / tuples
    conformations="all",        # 'all' | 'single' | int N
    min_matches=None,           # default = matchable query points
    keep="best",                # 'best' (one row per molecule) or 'all'
    metric="tanimoto",          # tie-break for keep='best'
    n_jobs=0,                   # 0 = all CPUs, 1 = sequential
    progress=True,              # tqdm progress bar over the molecule list
)
```

| Argument         | Default       | Description                                                                                                  |
| ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------ |
| `mols`           | —             | A single `Chem.Mol`, a list of `Chem.Mol`, or a list of tuples whose last element is a mol (the first element becomes the reported hit index, e.g. `(line_idx, smiles, mol)`). |
| `conformations`  | `"all"`       | `"all"` iterates every conformer of each molecule; `"single"` uses only the first; a positive `int N` uses the first N. |
| `min_matches`    | `None`        | Minimum number of query points that must map to a molecule feature for a hit. Defaults to `count_matchable_query_points(query)` (every point that is not `EXCL`). Values < 1 or > the matchable count raise `ValueError`. |
| `keep`           | `"best"`      | `"best"` keeps the single highest-scoring conformer per molecule; `"all"` keeps every conformer that satisfies `min_matches` (each result records the matching `conf_id`). |
| `metric`         | `"tanimoto"`  | Tie-breaker for `keep="best"`. One of `tanimoto`, `overlap_volume`, `excl_volume`, `tversky_ref`, `tversky_db`. Maximised, except `excl_volume` which is minimised. |
| `n_jobs`         | `0`           | Worker processes; `0` uses all CPUs, `1` runs sequentially.                                                  |
| `progress`       | `True`        | Show a tqdm progress bar (requires `tqdm`).                                                                  |

`screen()` **always** returns `list[tuple[int, MatchResult]]`. A single molecule with no match returns `[]`; a successful match returns `[(index, MatchResult)]`. With `keep="all"` a single molecule may produce several rows (one per matching conformer), and the same `index` is repeated.

## 5. Analysing the results

### `MatchResult` attributes

| Attribute                  | Type                    | Description                                                                                  |
| -------------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| `conf_id`                  | `int`                   | Conformer id of the input molecule that produced this match (`0` for single-conformer mols). |
| `ref_volume`               | `float`                 | Self-overlap volume of the query pharmacophore.                                              |
| `db_volume`                | `float`                 | Self-overlap volume of the molecule pharmacophore.                                           |
| `overlap_volume`           | `float`                 | Raw overlap volume between matched (query, molecule) pairs before subtracting exclusion overlap. |
| `excl_volume`              | `float`                 | Overlap volume between query `EXCL` points and molecule features (subtracted from the aligned score). |
| `tanimoto`                 | `float`                 | `aligned_overlap / (ref_volume + db_volume − aligned_overlap)`.                              |
| `tversky_ref`              | `float`                 | `aligned_overlap / ref_volume`.                                                              |
| `tversky_db`               | `float`                 | `aligned_overlap / db_volume`.                                                               |
| `mapping`                  | `list[tuple[int, int]]` | Matched feature pairs as `(query_index, molecule_index)` into the original point lists.       |
| `database_pharmacophore`   | `MoleculePharmacophore` | Molecule pharmacophore perceived for this conformer (before alignment).                      |
| `matched_db_pharmacophore` | `MoleculePharmacophore` | Subset of molecule features that appear in `mapping`, transformed into the query frame.       |
| `aligned_mol`              | `Chem.Mol \| None`      | Copy of the input molecule with the matching conformer transformed into the query frame.     |

### Sorting and printing

```python
hits = searcher.screen(mols)
ranked = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(ranked, limit=20)
```

`print_match_results(results, *, limit=None, file=None)` prints a tab-separated table — header row first, then one row per hit. `No hits.` is printed for empty lists. `file` is forwarded to `print(file=...)`; pass an open text stream to write the table to disk.

`sort_match_results(matches, *, sort, key)` returns a new sorted list; `key` is any numeric attribute on `MatchResult`.

### Writing hits to an SDF file

```python
n = write_hits_sdf(ranked[:50], "top_hits.sdf")
```

Each record is the aligned 3D molecule (one conformer) with SDF tags `index`, `conf_id`, `tanimoto`, `tversky_ref`, `tversky_db`, `overlap_volume`, `excl_volume`, `ref_volume`, `db_volume`. Hits without an aligned molecule are skipped with a one-line warning; the call returns the number of records written.

## License

LGPL-3.0-or-later (see `LICENSE`).
