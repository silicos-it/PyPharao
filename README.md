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


| Attribute | Type                            | Meaning                                                                                                        |
| --------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `type`    | `PointType`                     | Feature type (also exposed as `PharmacophorePoint.Type`).                                                      |
| `center`  | `(x, y, z)` floats in ångström  | Location.                                                                                                      |
| `sigma`   | `float` (ångström)              | Gaussian width.                                                                                                |
| `normal`  | `(nx, ny, nz)` floats or `None` | Absolute tip coordinates of the feature normal (Pharao convention) for types that carry one; `None` otherwise. |


Points are immutable; use `point.replace(...)` to derive a modified copy.

```python
p = PharmacophorePoint(type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1))
q = p.replace(center=(1, 0, 0))   # new sigma defaults are filled in for you
```

### `PointType` table


| Type            | Default sigma | Normal | Description                                               |
| --------------- | ------------- | ------ | --------------------------------------------------------- |
| `AROM`          | 0.7           | yes    | Aromatic ring centroids with plane normals                |
| `LIPO`          | 0.7           | no     | Lipophilic regions (from molecular surface), no aromatics |
| `AROM_OR_LIPO`  | 0.7           | yes    | Either an AROM or a LIPO group, or both (query only)      |
| `HDON`          | 1.0           | no     | H-bond donors (N/O with H, not negatively charged)        |
| `HACC`          | 1.0           | no     | H-bond acceptors (N/O, with Pharao-style filters)         |
| `HACC_AND_HDON` | 1.0           | no     | Both a HACC and an HDON at the same site                  |
| `HACC_OR_HDON`  | 1.0           | no     | Either an HACC or an HDON group (query only)              |
| `POSC`          | 1.0           | no     | Positively charged atoms                                  |
| `NEGC`          | 1.0           | no     | Negatively charged atoms                                  |
| `EXCL`          | 1.6           | no     | Exclusion sphere (query only; penalises overlap)          |
| `UNDEF`         | 1.0           | no     | Undefined placeholder (matches any molecule feature type) |


`PointType` enum members and their underlying `.value` strings both use the underscored spellings (`PointType.AROM_OR_LIPO`, `PointType.HACC_AND_HDON`, `PointType.HACC_OR_HDON`).

The defaults are available as `DEFAULT_SIGMA[PointType.X]` and `TYPE_HAS_NORMAL[PointType.X]`.

### Allowed types per subclass


|                         | Allowed `PointType`s                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| `QueryPharmacophore`    | every `PointType` (including `EXCL`, `UNDEF`, `AROM_OR_LIPO`, `HACC_OR_HDON`, `HACC_AND_HDON`) |
| `MoleculePharmacophore` | `AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC`                                            |


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
q.update_point(0, type=PointType.AROM_OR_LIPO)  # shorthand for set_point + replace
q.remove_point(1)
q.clear()
q.copy()                   # shallow copy preserving subclass + name
q.set_name("query_v2")
q.get_name()
```

> **Note on copying.** `pharmacophore_2 = pharmacophore_1` only binds a second
> name to the *same* object; later edits affect both names. Use
> `pharmacophore_2 = pharmacophore_1.copy()` whenever you need an independent
> pharmacophore to modify (for example to derive an `AROM → AROM_OR_LIPO`
> variant). The copy is shallow: it produces a new `Pharmacophore` whose
> `_points` list is new, but the `PharmacophorePoint` objects themselves are
> shared. Since `PharmacophorePoint` is immutable (`point.replace(...)` returns
> a new instance, line 64), this is safe in practice.

`q.points` returns a snapshot list; mutate the collection through `add_point` / `set_point` / `remove_point` / `clear` so the type rules are enforced.

### `PharmacophorePoint` API reference

`PharmacophorePoint` is a `@dataclass(frozen=True)`, so instances are immutable and hashable (usable as dict keys / set members). The constructor validates inputs and raises `ValueError` for inconsistent combinations (e.g. a missing `normal` on a type that requires one, or providing a `normal` for a type that doesn't carry one).

**Fields**


| Name             | Type        | Notes                                                                                                     |
| ---------------- | ----------- | --------------------------------------------------------------------------------------------------------- |
| `type`           | `PointType` | Coerced from `str` at construction if needed.                                                             |
| `x`, `y`, `z`    | `float`     | Centre coordinates, ångström.                                                                             |
| `sigma`          | `float`     | Gaussian width, ångström. Defaults to `DEFAULT_SIGMA[type]` if omitted at construction.                   |
| `nx`, `ny`, `nz` | `float`     | Absolute tip coordinates of the feature normal (Pharao convention). All `0.0` for types without a normal. |


**Class attribute**


| Name                | Purpose                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------- |
| `Type` (`ClassVar`) | Alias for the `PointType` enum (so `PharmacophorePoint.Type.AROM` is `PointType.AROM`). |


**Constructor**

```python
PharmacophorePoint(
    type: PointType | str,
    center: tuple[float, float, float],
    sigma: float | None = None,                        # defaults to DEFAULT_SIGMA[type]
    normal: tuple[float, float, float] | None = None,  # required iff TYPE_HAS_NORMAL[type]
)
```

**Properties (read-only)**


| Property     | Returns                                                          |
| ------------ | ---------------------------------------------------------------- |
| `center`     | `(x, y, z)` tuple.                                               |
| `normal`     | `(nx, ny, nz)` tuple for types that carry a normal, else `None`. |
| `has_normal` | `bool` — shortcut for `TYPE_HAS_NORMAL[self.type]`.              |


**Methods**


| Method                                                        | Purpose                                                                                                                                                                                                                             |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `replace(*, type=None, center=None, sigma=None, normal=None)` | Returns a **new** `PharmacophorePoint` with selected fields updated (keyword-only). If `type` changes to one that doesn't carry a normal, the new normal is zeroed; if it does and you don't pass one, the old normal is preserved. |


**Auto-generated by `@dataclass(frozen=True)`**


| Member                    | Behaviour                                                               |
| ------------------------- | ----------------------------------------------------------------------- |
| `__eq__`                  | Structural equality across all eight fields.                            |
| `__hash__`                | Hashable (because `frozen=True`).                                       |
| `__repr__`                | `PharmacophorePoint(type=…, x=…, y=…, z=…, sigma=…, nx=…, ny=…, nz=…)`. |
| `__setattr__` *(blocked)* | Raises `FrozenInstanceError` — instances are immutable.                 |


### `Pharmacophore` API reference

The tables below list everything available on a `Pharmacophore` (and its `QueryPharmacophore` / `MoleculePharmacophore` subclasses). Anything that mutates the point list goes through `_check_type` and raises `ValueError` if the point's type isn't in the subclass's `allowed_types`.

**Class attributes**


| Name            | Type                             | Purpose                                                                         |
| --------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `allowed_types` | `ClassVar[frozenset[PointType]]` | Which point types may be added.                                                 |
| `kind`          | `ClassVar[str]`                  | Tag stored in JSON (`"query"` / `"molecule"`); used to round-trip the subclass. |


**Iteration and indexing**


| Member              | Behaviour                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `points` (property) | Returns a *snapshot* `list[PharmacophorePoint]`; mutating the list does **not** mutate the pharmacophore. |
| `len(ph)`           | Number of points.                                                                                         |
| `for p in ph: ...`  | Iterate points in order.                                                                                  |
| `ph[i]`             | Get the `PharmacophorePoint` at index `i`.                                                                |


**Editing points**


| Method         | Signature          | Notes                                                                                                          |
| -------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| `add_point`    | `(point)`          | Append after validating type.                                                                                  |
| `set_point`    | `(idx, point)`     | Replace at `idx` after validating type.                                                                        |
| `update_point` | `(idx, **changes)` | Shorthand for `set_point(idx, self[idx].replace(**changes))`. Accepts `type=`, `center=`, `sigma=`, `normal=`. |
| `remove_point` | `(idx)`            | Delete the point at `idx`.                                                                                     |
| `clear`        | `()`               | Remove every point.                                                                                            |


**Copying**


| Method   | Returns                                                                                                                                                                                 |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `copy()` | Shallow copy preserving the subclass and (for `QueryPharmacophore`) the name. The `_points` list is new; `PharmacophorePoint` objects are shared (they are immutable, so this is safe). |


**JSON I/O**

Import helpers are **classmethods**: they **return a new** ``Pharmacophore`` (usually ``QueryPharmacophore`` or ``MoleculePharmacophore`` from the JSON ``kind``). They do **not** modify an object you already constructed — always assign the result, e.g. ``q = Pharmacophore.read_json(path)``.


| Method                                 | Direction | Returns / accepts                                                                                     |
| -------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------- |
| `to_json_dict()`                       | export    | `dict[str, Any]`                                                                                    |
| `to_json(**kwargs)`                    | export    | `str` (kwargs forwarded to `json.dumps`; default `indent=2`; pass `indent=None` for one line)          |
| `write_json(path, **kwargs)`         | export    | writes UTF-8 file (same JSON defaults as `to_json`)                                                   |
| `from_json_dict(d)` *(classmethod)*    | import    | **returns new** pharmacophore; subclass from JSON ``kind``                                          |
| `from_json(s)` *(classmethod)*         | import    | **returns new** pharmacophore from a JSON string                                                   |
| `read_json(path)` *(classmethod)*      | import    | **returns new** pharmacophore from a UTF-8 JSON file                                                  |


**Pharao `.phar` text I/O**

``from_phar_text`` / ``read_phar`` are **classmethods** that **return a new** pharmacophore (same assignment rule as JSON).


| Method                                 | Direction | Returns / accepts                                                                 |
| -------------------------------------- | --------- | --------------------------------------------------------------------------------- |
| `to_phar_text()`                       | export    | `str` (Pharao `.phar` format: name line, point lines, ``$$$$``)                   |
| `write_phar(path)`                     | export    | writes UTF-8 file (same layout as `to_phar_text`)                                 |
| `from_phar_text(text)` *(classmethod)* | import    | **returns new** pharmacophore from a `.phar` string                               |
| `read_phar(path)` *(classmethod)*      | import    | **returns new** pharmacophore from a UTF-8 `.phar` file                           |


**SDF / PDB I/O (requires RDKit, export only)**


| Method                          | Direction                                                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write_sdf(path, *, name=None)` | write a single-record SDF (one pseudo-atom per feature, plus per-mol properties `kind`, `name`, `num_features`, `types`, `sigmas`, `centers`, `normals`) |
| `write_pdb(path, *, name=None)` | write a PDB with one `HETATM` per feature; the residue name encodes the feature type (`ARO`, `LIP`, `HDO`, `HAC`, `DAC`, ...)                            |


Both methods reuse `pharmacophore_to_mol` so the format matches what the multi-record `[write_hits_sdf` / `write_hits_pdb](#writing-hits-to-an-sdf-or-pdb-file)` writers produce. SDF/PDB are export-only; round-trip back to a `Pharmacophore` is not supported (use JSON or `.phar` for that).

**`QueryPharmacophore` extras**


| Member                           | Kind                   |
| -------------------------------- | ---------------------- |
| `__init__(points=None, name="")` | overridden constructor |
| `get_name()` / `set_name(name)`  | explicit accessors     |
| `name`                           | property + setter      |


`MoleculePharmacophore` adds no methods; it narrows `allowed_types` to `{AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC}`.

**Module-level helpers**


| Function               | Purpose                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| `distance(p, q)`       | Euclidean distance between two `PharmacophorePoint` centres.                                              |
| `cosine_normals(p, q)` | Cosine between the (relative) normal vectors of `p` and `q`; returns `0.0` if either point has no normal. |


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

These calls **return** a loaded pharmacophore; assign to a variable (calling them on an empty ``QueryPharmacophore()`` and ignoring the return value leaves that instance unchanged):

```python
q = Pharmacophore.read_json("query.json")
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

`query_pharmacophore_from_molecule` returns a `QueryPharmacophore`. Compound query types (`AROM_OR_LIPO`, `HACC_OR_HDON`) are **not** auto-perceived — refine them by hand if desired. `HACC_AND_HDON` is created automatically whenever an `HDON` and an `HACC` sit at the same atom (the two originals are removed).

```python
query_pharmacophore_from_protein(...)  # raises NotImplementedError (placeholder)
```

#### Adding excluded volumes around a molecule (requires RDKit)

`EXCL` points are never auto-perceived. The convenience builder `add_excluded_volume` paints a *shape-complementarity envelope* around one or more reference 3D structures by appending `EXCL` features to an existing pharmacophore:

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from pypharao import *

mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

q = query_pharmacophore_from_molecule(mol, name="phenol")
n_excl = add_excluded_volume(
    mol, q,
    shell_inner=1.0,        # start 1.0 Å outside the vdW surface
    shell_outer=2.5,        # end   2.5 Å outside the vdW surface
    spacing=1.5,            # grid step (Å)
    feature_clearance=1.5,  # drop grid points within 1.5 Å of an existing feature
    # max_excl=0 by default (no cap); pass max_excl=512 etc. to limit count
)
# Also allowed: add_excluded_volume([mol_a, mol_b, ...], q, ...) with ligands
# superimposed to the same frame; pass conf_id=0 (or any int) to use only that
# conformer index in each molecule.
```

Internally it lays a regular 3D grid over the bounding box of all heavy-atom centres that enter the calculation (vectorised with NumPy). Every heavy atom from every selected molecule and conformer is treated as part of **one artificial molecule**: at each vertex the vdW surface distance is the minimum over **all** those atoms (union of vdW spheres). Vertices in `[shell_inner, shell_outer]` relative to that union surface become candidates. After optional `feature_clearance` filtering, candidates are **thinned** (see below). Hydrogens are ignored. Several ligands should share one **aligned** coordinate frame.

**How markers are chosen (`spacing` and `max_excl`).** Selection is **not random**; it is fully **deterministic**. Candidate grid vertices are sorted by **increasing union vdW surface distance** (smaller values are closer to the combined vdW surface and are considered first). The implementation then scans that order and **keeps** a vertex only if its centre is at least **`spacing`** ångström away from **every** centre already kept—so accepted `EXCL` sites obey a minimum pairwise separation. By default **`max_excl` is `0`**, meaning there is **no upper bound**: thinning runs until every candidate has been visited. If **`max_excl`** is **positive**, scanning **stops as soon as** that many centres have been kept (still respecting the separation rule among kept points).

In Pharao scoring `EXCL` query points contribute two layers of shape filtering:

1. A **soft volume penalty** (`with_exclusion=True`): the overlap between each `EXCL` point and the candidate's pharmacophore features is subtracted from the aligned overlap volume and reflected by `MatchResult.excl_volume`.
2. A **hard atom-clash filter** (`excl_hard_filter=True`, on by default): after alignment, every heavy atom of the aligned database molecule is treated as a sphere of its van der Waals radius. The hit is rejected as soon as `dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius` for any heavy atom / EXCL pair. The default `excl_clash_radius=0.0` means the EXCL is a bare marker point — a clash fires whenever an atom's vdW sphere swallows the marker. Increase `excl_clash_radius` to enforce a larger buffer; set `excl_hard_filter=False` to recover the pure soft-penalty behaviour. This criterion is independent of the EXCL's Gaussian `sigma` (which only feeds the soft penalty above).

A shell placed *outside* the reference ligand therefore acts as a shape filter on two levels: candidates whose pharmacophore features extend beyond the envelope are down-weighted, *and* candidates whose heavy atoms intrude into the forbidden shell are discarded. `add_excluded_volume` expects a pharmacophore that allows `EXCL` (a `QueryPharmacophore`); passing a `MoleculePharmacophore` raises `ValueError`.


| Argument            | Default                          | Description                                                                                       |
| ------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------- |
| `mol`               | —                                | One 3D RDKit molecule **or** a non-empty iterable of molecules (heavy atoms only). Use a **shared** frame when passing several ligands.              |
| `pharmacophore`     | —                                | Target pharmacophore; must allow `EXCL` (typically a `QueryPharmacophore`).                       |
| `conf_id`           | `None`                           | `None`: union includes **all** conformers of every molecule. An `int`: only that conformer index per molecule. |
| `sigma`             | `DEFAULT_SIGMA[PointType.EXCL]`  | Gaussian width (Å) of each new `EXCL` point (default `1.6`).                                      |
| `shell_inner`       | `1.0`                            | Inner distance (Å) of the shell, measured from the nearest heavy-atom vdW surface (`>= 0`).       |
| `shell_outer`       | `3.0`                            | Outer distance (Å) of the shell (`> shell_inner`).                                                |
| `spacing`           | `1.5`                            | Grid step (Å) along each axis (`> 0`); also the minimum separation between output `EXCL` centres after thinning. |
| `feature_clearance` | `0.0`                            | If `> 0`, candidate EXCL points within this distance (Å) of an existing feature centre are dropped. |
| `max_excl`          | `0`                              | Maximum number of EXCL markers after thinning (`0` or negative = no limit); see **How markers are chosen** above. |


See `examples/03-excluded-volumes.py` for a full end-to-end example (query + EXCL envelope + screening), and `examples/04-excluded-volumes-from-multiple-ligands.py` for building an envelope from several aligned ligands and their conformers. Raise `spacing` for sparser sampling on the grid (and larger minimum separation between accepted centres). Pass a positive `max_excl` if you want to cap how many markers survive thinning (default `0` = no cap). Tune `shell_inner` / `shell_outer` to make the envelope tighter or looser.

## 3. Pharmacophore perception

A `PharmacophorePerception` instance controls **which feature types** are emitted when a pharmacophore is built from a 3D molecule. There are two subclasses:

- `QueryPharmacophorePerception` — flags driving `query_pharmacophore_from_molecule`.
- `MoleculePharmacophorePerception` — flags driving the molecule pharmacophore perception used inside `PharmacophoreSearch.screen`.

Both subclasses cover the same seven auto-perceivable types and default to *all enabled*:

```
AROM, LIPO, HDON, HACC, HACC_AND_HDON, POSC, NEGC
```

`AROM` and `LIPO` are mutually exclusive at perception time: an aromatic ring is reported only as `AROM`, while a lipophilic moiety that is *not* aromatic is reported as `LIPO`. `EXCL`, `UNDEF`, `AROM_OR_LIPO` and `HACC_OR_HDON` are never auto-perceived; introduce them by hand on the resulting `QueryPharmacophore` (e.g. by converting an `AROM`/`LIPO` point to `AROM_OR_LIPO` or an `HDON`/`HACC` pair to `HACC_OR_HDON`, or by calling [`add_excluded_volume`](#adding-excluded-volumes-around-a-molecule-requires-rdkit) to wrap a shape-complementarity envelope around a ligand — see `examples/03-excluded-volumes.py`). The base `PharmacophorePerception` is abstract — instantiate one of the two subclasses instead.

### API

```python
perception = QueryPharmacophorePerception()
perception.print_features()                  # one line per feature type
perception.is_enabled(PointType.LIPO)         # True
perception.disable(PointType.LIPO)
perception.enable("HACC_AND_HDON")
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

### `PharmacophoreSearch` API reference

`PharmacophoreSearch` is a `@dataclass`, so the constructor accepts the fields below (positional or keyword) and each field is also accessible / reassignable on an existing instance.

**Fields (= constructor parameters)**


| Name               | Type                                     | Default      | Purpose                                                                                                                     |
| ------------------ | ---------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `query`             | `QueryPharmacophore`                     | *(required)* | Query pharmacophore. A `TypeError` is raised at construction if this isn't a `QueryPharmacophore`.                                                                                                              |
| `perception`        | `MoleculePharmacophorePerception | None` | `None`       | How database molecules are perceived. `None` ⇒ a default `MoleculePharmacophorePerception()` is created in `__post_init__`.                                                                                     |
| `epsilon`           | `float`                                  | `0.5`        | Tolerance used by `FunctionMapping` when proposing candidate feature mappings.                                                                                                                                  |
| `use_direction`     | `bool`                                   | `True`       | Use feature normals (AROM / AROM_OR_LIPO) in volume scoring and alignment.                                                                                                                                      |
| `with_exclusion`    | `bool`                                   | `True`       | Include `EXCL` spheres in the Pharao soft penalty (subtracted from the aligned overlap volume; reflected by `MatchResult.excl_volume`).                                                                         |
| `early_exit_score`  | `float`                                  | `0.98`       | Tanimoto threshold above which the search stops exploring further mappings.                                                                                                                                     |
| `excl_hard_filter`  | `bool`                                   | `True`       | After alignment, reject any conformer whose heavy atoms (ignoring hydrogens) intrude into a query `EXCL` sphere. Set `False` to revert to the pure Pharao soft penalty.                                                                                                                                                                  |
| `excl_clash_radius` | `float`                                  | `0.0`        | Extra padding (Å, `>= 0`) added to every heavy-atom van der Waals radius when checking for a clash with an `EXCL` marker. The clash criterion is `dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius`; `0.0` (default) means "an atom's vdW sphere must not contain any EXCL marker". Independent of the EXCL Gaussian `sigma`. |


**Public methods**


| Method                                                                                                         | Purpose                                                                                                                                               |
| -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `screen(mols, *, conformations='all', min_matches=0, keep='best', metric='tanimoto', n_jobs=0, progress=True)` | The single public entry point. See `[screen()` parameters](#screen-parameters) below for the full breakdown. Returns `list[tuple[int, MatchResult]]`. |


**Private / internal methods** (prefixed with `_`, not part of the supported API)


| Method                                                              | Role                                                                                                                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `__post_init__()`                                                   | Dataclass hook — validates `query` and fills in a default `perception`.                                                                                            |
| `_search_with_alignment(query, db, min_matches)`                    | Core alignment loop: enumerate candidate function mappings via `FunctionMapping`, run `Alignment.align`, score, early-exit. Returns `(MatchResult, SolutionInfo)`. |
| `_screen_one_mol(mol, *, conformations, min_matches, keep, metric)` | Iterates conformers of one molecule, calls `_search_with_alignment` per conformer, applies the `keep`/`metric` selection.                                          |


**Auto-generated by `@dataclass`**


| Member     | Behaviour                                                    |
| ---------- | ------------------------------------------------------------ |
| `__init__` | Built from the six fields above (positional or keyword).     |
| `__eq__`   | Structural equality across the six fields.                   |
| `__repr__` | `PharmacophoreSearch(query=…, perception=…, epsilon=0.5, …)` |


**Module-level helpers used alongside `PharmacophoreSearch`** (re-exported by `from pypharao import *`)


| Name                                          | Purpose                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `MatchResult`                                 | Dataclass returned by `screen()`. See [§5 *Analysing the results](#5-analysing-the-results)*. |
| `sort_match_results(hits, sort=..., key=...)` | Sort hits by any metric.                                                                      |
| `print_match_results(hits, limit=...)`        | Pretty-print hits as a table.                                                                 |
| `count_matchable_query_points(query)`         | Resolved value when `min_matches=0`; counts query points other than `EXCL` / `UNDEF`.         |


### `screen()` parameters

```python
hits = searcher.screen(
    mols,                       # one Chem.Mol or a list of mols / tuples
    conformations="all",        # 'all' | 'single' | int N
    min_matches=0,              # 0 = auto, resolved to matchable query points
    keep="best",                # 'best' (one row per molecule) or 'all'
    metric="tanimoto",          # tie-break for keep='best'
    n_jobs=0,                   # 0 = all CPUs, 1 = sequential
    progress=True,              # tqdm progress bar over the molecule list
)
```


| Argument        | Default      | Description                                                                                                                                                                                                                                                                                  |
| --------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mols`          | —            | A single `Chem.Mol`, a list of `Chem.Mol`, or a list of tuples whose last element is a mol (the first element becomes the reported hit index, e.g. `(line_idx, smiles, mol)`).                                                                                                               |
| `conformations` | `"all"`      | `"all"` iterates every conformer of each molecule; `"single"` uses only the first; a positive `int N` uses the first N.                                                                                                                                                                      |
| `min_matches`   | `0`          | Minimum number of query points that must map to a molecule feature for a hit. A value of `0` (the default) is auto-resolved to `count_matchable_query_points(query)` (every point that is not `EXCL` / `UNDEF`). Negative values or values exceeding the matchable count raise `ValueError`. |
| `keep`          | `"best"`     | `"best"` keeps the single highest-scoring conformer per molecule; `"all"` keeps every conformer that satisfies `min_matches` (each result records the matching `conf_id`).                                                                                                                   |
| `metric`        | `"tanimoto"` | Tie-breaker for `keep="best"`. One of `tanimoto`, `overlap_volume`, `excl_volume`, `tversky_ref`, `tversky_db`. Maximised, except `excl_volume` which is minimised.                                                                                                                          |
| `n_jobs`        | `0`          | Worker processes; `0` uses all CPUs, `1` runs sequentially.                                                                                                                                                                                                                                  |
| `progress`      | `True`       | Show a tqdm progress bar (requires `tqdm`).                                                                                                                                                                                                                                                  |


`screen()` **always** returns `list[tuple[int, MatchResult]]`. A single molecule with no match returns `[]`; a successful match returns `[(index, MatchResult)]`. With `keep="all"` a single molecule may produce several rows (one per matching conformer), and the same `index` is repeated.

## 5. Analysing the results

### `MatchResult` attributes


| Attribute                  | Type                    | Description                                                                                           |
| -------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `conf_id`                  | `int`                   | Conformer id of the input molecule that produced this match (`0` for single-conformer mols).          |
| `ref_volume`               | `float`                 | Self-overlap volume of the query pharmacophore.                                                       |
| `db_volume`                | `float`                 | Self-overlap volume of the molecule pharmacophore.                                                    |
| `overlap_volume`           | `float`                 | Raw overlap volume between matched (query, molecule) pairs before subtracting exclusion overlap.      |
| `excl_volume`              | `float`                 | Overlap volume between query `EXCL` points and molecule features (subtracted from the aligned score). |
| `tanimoto`                 | `float`                 | `aligned_overlap / (ref_volume + db_volume − aligned_overlap)`.                                       |
| `tversky_ref`              | `float`                 | `aligned_overlap / ref_volume`.                                                                       |
| `tversky_db`               | `float`                 | `aligned_overlap / db_volume`.                                                                        |
| `mapping`                  | `list[tuple[int, int]]` | Matched feature pairs as `(query_index, molecule_index)` into the original point lists.               |
| `database_pharmacophore`   | `MoleculePharmacophore` | Molecule pharmacophore perceived for this conformer (before alignment).                               |
| `matched_db_pharmacophore` | `MoleculePharmacophore` | Subset of molecule features that appear in `mapping`, transformed into the query frame.               |
| `aligned_mol`              | `Chem.Mol | None`       | Copy of the input molecule with the matching conformer transformed into the query frame.              |


### Sorting and printing

```python
hits = searcher.screen(mols)
ranked = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(ranked, limit=20)
```

`print_match_results(results, *, limit=None, file=None)` prints a tab-separated table — header row first, then one row per hit. `No hits.` is printed for empty lists. `file` is forwarded to `print(file=...)`; pass an open text stream to write the table to disk.

`sort_match_results(matches, *, sort, key)` returns a new sorted list; `key` is any numeric attribute on `MatchResult`.

### Writing hits to an SDF or PDB file

```python
n = write_hits_sdf(ranked[:50], "top_hits.sdf")
n = write_hits_sdf(ranked[:50], "top_hits.sdf", pharmacophore=query)

n = write_hits_pdb(ranked[:50], "top_hits.pdb")
n = write_hits_pdb(ranked[:50], "top_hits.pdb", pharmacophore=query)
```

`write_hits_sdf` writes each aligned hit (one conformer) as a record with SDF tags `index`, `conf_id`, `tanimoto`, `tversky_ref`, `tversky_db`, `overlap_volume`, `excl_volume`, `ref_volume`, `db_volume`. `write_hits_pdb` writes each hit as a separate `MODEL` block in a single PDB file, terminated by an `END`.

When `pharmacophore=` is supplied, the pharmacophore is rendered (one pseudo-atom per feature) and prepended as the **first** record of the output file. Aligned hit molecules live in the query coordinate frame, so passing the query pharmacophore here produces a single SDF / PDB that overlays the query on top of the aligned hits.

The pseudo-atom convention (configurable via `pharmacophore_to_mol`):


| `PointType`     | Element | PDB residue name |
| --------------- | ------- | ---------------- |
| `AROM`          | C       | `ARO`            |
| `LIPO`          | C       | `LIP`            |
| `AROM_OR_LIPO`  | C       | `AOL`            |
| `HDON`          | N       | `HDO`            |
| `HACC`          | O       | `HAC`            |
| `HACC_AND_HDON` | S       | `DAC`            |
| `HACC_OR_HDON`  | S       | `DAO`            |
| `POSC`          | Na      | `POS`            |
| `NEGC`          | Cl      | `NEG`            |
| `EXCL`          | F       | `EXC`            |
| `UNDEF`         | He      | `UND`            |


Both writers skip hits with no aligned molecule (one-line warning to `stderr`) and return the total number of records / `MODEL`s written, including the pharmacophore record when one was passed in.

`pharmacophore_to_mol(ph, *, name="pharmacophore")` is also exported in case you want to embed a pharmacophore in your own RDKit pipelines (it returns a `Chem.Mol` with one disconnected pseudo-atom per feature, plus per-mol SDF properties `kind`, `name`, `num_features`, `types`, `sigmas`, `centers`, `normals`).

## License

LGPL-3.0-or-later (see `LICENSE`).