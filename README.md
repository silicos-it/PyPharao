# PyPharao

This is Python implementation for **3D pharmacophore** representation and screening using **Gaussian volume alignment**. This implementation matches the original [Pharao](https://github.com/silicos-it/pharao) C++ implementation, with perception from 3D molecules via the [RDKit toolkit](https://www.rdkit.org).

If this work is used in any project, then please include the following citation in your publications: 

Taminau, J.; Thijs, G.; De Winter, H. (2008) Pharao: Pharmacophore alignment and optimization, *J. Mol. Graph. Model.* **27**, 161-169.


---

## Table of contents

- [Table of contents](#table-of-contents)
- [1. Installing PyPharao](#1-installing-pypharao)
- [2. Introduction](#2-introduction)
  - [2.1 The general flow of a pharmacophore screening experiment](#21-the-general-flow-of-a-pharmacophore-screening-experiment)
  - [2.2 The concept of pharmacophores and pharmacophore points](#22-the-concept-of-pharmacophores-and-pharmacophore-points)
    - [2.2.1 What is a pharmacophore?](#221-what-is-a-pharmacophore)
    - [2.2.2 What is a pharmacophore point?](#222-what-is-a-pharmacophore-point)
- [3. Working with pharmacophores](#3-working-with-pharmacophores)
  - [3.1 Pharmacophore perception](#31-pharmacophore-perception)
  - [3.2 Creating pharmacophores](#32-creating-pharmacophores)
  - [3.3 Editing query pharmacophores](#33-editing-query-pharmacophores)
  - [3.4 Copying query pharmacophores](#34-copying-query-pharmacophores)
  - [3.5 Input and output](#35-input-and-output)
  - [3.6 Working with excluded volumes](#36-working-with-excluded-volumes)
  - [3.7 Clearing and removing points](#37-clearing-and-removing-points)
- [4. Executing a pharmacophore screen](#4-executing-a-pharmacophore-screen)
  - [4.1 The PharmacophoreSearch class](#41-the-pharmacophoresearch-class)
  - [4.2 MatchResult, sorting, and hit I/O](#42-matchresult-sorting-and-hit-io)
- [5. API summary](#5-api-summary)
- [6. License](#6-license)



---

## 1. Installing PyPharao

The installable library is `src/pypharao/` only. The `examples/` and `tests/` directories stay in the repository and are **not** shipped in the package.

Runtime dependencies (installed automatically with PyPharao):

| Component | Required for                                    |
| --------- | ----------------------------------------------- |
| **NumPy** | Core library                                    |
| **RDKit** | Perception, molecule I/O, screening             |
| **tqdm**  | Progress bars when `screen(..., progress=True)` |


### pip

**From a git checkout** (installs `src/pypharao` into your environment):

```bash
git clone https://github.com/silicos-it/PyPharao.git
cd PyPharao
pip install .
```

**Directly from GitHub** (no clone required; `git` must be available locally):

```bash
pip install git+https://github.com/silicos-it/PyPharao.git
```

If `pip install rdkit` is unavailable on your platform, install RDKit from conda-forge first, then install PyPharao (conda’s RDKit satisfies the requirement):

```bash
conda install -c conda-forge rdkit numpy tqdm
pip install .
```

**Update an existing pip install**

```bash
pip install -U .                 # from a checkout (after git pull)
```

### conda

**Environment from a checkout**:

```bash
git clone https://github.com/silicos-it/PyPharao.git
cd PyPharao
conda env create -f environment.yml
conda activate pypharao
```

**Build and install a local conda package** (see `conda-recipe/`):

```bash
conda build conda-recipe
conda install pypharao --use-local
```

**Update:**

```bash
conda activate pypharao
git pull
pip install -U .                         # refresh library from checkout
# or, after rebuilding the recipe:
conda install pypharao --use-local
```

**Contributors / editable install with tests:**

```bash
conda env create -f environment-dev.yml
conda activate pypharao-dev
pytest
```

### uv

**From a git checkout:**

```bash
git clone https://github.com/silicos-it/PyPharao.git
cd PyPharao
uv pip install .
```

**Update:**

```bash
uv pip install -U .             # from a checkout (after git pull)
```

### Importing PyPharao

After installation, the library can be used from within your `Python` scripts. The whole public API lives at the top of the package, so a single import is enough:

```python
from pypharao import *
```


---

## 2. Introduction

### 2.1 The general flow of a pharmacophore screening experiment

A typical **Pharao** screen has three conceptual steps: obtain a **query pharmacophore**, prepare a **3D compound collection**, and **match** each compound (per conformer) to the query using Gaussian volume overlap after alignment:

##### Step 1. Build a query pharmacophore

A **query pharmacophore** can be created in any of these ways:

| Source                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------ |
| [From the 3D coordinates of a single small molecule (RDKit)](#from-a-3d-molecule-requires-rdkit)                   |
| [From a protein structure plus optional exclusion geometry (RDKit, PDB)](#from-a-protein-structure-requires-rdkit) |
| [Manual construction in the Python code](#manually)                                                                |
| [By loading an existing file (JSON or Pharao `.phar` format)](#from-json-or-pharao-phar-files)                     |


##### Step 2. Prepare the molecule database

Each screening candidate must be an RDKit `Chem.Mol` molecule with at least one 3D conformer (embedding and force fields are your responsibility).


##### Step 3. Run the screen

A pharmacophore screen is performed by constructing a `PharmacophoreSearch(query)` object (optionally pass `MoleculePharmacophorePerception` to control which feature types are perceived on database molecules), followed by calling the `screen(...)` method on the prepared molecule list. The search enumerates **feature mappings** between the query and molecule pharmacophores, **aligns** the molecule pharmacophore into the query frame, and **scores** the match using a **Gaussian volume overlap**.

In the following sections all these steps are explained in more detail.


### 2.2 The concept of pharmacophores and pharmacophore points

#### 2.2.1 What is a pharmacophore?

A **pharmacophore** is an ordered collection of **pharmacophore points**: abstract sites that specify chemical functionalities in a three-dimensional contexts. Within **PyPharao**, these **pharmacophore points** are called `PharmacophorePoint`s.

**PyPharao** has two concrete subclasses of the base `Pharmacophore` class:


| Class                   | Role           | Allowed `PointType`s                |
| ----------------------- | ---------------------------------------------------------------- | ----------------------------------- |
| `QueryPharmacophore`    | The search template                                              | Every `PointType`, including `EXCL`, `UNDEF`, and query-only OR types (`AROM_OR_LIPO`, `HACC_OR_HDON`, `HACC_AND_HDON`) |
| `MoleculePharmacophore` | Built automatically from each database molecule during screening | `AROM`, `LIPO`, `HDON`, `HACC`, `HACC_AND_HDON`, `POSC`, `NEGC` only   |


#### 2.2.2 Pharmacophore points

A `PharmacophorePoint` is a **Gaussian feature** site. Conceptually it consists of a **type**, **location**, **width**, and optionally a **normal** (or direction):

| Concept      | Implementation                                                            | Exposed as                  |
| ------------ | ------------------------------------------------------------------------- |---------------------------- |
| **Type**     | `PointType`                                                               | `PharmacophorePoint.Type`   |
| **Location** | `center` → `(x, y, z)` in ångström                                        | `PharmacophorePoint.center` | 
| **Width**    | `sigma` (Gaussian width, ångström)                                        | `PharmacophorePoint.sigma`  |
| **Normal**   | `normal` → `(nx, ny, nz)` for types that carry a plane; `None` otherwise. | `PharmacophorePoint.normal` |      


**`PointType`s:**

Each `PharmacophorePoint` contains a given **type**. There are 11 types in total:

| Type            | Default `sigma` | Normal | Description                                               |
| --------------- | --------------- | ------ | --------------------------------------------------------- |
| `AROM`          | 0.7             | yes    | Aromatic ring centroids with plane normals                |
| `LIPO`          | 0.7             | no     | Lipophilic regions (from molecular surface), no aromatics |
| `AROM_OR_LIPO`  | 0.7             | yes    | Either an `AROM` or a `LIPO` group, or both               |
| `HDON`          | 1.0             | no     | H-bond donors (N/O with H, not negatively charged)        |
| `HACC`          | 1.0             | no     | H-bond acceptors (N/O)                                    |
| `HACC_AND_HDON` | 1.0             | no     | Both a `HACC` and a `HDON` at the same site               |
| `HACC_OR_HDON`  | 1.0             | no     | Either an `HACC` or a `HDON` group                        |
| `POSC`          | 1.0             | no     | Positively charged atoms                                  |
| `NEGC`          | 1.0             | no     | Negatively charged atoms                                  |
| `EXCL`          | 1.6             | no     | Exclusion sphere                                          |
| `UNDEF`         | 1.0             | no     | Undefined placeholder (matches any molecule feature type) |

Note: `PointType` enum members and their underlying `.value` strings both use the underscored spellings (`PointType.AROM_OR_LIPO`, `PointType.HACC_AND_HDON`, `PointType.HACC_OR_HDON`).

Defaults are available as `DEFAULT_SIGMA[PointType.X]` and `TYPE_HAS_NORMAL[PointType.X]`.

There is a difference between `QueryPharmacophore` and `MoleculePharmacophore` pharmacophores with respect to the types of pharmacophore point they can be composed of:

| Type            | Can be present in a `QueryPharmacophore`? | Can be present in a `MoleculePharmacophore`? |
| --------------- | ----------------------------------------- | -------------------------------------------- |
| `AROM`          | yes                                       | yes                                          | 
| `LIPO`          | yes                                       | yes                                          | 
| `AROM_OR_LIPO`  | yes                                       | no                                           | 
| `HDON`          | yes                                       | yes                                          | 
| `HACC`          | yes                                       | yes                                          | 
| `HACC_AND_HDON` | yes                                       | yes                                          | 
| `HACC_OR_HDON`  | yes                                       | no                                           | 
| `POSC`          | yes                                       | yes                                          | 
| `NEGC`          | yes                                       | yes                                          | 
| `EXCL`          | yes                                       | no                                           | 
| `UNDEF`         | yes                                       | no                                           | 


**Which query types can match which molecule types?**

There are rules that define which types from both a `QueryPharmacophore` and a `MoleculePharmacophore` may *match*. The following table summarizes these directional rules (the query point decides compatibility; molecule `EXCL` does not exist):

| Query type      | May map to molecule type(s)                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| `AROM`          | `AROM`                                                                         |
| `LIPO`          | `LIPO`                                                                         |
| `AROM_OR_LIPO`  | `AROM` or `LIPO`                                                               |
| `HDON`          | `HDON`                                                                         |
| `HACC`          | `HACC`                                                                         |
| `HACC_AND_HDON` | `HACC_AND_HDON` only                                                           |
| `HACC_OR_HDON`  | `HDON` or `HACC`                                                               |
| `POSC`          | `POSC`                                                                         |
| `NEGC`          | `NEGC`                                                                         |
| `UNDEF`         | Any molecule feature type in the table above                                   |
| `EXCL`          | Never part of the feature mapping (overlap is handled separately as a penalty) |


**How a match is computed (high level):**

1. **Mapping**: a `functionMapping` walks candidate assignments of query features (all except `EXCL`) to molecule features that are 
    **type-compatible** and pass a pairwise **volume-overlap gate** controlled by `epsilon` on the searcher.
2. **Alignment**: for promising maps, `alignment` finds a rigid transform that improves the **Gaussian volume overlap** between 
    paired features (optionally using **direction vectors** for types with normals when `use_direction=True`).
3. **Scoring**: the aligned overlap volume yields **Tanimoto**, **Tversky**, and related metrics; query `EXCL` points contribute 
    a **soft overlap penalty** (`with_exclusion`) and optionally a **hard heavy-atom clash filter** after alignment (`excl_hard_filter`).


---

## 3. Working with pharmacophores

### 3.1 Pharmacophore perception

A `PharmacophorePerception` instance controls **which feature types** are allowed when a pharmacophore is built from a 3D molecule. 
The `PharmacophorePerception` base class is **not** meant to be instantiated directly, instead the appropriate derived classes should be used:

| Class                             | Used when                                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `QueryPharmacophorePerception`    | Building a `QueryPharmacophore` via the `query_pharmacophore_from_molecule` function                            |
| `MoleculePharmacophorePerception` | Perceiving **database** molecules inside a `PharmacophoreSearch.screen` (constructor argument `perception=...`) |

Both subclasses cover the same **seven** auto-perceivable types and default to *all enabled*: `AROM`, `LIPO`, `HDON`, `HACC`, `HACC_AND_HDON`, `POSC`, `NEGC`

`AROM` and `LIPO` are **mutually exclusive** at perception time: an aromatic ring is reported only as `AROM`, while a lipophilic moiety 
that is *not* aromatic is reported as `LIPO`.

`EXCL`, `UNDEF`, `AROM_OR_LIPO`, and `HACC_OR_HDON` are **never** auto-perceived with a `PharmacophorePerception` object.
Nevertheless, these can be added by editing the query manually (for example convert `AROM`/`LIPO` to `AROM_OR_LIPO`, 
or use `add_excluded_volume` — see [§3.6](#36-working-with-excluded-volumes)).


**Construction of pharmacophore perception instances:**

```python
from pypharao import *

perception = QueryPharmacophorePerception()
perception = QueryPharmacophorePerception(LIPO=False, HDON=True)  # kwargs override defaults
```

When you create a perception object with keyword flags, the keyword names must correspond to PointType names, e.g. `AROM`, `LIPO`, `HACC_AND_HDON`.
You can pass them as plain strings, and the code converts (“coerces”) them to the enum values internally.


**Public API (`PharmacophorePerception` and subclasses):**


| Method / member                | Returns           | Default / notes                                                         |
| ------------------------------ | ----------------- | ----------------------------------------------------------------------- |
| `__init__(**flags: bool)`      | `None`            | Each allowed type defaults to `True`; kwargs set initial on/off         |
| `is_enabled(t)`                | `bool`            | `t` may be `PointType` or string                                        |
| `enable(t)`                    | `None`            |                                                                         |
| `disable(t)`                   | `None`            |                                                                         |
| `types_enabled()`              | `list[PointType]` | Types currently on, in `PointType` enum order                           |
| `__iter__()`                   | iterator          | Allowed types for this subclass (declaration order)                     |
| `print_features(*, file=None)` | `None`            | One line per type: `TYPE status description`; `file` defaults to stdout |
| `__repr__()`                   | `str`             | Shows flag values                                                       |


**Examples:**

```python
from pypharao import *

perception = QueryPharmacophorePerception()
perception.print_features()                  # one line per feature type
perception.is_enabled(PointType.LIPO)        # True
perception.disable(PointType.LIPO)
perception.enable("HACC_AND_HDON")
perception.types_enabled()                   # list of currently enabled types
for t in perception:                         # iterate over allowed types
    ...
```

When the `query_pharmacophore_from_molecule` function (or the internal molecule perception used by `screen()`) is called with no `perception` argument, the relevant default `QueryPharmacophorePerception()` or `MoleculePharmacophorePerception()` is used.


### 3.2 Creating pharmacophores

#### 3.2.1 Query pharmacophores

##### Manually creating a query pharmacophore

Manually creating pharmacophores can be done with the `QueryPharmacophore()` constructor, followed by the `.add_point()` method:

```python
from pypharao import *

q = QueryPharmacophore(name="manual")
q.add_point(PharmacophorePoint(type=PointType.AROM_OR_LIPO, center=(0, 0, 0), normal=(0, 0, 1)))
q.add_point(PharmacophorePoint(type=PointType.HACC_AND_HDON, center=(3, 0, 0)))
q.add_point(PharmacophorePoint(type=PointType.EXCL, center=(0, 0, 2.5)))
```

##### Creating a query pharmacophore from JSON or Pharao `.phar` files

There are a number of ** class methods* that create and return an instance of a pharmacophore object from a JSON or Pharao `.phar` files:

`QueryPharmacophore.from_json_dict()`
`QueryPharmacophore.from_json()`
`QueryPharmacophore.read_json()`
`QueryPharmacophore.read_phar()`
`QueryPharmacophore.from_phar_text()`

These class methods create and return a new pharmacophore object. They do not modify an existing instance in place.

So this is wrong (result discarded):

```python
from pypharao import *

q = QueryPharmacophore()
q.read_json("query.json")   # return value ignored -> q is still empty
```

Instead one should assign the returned pharmacophore object:

```python
q = QueryPharmacophore.read_json("query.json")
q = QueryPharmacophore.from_json(json_text)
```

When you load from JSON using the base `Pharmacophore` loader, it looks at the JSON field "kind" to decide which concrete object to create: if "kind" says "query" (or is missing), you get a `QueryPharmacophore`; if it says "molecule", you get a `MoleculePharmacophore`.
The `Pharmacophore.from_phar_text()` method uses `QueryPharmacophore` as default target when called on base class, and sets the first line (header) as the query name.


##### Creating a query pharmacophore from a 3D molecule

In order to create a pharmacophore from the three-dimensional structure of a molecule, the following **standalone functions** are available:

`query_pharmacophore_from_molecule()`
`molecule_pharmacophore_from_molecule()`


```python
from rdkit import Chem
from rdkit.Chem import AllChem
from pypharao import *

mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
AllChem.UFFOptimizeMolecule(mol)

q = query_pharmacophore_from_molecule(mol, name="phenol")
```

`query_pharmacophore_from_molecule()` returns a `QueryPharmacophore`. Compound query types (`AROM_OR_LIPO`, `HACC_OR_HDON`) are **not** auto-perceived — refine them by hand if desired. `HACC_AND_HDON` is created automatically whenever an `HDON` and an `HACC` sit at the same atom (the two originals are removed).

Optional arguments are (see source docstrings): `perception`, `conf_id` (default `0`), and `name`.


##### Creating a query pharmacophore from a protein structure

In order to create a query pharmacophore from the three-dimensional structure of a protein structure, the following **standalone function** is available:

`query_pharmacophore_from_protein()`


```python
from pypharao import *

q = query_pharmacophore_from_protein(
    "binding_site.pdb",
    "excluded_atoms.pdb",
    min_distance_between_excl_points=1.5,
)
```

This function uses the same perception as `query_pharmacophore_from_molecule()` on the protein PDB, then places `EXCL` points at each atom of the exclusion PDB and 
thins them so pairwise distances are at least `min_distance_between_excl_points` Å. See `query_pharmacophore_from_protein` in `rdkit_perception.py` for `perception`, `conf_id`, `excl_conf_id`, `name`, and `excl_sigma`.


#### 3.2.2 Molecule pharmacophores

`MoleculePharmacophore` instances are **not** normally authored by hand for screening: `PharmacophoreSearch` builds one **per database molecule** (and **per conformer**) via `molecule_pharmacophore_from_molecule()`, using `MoleculePharmacophorePerception` to decide which of the seven types may appear.


### 3.3 Editing query pharmacophores

Editing a query pharmacophore use the `Pharmacophore` API (same for `MoleculePharmacophore`, within `allowed_types`):


**Public editing API:**

| Method         | Signature                               | Returns | Notes                                                                                                                             |
| -------------- | --------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `add_point`    | `(point: PharmacophorePoint)`           | `None`  | Append after type validation                                                                                                      |
| `set_point`    | `(idx: int, point: PharmacophorePoint)` | `None`  | Replace index `idx`                                                                                                          |
| `update_point` | `(idx: int, **changes)`                 | `None`  | Shorthand for `set_point(idx, self[idx].replace(**changes))`; accepts `type=`, `center=`, `sigma=`, `normal=` |
| `remove_point` | `(idx: int)`                            | `None`  | Delete; later indices shift down                                                                                                  |
| `clear`        | `()`                                    | `None`  | Remove every point                                                                                                                |


**Iteration and access:**

| Member              | Behaviour                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------- |
| `points` (property) | Returns a **snapshot** `list[PharmacophorePoint]`; mutating that list does **not** change the pharmacophore |
| `len(ph)`           | Number of points                                                                                            |
| `for p in ph: ...`  | Iterate in order                                                                                            |
| `ph[i]`             | Index access                                                                                                |


**`QueryPharmacophore` name API:**

| Member                | Returns                  |
| --------------------- | ------------------------ |
| `get_name()`          | `str`                    |
| `set_name(name: str)` | `None`                   |
| `name`                | Property (get/set `str`) |


**Examples:** (`q` is a `QueryPharmacophore`):

```python
from pypharao import PharmacophorePoint, PointType, QueryPharmacophore

q = QueryPharmacophore(name="demo")

\# add_point — full PharmacophorePoint (AROM needs a plane normal as absolute tip coords)
q.add_point(
    PharmacophorePoint(
        type=PointType.AROM,
        center=(10.0, 0.0, 0.0),
        normal=(11.0, 0.0, 0.0),
        sigma=0.7,
    )
)
q.add_point(
    PharmacophorePoint(
        type=PointType.EXCL,
        center=(5.0, 5.0, 5.0),
        sigma=1.6,
    )
)
```

```python
\# update_point — tweak fields without building a replacement by hand
q.update_point(0, sigma=1.0)
q.update_point(0, center=(10.1, 0.05, -0.02))

for i, p in enumerate(q):
    if p.type == PointType.AROM:
        q.update_point(i, type=PointType.AROM_OR_LIPO)  # normal kept by replace()
        break
```

```python
\# set_point — replace slot i with another immutable point wholesale
old = q[1]
replacement = PharmacophorePoint(type=PointType.HACC, center=old.center, sigma=1.05)
q.set_point(1, replacement)
```

```python
q.remove_point(0)  # indices after this slot shift down
q.clear()
```

```python
len(q)                     # 2 (after adding points)
for p in q: ...           # iterate
q[0]                      # index access
q.set_name("query_v2")
q.get_name()
```

For `MoleculePharmacophore`, the same methods apply, but when a `PointType` is not allowed for that subclass (for example `EXCL`), then a `ValueError` is raised.

**Note on copying:** `pharmacophore_2 = pharmacophore_1` only binds a second name to the *same* object; later edits affect both names. 
Use `pharmacophore_2 = pharmacophore_1.copy()` whenever you need an independent pharmacophore to modify (for example to derive an 
`AROM` → `AROM_OR_LIPO` variant). The copy is shallow: it produces a new `Pharmacophore` whose `_points` list is new, 
but the `PharmacophorePoint` objects themselves are shared. Since `PharmacophorePoint` is immutable (`point.replace(...)` 
returns a new instance), this is safe in practice.



### 3.4 Copying query pharmacophores


| Method   |Returns                                                                         |
| -------- | ------------------------------------------------------------------------------ |
| `copy()` | Shallow copy preserving the subclass and (for `QueryPharmacophore`) the name.  |


The internal points list of the new pharmacophore is new; `PharmacophorePoint` instances are shared. This is safe because they are immutable.

```python
q2 = q.copy()
```


### 3.5 Input and output

**JSON**

| Method                       | Direction | Returns / side effect                     |
| -----------------------------| ----------| ------------------------------------------|
| `to_json_dict()`             | export    | `dict[str, Any]`                          |
| `to_json(**kwargs)`          | export    | `str` (`**kwargs` forwarded to `json.dumps`, default `indent=2`, use `indent=None` for one line) |
| `write_json(path, **kwargs)` | export    | writes UTF-8 file                         |
| `from_json_dict(d)`          | import    | **new** pharmacophore                     |
| `from_json(s)`               | import    | **new** pharmacophore                     |
| `read_json(path)`            | import    | **new** pharmacophore                     |


**Pharao `.phar` text**

| Method                                 | Direction | Returns / side effect                    |
| -------------------------------------- | --------- | ---------------------------------------- |
| `to_phar_text()`                       | export    | `str` (name line, point lines, `$$$$`)   |
| `write_phar(path)`                     | export    | writes UTF-8 file                        |
| `from_phar_text(text)` *(classmethod)* | import    | **new** pharmacophore                    |
| `read_phar(path)` *(classmethod)*      | import    | **new** pharmacophore from UTF-8 `.phar` |


Import helpers are **classmethods** — they **return a new** `Pharmacophore`; always assign, e.g. `q = Pharmacophore.read_json(path)`.


**SDF / PDB export)**

| Method                          | Notes                                                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `write_sdf(path, *, name=None)` | Single-record SDF; one pseudo-atom per feature; properties `kind`, `name`, `num_features`, `types`, `sigmas`, `centers`, `normals` |
| `write_pdb(path, *, name=None)` | PDB with one `HETATM` per feature; residue encodes type                                                                            |


Both methods reuse `pharmacophore_to_mol`, so the layout matches the pharmacophore records written by `write_hits_sdf` / `write_hits_pdb` when you pass `pharmacophore=` (see [§4.2](#42-matchresult-sorting-and-hit-io)).

```python
q.write_json("out.json")
q.write_phar("out.phar")
\# With RDKit:
q.write_sdf("query.sdf")
q.write_pdb("query.pdb")
```


---

### 3.6 Working with excluded volumes

`EXCL` points are never auto-perceived from molecules. Two common workflows exist:

1. **Protein / atom list** — use `query_pharmacophore_from_protein()`
2. **Shape envelope around ligand(s)** — use `add_excluded_volume()`

The `add_excluded_volume()` function paints a *shape-complementarity envelope* around one or more reference 3D structures by appending `EXCL` features to an existing pharmacophore:

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
\# Also allowed: add_excluded_volume([mol_a, mol_b, ...], q, ...) with ligands
\# superimposed to the same frame; pass conf_id=0 (or any int) to use only that
\# conformer index in each molecule.
```


| Argument            | Default                         | Description                                                                               |
| ------------------- | ------------------------------- | ----------------------------------------------------------------------------------------- |
| `mol`               | —                               | One 3D RDKit molecule **or** a non-empty iterable of molecules (heavy atoms only).        |
| `pharmacophore`     | —                               | Target; must allow `**EXCL`** (typically `QueryPharmacophore`).                           |
| `conf_id`           | `None`                          | `None`: union includes **all** conformers. `int`: only that conformer index per molecule. |
| `sigma`             | `DEFAULT_SIGMA[PointType.EXCL]` | Gaussian width (Å) of each new `EXCL` (default 1.6).                                      |
| `shell_inner`       | 1.0                             | Inner distance (Å) from nearest heavy-atom vdW surface (>= 0).                            |
| `shell_outer`       | 3.0                             | Outer distance (Å) (`> shell_inner`).                                                     |
| `spacing`           | 1.5                             | Grid step (Å); also minimum separation between output centres after thinning.             |
| `feature_clearance` | 0.0                             | If > 0, drop candidates within this distance (Å) of an existing feature centre.           |
| `max_excl`          | 0                               | Cap on number of `EXCL` markers (0 = no limit).                                           |


Internally it lays a regular 3D grid over the bounding box of all heavy-atom centres. Every heavy atom from every selected molecule and conformer 
contributes to **one artificial molecule**: at each vertex the vdW surface distance is the minimum over **all** those atoms. Vertices in 
`[shell_inner, shell_outer]` become candidates. After optional `feature_clearance` filtering, candidates are **thinned** (see below). Hydrogens are ignored.

**How markers are chosen (`spacing` and `max_excl`). ** Selection is **deterministic**. Candidate grid vertices are sorted by 
**increasing union vdW surface distance** (closer to the combined surface first). The implementation scans that order and **keeps** a vertex 
only if its centre is at least `spacing` Å from **every** centre already kept. `max_excl = 0` means **no** upper bound; a positive `max_excl`
stops once that many centres have been kept.

In Pharao scoring `EXCL` query points contribute two layers:

1. **Soft volume penalty** (`with_exclusion=True`): overlap between each `EXCL` point and the candidate's pharmacophore features is subtracted from the aligned overlap volume (`MatchResult.excl_volume`).
2. **Hard atom-clash filter** (`excl_hard_filter=True`, default): after alignment, every heavy atom of the aligned database molecule is a sphere of its van der Waals radius. The hit is rejected when `dist(atom_center, EXCL_center) < vdW(atom) + excl_clash_radius` for any pair. Default `excl_clash_radius=0.0`. Set `excl_hard_filter=False` for soft penalty only. Independent of `EXCL` Gaussian `sigma`.

**Purging distant exclusion spheres** (for `QueryPharmacophore` only):

| Method                        | Signature                 | Returns                                 |
| ----------------------------- | ------------------------- | --------------------------------------- |
| `purge_exclusion_spheres`     | `(distance: float = 8.0)` | `int` — number of `EXCL` points removed |


Removes each `EXCL` whose centre is farther than `distance` Å from **every** non-`EXCL` point. If there are no non-`EXCL` points, all `EXCL` are removed. `distance` must be >= 0.

```python
n_removed = query.purge_exclusion_spheres(distance=8.0)
```

`add_excluded_volume` expects a pharmacophore that allows `EXCL`; passing a `MoleculePharmacophore` raises `ValueError`.

See `examples/03-excluded-volumes.py` and `examples/04-excluded-volumes-from-multiple-ligands.py`.


### 3.7 Clearing and removing points

There is no separate “delete file” API for a pharmacophore object. To empty or shrink a `QueryPharmacophore`, use the following methods:

| Action            | Method                  |
| ----------------- | ----------------------- |
| Remove one point  | `remove_point(idx)`     |
| Remove all points | `clear()`               |


To remove the Python object, simply drop all references (e.g. `del q`); persistent copies may still exist if referenced elsewhere.


---

## 4. Executing a pharmacophore screen

Prerequisites: a `QueryPharmacophore`, RDKit `Chem.Mol` objects with 3D coordinates, and optionally a `MoleculePharmacophorePerception`.

```python
searcher = PharmacophoreSearch(query)                    # default molecule perception
searcher = PharmacophoreSearch(query, perception=opts)   # custom molecule perception
```


### 4.1 The PharmacophoreSearch class

`PharmacophoreSearch` is a `@dataclass`. The constructor accepts the fields below (positional or keyword); fields remain readable and assignable on the instance.

**Fields (= constructor parameters)**


| Name                | Type                                       | Default      | Purpose                                                                             |
| ------------------- | ------------------------------------------ | ------------ | ----------------------------------------------------------------------------------- |
| `query`             | `QueryPharmacophore`                       | *(required)* | Raises `TypeError` if not a `QueryPharmacophore`.                                   |
| `perception`        | `MoleculePharmacophorePerception` | `None` | `None`       | `None` ⇒ `MoleculePharmacophorePerception()` in `__post_init__`.                    |
| `epsilon`           | `float`                                    | 0.5          | Tolerance for `FunctionMapping` when proposing candidate feature pairings.          |
| `use_direction`     | `bool`                                     | `True`       | Use feature normals (e.g. `AROM` / `AROM_OR_LIPO`) in volume scoring and alignment. |
| `with_exclusion`    | `bool`                                     | `True`       | Include `EXCL` in the soft penalty (`excl_volume`).                                 |
| `early_exit_score`  | `float`                                    | 0.98         | Stop exploring further mappings when Tanimoto-like score exceeds this threshold.    |
| `excl_hard_filter`  | `bool`                                     | `True`       | After alignment, reject conformers whose heavy atoms clash with `EXCL` markers.     |
| `excl_clash_radius` | `float`                                    | 0.0          | Padding (Å, >= 0) added to vdW radii in the clash test.                             |


**Public method**


| Method                                                                                                             | Purpose                                                                 |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| `screen(mols, *, conformations='all', min_matches=0, keep='best', metric='tanimoto', n_jobs=0, progress=True)`     | Single public entry point; returns `list[tuple[int, MatchResult]]`.     |


**`screen()` parameters:**

```python
hits = searcher.screen(
    mols,                       # one Chem.Mol or a list of mols / tuples
    conformations="all",        # 'all' | 'single' | int N
    min_matches=0,              # 0 = auto (see below)
    keep="best",                # 'best' (one row per molecule) or 'all'
    metric="tanimoto",          # tie-break for keep='best'
    n_jobs=0,                   # 0 = all CPUs, 1 = sequential
    progress=True,              # tqdm progress bar over the molecule list
)
```


| Argument        | Default      | Description                                                                                                                                                                                                                                                                                     |
| --------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mols`          | —            | A single `Chem.Mol`, a list of `Chem.Mol`, or tuples whose **last** element is a mol (e.g. `(line_idx, smiles, mol)` — the **first** element is the reported index).  |
| `conformations` | `"all"`      | `"all"`: every conformer; `"single"`: first only; `int N`: first **N** conformers.  |
| `min_matches`   | 0            | Minimum number of **matched query features** (excluding `EXCL` pairings from the count). `0` is resolved to `**count_matchable_query_points(query)**`, i.e. the number of query points whose type is *not* `EXCL`. Negative values or values above that count raise `ValueError`. |
| `keep`          | `"best"`     | `"best"`: highest-scoring conformer per molecule; `"all"`: every matching conformer.    |
| `metric`        | `"tanimoto"` | Tie-breaker for `keep="best"`. One of `tanimoto`, `overlap_volume`, `excl_volume`, `tversky_ref`, `tversky_db`. All are **maximised** except `excl_volume` (**minimised**).                                                                                         |
| `n_jobs`        | 0            | Number of CPU-cores to use: 0 means that all CPUs are used.   |
| `progress`      | `True`       | tqdm bar (requires `tqdm`).   |


**Return value:** `screen()` always returns `list[tuple[int, MatchResult]]`. No match for a molecule yields no rows for that index. With `keep="all"`, the same index may repeat (one row per matching conformer).


**Metrics (summary):**


| Metric           | Meaning (after alignment)                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tanimoto`       | aligned_overlap / (ref_volume + db_volume − aligned_overlap) (default for `keep="best"`)                                                             |
| `overlap_volume` | Raw overlap volume of matched pairs (before subtracting exclusion overlap in the sense of `MatchResult.overlap_volume` field — see attributes below) |
| `excl_volume`    | Overlap volume between `EXCL` and molecule features (**smaller is better** for ranking)                                                              |
| `tversky_ref`    | aligned_overlap / ref_volume                                                                                                                         |
| `tversky_db`     | aligned_overlap / db_volume*                                                                                                                         |



### 4.2 MatchResult, sorting, and hit I/O

`MatchResult` is a `@dataclass` with **no** custom instance methods — you work with its **fields**.

| Attribute                  | Type                    | Description                                                                                                |
| -------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `conf_id`                  | `int`                   | Conformer id that produced this match (`0` for single-conformer mols).                                     |
| `ref_volume`               | `float`                 | Self-overlap volume of the query pharmacophore.                                                            |
| `db_volume`                | `float`                 | Self-overlap volume of the molecule pharmacophore.                                                         |
| `overlap_volume`           | `float`                 | Raw overlap volume between matched pairs before subtracting exclusion overlap.                             |
| `excl_volume`              | `float`                 | Overlap volume between query `EXCL` points and molecule features.                                          |
| `tanimoto`                 | `float`                 | Tanimoto on the **aligned overlap** (after `EXCL` subtraction in the score).                               |
| `tversky_ref`              | `float`                 | Tversky with query as reference.                                                                           |
| `tversky_db`               | `float`                 | Tversky with database as reference.                                                                        |
| `mapping`                  | `list[tuple[int, int]]` | `(query_index, molecule_index)` pairs (`EXCL` never appears as a paired query feature in the overlap sum). |
| `database_pharmacophore`   | `MoleculePharmacophore` | Full perceived molecule pharmacophore for this conformer.                                                  |
| `matched_db_pharmacophore` | `MoleculePharmacophore` | Matched subset, transformed into the query frame.                                                          |
| `aligned_mol`              | `Chem.Mol` | `None`     | Copy of the input molecule with the hit conformer transformed into the query frame.                        |


**Sorting and printing:**

| Function                  | Signature                                     | Returns                                        |
| ------------------------- | --------------------------------------------- | ---------------------------------------------- |
| `sort_match_results`  | `(matches, *, sort="descending", key="tanimoto")` | New sorted `list[tuple[int, MatchResult]]`     |
| `print_match_results` | `(results, *, limit=None, file=None)`             | `None` (prints TSV table; `No hits.` if empty) |


```python
hits = searcher.screen(mols)
ranked = sort_match_results(hits, sort="descending", key="tanimoto")
print_match_results(ranked, limit=20)
```

`print_match_results` writes a tab-separated table — header row first, then one row per hit. `limit` truncates display. `file` is forwarded to `print(..., file=...)`.


**Writing hits to SDF or PDB:**

| Function                                              | Returns                          |
| ----------------------------------------------------- | -------------------------------- |
| `write_hits_sdf(hits, path, pharmacophore=None, ...)` | Number of records written        |
| `write_hits_pdb(hits, path, pharmacophore=None, ...)` | Number of `MODEL` blocks written |


```python
n = write_hits_sdf(ranked[:50], "top_hits.sdf")
n = write_hits_sdf(ranked[:50], "top_hits.sdf", pharmacophore=query)

n = write_hits_pdb(ranked[:50], "top_hits.pdb")
n = write_hits_pdb(ranked[:50], "top_hits.pdb", pharmacophore=query)
```

SDF tags include `index`, `conf_id`, `tanimoto`, `tversky_ref`, `tversky_db`, `overlap_volume`, `excl_volume`, `ref_volume`, `db_volume`. PDB output uses separate `MODEL` blocks; file ends with `END`.

When `pharmacophore=` is supplied, that pharmacophore is rendered as the **first** record (pseudo-atoms), overlaying aligned hits in the query frame.


**Pseudo-atom convention** (see also `pharmacophore_to_mol`):

| `PointType`     | Element | PDB residue |
| --------------- | ------- | ----------- |
| `AROM`          | C       | `ARO`       |
| `LIPO`          | C       | `LIP`       |
| `AROM_OR_LIPO`  | C       | `AOL`       |
| `HDON`          | N       | `HDO`       |
| `HACC`          | O       | `HAC`       |
| `HACC_AND_HDON` | S       | `DAC`       |
| `HACC_OR_HDON`  | S       | `DAO`       |
| `POSC`          | Na      | `POS`       |
| `NEGC`          | Cl      | `NEG`       |
| `EXCL`          | F       | `EXC`       |
| `UNDEF`         | He      | `UND`       |



**Module-level helpers used next to `PharmacophoreSearch`:**

| Name                                     | Purpose                                                                            |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `count_matchable_query_points(query)`    | For default `min_matches`: number of query points with `type` not equal to `EXCL`. |
| `matched_query_features(query, mapping)` | Counts query indices in `mapping` excluding `EXCL`.                                |



---

## 5. API summary

Condensed reference for exported names (see `pypharao.__init__`). For details and examples, see the sections above.

### Perception

| Class                                 | Key public API                                                                                                                  | Returns                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `PharmacophorePerception` (base)  | `__init__(**flags)`, `is_enabled`, `enable`, `disable`, `types_enabled`, `__iter__`, `print_features(*, file=None)`, `__repr__` | Mostly `None`; `is_enabled` → `bool`; `types_enabled` → `list[PointType]` |
| `QueryPharmacophorePerception`    | same   | same   |
| `MoleculePharmacophorePerception` | same   | same   |


### `PharmacophorePoint`

| Member                                                        | Returns              |
| ------------------------------------------------------------- | -------------------- |
| `Type` (alias for `PointType`)                                | class                |
| Constructor `(type, center, sigma=None, normal=None)`         | `PharmacophorePoint` |
| `center`                                                      | `tuple(3)`           |
| `normal`                                                      | `tuple(3)`           |
| `has_normal`                                                  | `bool`               |
| `replace(*, type=None, center=None, sigma=None, normal=None)` | `PharmacophorePoint` |


### `Pharmacophore` / `QueryPharmacophore` / `MoleculePharmacophore`

| Member      | Notes    | Returns   |
| ----------- | -------- | --------- |
| `__init__`  | `QueryPharmacophore(points=None, name="")`; others `points=None` | `None`   |
| `points`    | Snapshot list   | `list[PharmacophorePoint]` |
| `__len__`, `__getitem__`, `__iter__`   | Sequence protocol                                          |                            |
| `add_point`, `set_point`, `update_point`, `remove_point`, `clear`       | Validated edits           | `None`                     |
| `copy`                             | Shallow                           | `Pharmacophore`        |
| `to_json_dict`, `to_json`, `write_json`, `from_json_dict`, `from_json`, `read_json`     | JSON I/O                   | `dict`/ `str` / new instance  |
| `to_phar_text`, `write_phar`, `from_phar_text`, `read_phar`                | `.phar` I/O                        | `str` / new instance         |
| `write_sdf`, `write_pdb`                   | RDKit export                               | `None`                     |
| `QueryPharmacophore`: `get_name`, `set_name`, `name`, `purge_exclusion_spheres(distance=8)` |           | `str` / `None` / `int`       |


### `PharmacophoreSearch` and `MatchResult`

| Class                 | Key API                          | Returns                             |
| --------------------- | -------------------------------- | ----------------------------------- |
| `PharmacophoreSearch` | Dataclass fields + `screen(...)` | `list[tuple[int, MatchResult]]` |
| `MatchResult`         | Dataclass fields only            | (container)                         |


### Standalone functions (exported)

| Function                                                                | Returns                       |
| ----------------------------------------------------------------------- | ----------------------------- |
| `query_pharmacophore_from_molecule`, `query_pharmacophore_from_protein` | `QueryPharmacophore`          |
| `molecule_pharmacophore_from_molecule`                                  | `MoleculePharmacophore`       |
| `add_excluded_volume`                                                   | `int` (number added)          |
| `count_matchable_query_points`, `matched_query_features`                | `int`                         |
| `sort_match_results`                                                    | sorted hits                   |
| `print_match_results`                                                   | `None`                        |
| `write_hits_sdf`, `write_hits_pdb`                                      | `int`                         |
| `pharmacophore_to_mol`                                                  | RDKit `Chem.Mol`              |
| `distance`, `cosine_normals`                                            | `float`                       |
| `volume_overlap`                                                        | `float`                       |
| `Alignment`, `position_molecule_coords`, `position_pharmacophore`       | (low-level alignment helpers) |
| `quat_to_rotation_matrix`                                               | `numpy.ndarray`               |


---

## 6. License

LGPL-3.0-or-later (see `LICENSE`).


---

## Appendix — full `PharmacophorePoint` and `Pharmacophore` reference

The following tables collect the longer-form reference from earlier README versions; behavior is unchanged.

### `PharmacophorePoint` (detail)

`PharmacophorePoint` is a `@dataclass(frozen=True)`, so instances are immutable and hashable (usable as dict keys / set members). The constructor validates inputs and raises `ValueError` for inconsistent combinations (e.g. a missing `normal` on a type that requires one, or providing a `normal` for a type that doesn't carry one).

**Fields**

| Name             | Type        | Notes                                                                                                     |
| ---------------- | ----------- | --------------------------------------------------------------------------------------------------------- |
| `type`           | `PointType` | Coerced from `str` at construction if needed.                                                             |
| `x`, `y`, `z`    | `float`     | Centre coordinates, ångström.                                                                             |
| `sigma`          | `float`     | Gaussian width, ångström. Defaults to `DEFAULT_SIGMA[type]` if omitted at construction.                   |
| `nx`, `ny`, `nz` | `float`     | Absolute tip coordinates of the feature normal (Pharao convention). All `0.0` for types without a normal. |


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

`replace` — returns a **new** `PharmacophorePoint` with selected fields updated (keyword-only). If `type` changes to one that doesn't carry a normal, the new normal is zeroed; if it does and you don't pass one, the old normal is preserved.

**Auto-generated by `@dataclass(frozen=True)`**

| Member                    | Behaviour                                                               |
| ------------------------- | ----------------------------------------------------------------------- |
| `__eq__`                  | Structural equality across all eight fields.                            |
| `__hash__`                | Hashable (because `frozen=True`).                                       |
| `__repr__`                | `PharmacophorePoint(type=…, x=…, y=…, z=…, sigma=…, nx=…, ny=…, nz=…)`. |
| `__setattr__` *(blocked)* | Raises `FrozenInstanceError` — instances are immutable.                 |


**Quick example**

```python
p = PharmacophorePoint(type=PointType.AROM, center=(0, 0, 0), normal=(0, 0, 1))
q = p.replace(center=(1, 0, 0))   # new sigma defaults are filled in for you
```

### `Pharmacophore` (detail)

**Class attributes**

| Name            | Type                             | Purpose                                                                         |
| --------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `allowed_types` | `ClassVar[frozenset[PointType]]` | Which point types may be added.                                                 |
| `kind`          | `ClassVar[str]`                  | Tag stored in JSON (`"query"` / `"molecule"`); used to round-trip the subclass. |


**Private / internal**: methods prefixed with `_`.


### `PharmacophoreSearch` (internal hooks)

These exist for maintainers; normal callers use `screen()` only:

| Method                                           | Role                                              |
| ------------------------------------------------ | ------------------------------------------------- |
| `__post_init__()`                                | Validates `query` and fills default `perception`. |
| `_search_with_alignment(query, db, min_matches)` | Core alignment loop.                              |
| `_screen_one_mol(mol, *, ...)`                   | Conformer loop for one molecule.                  |


`PharmacophoreSearch` also inherits dataclass `__eq__`, `__repr__`.

```python
pharmacophore_to_mol(ph, *, name="pharmacophore")
```

is also exported if you want to embed a pharmacophore in your own RDKit pipelines (returns `Chem.Mol` with one disconnected pseudo-atom per feature, plus per-mol SDF properties `kind`, `name`, `num_features`, `types`, `sigmas`, `centers`, `normals`).