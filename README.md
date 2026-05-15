# PyPharao

Python library for **3D pharmacophore** representation and **Gaussian volume alignment** matching the [Pharao](https://github.com/silicos-it/pharao) C++ implementation, with optional perception from 3D molecules (via RDKit).

## 1. Install

**pip** (NumPy only for core geometry; install the `rdkit` extra for `pharmacophore_from_molecule`):

```bash
pip install -e ".[dev,rdkit,examples]"
```

(`examples` pulls in **tqdm** for progress bars in `examples/`. To add it later: `pip install tqdm` or `pip install -e ".[examples]"`.)

Use a Conda **conda-forge** environment if `pip install rdkit` is unavailable on your platform:

```bash
conda install -c conda-forge rdkit numpy
pip install -e .
```

**conda-build** (local recipe under `conda-recipe/`):

```bash
conda build conda-recipe
```

## 2. Quick example: pharmacophore search

```python
from rdkit import Chem
from pypharao import Pharmacophore, PharmacophoreSearch, pharmacophore_from_molecule, PerceptionOptions

ref = Pharmacophore.from_json_file("query.json")
mol = Chem.MolFromMolFile("ligand.sdf", removeHs=False)
searcher = PharmacophoreSearch(ref)  # default: all query features must match
# searcher = PharmacophoreSearch(ref, min_matched_query_features=2)  # partial OK
result = searcher.search_with_molecule(mol)
print(result.tanimoto, result.overlap_volume)
```

## 3. Creating a pharmacophore

Feature centers are in **ångström**. For features that use a normal vector, `nx`, `ny`, and `nz` are the **absolute coordinates of the normal tip** (same convention as Pharao `.phar` files), not a unit direction on its own.

#### From Python (manual)

Build a `Pharmacophore` from `PharmacophorePoint` instances. Use `FuncGroup` for the feature type, `alpha` for the Gaussian width, and set `has_normal` when normals apply. Helpers `default_alpha()` and `FUNC_HAS_NORMAL` match Pharao-style defaults.

```python
from pypharao import (
    FuncGroup,
    Pharmacophore,
    PharmacophorePoint,
    default_alpha,
)

ph = Pharmacophore(
    name="my_query",
    points=[
        PharmacophorePoint(
            x=1.0,
            y=2.0,
            z=3.0,
            func=FuncGroup.AROM,
            alpha=default_alpha(FuncGroup.AROM),
            has_normal=True,
            nx=2.0,
            ny=2.0,
            nz=4.0,
        ),
    ],
)

# Or start empty and append
ph2 = Pharmacophore(name="empty")
ph2.append_point(
    PharmacophorePoint(0.0, 0.0, 0.0, FuncGroup.LIPO, default_alpha(FuncGroup.LIPO), False)
)
```

#### From JSON

```python
from pypharao import Pharmacophore

ph = Pharmacophore.from_json_file("query.json")
ph = Pharmacophore.from_json('{"version": 1, "name": "q", "points": []}')
```

#### From Pharao `.phar`

```python
from pypharao import Pharmacophore

ph = Pharmacophore.read_phar("query.phar")
ph = Pharmacophore.from_phar_text(text)
```

#### From a 3D molecule (`pharmacophore_from_molecule`)

Requires the `rdkit` extra and a **3D structure** as an RDKit `Chem.Mol` with at least one conformer. Pass `PerceptionOptions` to control which feature types are detected (see below).

```python
from rdkit import Chem
from pypharao import pharmacophore_from_molecule, PerceptionOptions

mol = Chem.MolFromMolFile("ligand.sdf", removeHs=False)
opts = PerceptionOptions()
ph = pharmacophore_from_molecule(mol, opts, conf_id=0, name="ligand_0")
```

If RDKit is not installed, `pharmacophore_from_molecule` is `None` on import. The previous name `pharmacophore_from_rdkit` remains available as an alias.

You can also match a reference pharmacophore directly against a molecule:

```python
result = PharmacophoreSearch(ref).search_with_molecule(mol, conf_id=0)
```

(`search_with_rdkit_mol` is an alias for backward compatibility.)

### `PerceptionOptions`

`PerceptionOptions` is a small configuration object that controls **which pharmacophore feature types are detected** when building a pharmacophore from a 3D molecule via `pharmacophore_from_molecule()`. It has no RDKit import of its own, so you can configure perception without loading RDKit until you call the perception function.

| Flag | Default | When `True`, detects… |
|------|---------|------------------------|
| `arom` | `True` | **AROM** — aromatic ring centroids with plane normals |
| `hdon` | `True` | **HDON** — H-bond donors (N/O with H, not negatively charged) |
| `hacc` | `True` | **HACC** — H-bond acceptors (N/O, with Pharao-style filters) |
| `lipo` | `True` | **LIPO** — lipophilic regions (from molecular surface) |
| `posc` | `True` | **POSC** — positively charged atoms |
| `negc` | `True` | **NEGC** — negatively charged atoms |
| `hybh` | `True` | **HYBH** — hybrid H-bond features (see below) |
| `hybl` | `True` | **HYBL** — hybrid lipophilic/aromatic features (see below) |

All flags default to `True`, so `PerceptionOptions()` enables the full Pharao-style perception set.

`pharmacophore_from_molecule(mol, options, ...)` reads each flag and **skips** the corresponding detection step when it is `False`. Hybrid features have extra dependencies:

- **`hybh`** runs only if `hybh`, `hdon`, and `hacc` are all enabled. It merges nearby donor/acceptor pairs into **HYBH** points.
- **`hybl`** runs only if `hybl`, `arom`, and `lipo` are all enabled. It merges overlapping aromatic and lipophilic sites into **HYBL** points.

Turning off `hdon` also disables hybrid H-bond perception, even if `hybh` is still `True`. The same applies to `hybl` when `arom` or `lipo` is off.

To inspect which feature types exist and whether they will be detected for the current flags:

- **`print_features()`** — prints every `FuncGroup` with a short description and detection status (`on` / `off` / `—` for manual-only types such as **EXCL** and **UNDEF**). Hybrid types show `off (needs …)` when their flag is on but a prerequisite is off.
- **`is_enabled_for_perception(func)`** — returns `True` or `False` when molecule perception is controlled by a flag; returns `None` for **EXCL** and **UNDEF** (not emitted from molecules). Hybrid prerequisites are applied the same way as in `pharmacophore_from_molecule()`.
- **`enable(func)`** / **`disable(func)`** / **`set_enabled(func, enabled)`** — turn flags on or off after construction (`func` is a `FuncGroup` or string such as `"LIPO"`). Raises `ValueError` for **EXCL** and **UNDEF**. Boolean fields (`opts.arom`, etc.) remain writable directly.

```python
from pypharao import FuncGroup, PerceptionOptions

opts = PerceptionOptions()
opts.disable(FuncGroup.LIPO)
opts.enable("AROM")
opts.print_features()
assert opts.is_enabled_for_perception(FuncGroup.LIPO) is False
assert opts.is_enabled_for_perception(FuncGroup.EXCL) is None
```

`PerceptionOptions` does **not** affect manual pharmacophores, JSON/`.phar` loading, or search/alignment — only molecule perception. It does not tune Gaussian widths (`alpha`), distances, or perception thresholds beyond on/off per feature class.

When you use `PharmacophoreSearch(ref)`, database molecules are perceived with flags derived automatically from the feature types in `ref` (`perception_options_from_pharmacophore`). For example, a query with **AROM** and **HYBH** only enables `arom`, `hdon`, `hacc`, and `hybh` on ligands—not `lipo` or `hybl`, which avoids spurious **HYBL** sites on aliphatic rings matching query **AROM**.

By default, a hit must map **every** matchable query point (all types except **EXCL** and **UNDEF**). Pass `min_matched_query_features` at construction to allow partial matches, or set it explicitly to `count_query_features(ref)` for the same strict behaviour.

```python
from pypharao import PerceptionOptions, pharmacophore_from_molecule

# Only aromatic and H-bond features; no lipophilic, charge, or hybrid
opts = PerceptionOptions(
    lipo=False,
    posc=False,
    negc=False,
    hybl=False,
)
ph = pharmacophore_from_molecule(mol, opts, conf_id=0)
```

## 4. Pharmacophores and PharmacophorePoints

A **`Pharmacophore`** is a named collection of features (`points`). A **`PharmacophorePoint`** is one Gaussian pharmacophore site: coordinates `(x, y, z)`, feature type (`func`), width (`alpha`), and optional normal tip `(nx, ny, nz)` when `has_normal` is true (Pharao convention: normal tip in absolute ångström, not a unit vector by itself).

### `PharmacophorePoint` (immutable)

Points are immutable value objects. To change coordinates, type, or width, build a new point and assign it into the pharmacophore:

```python
from pypharao import FuncGroup, PharmacophorePoint, default_alpha

p = PharmacophorePoint(0.0, 0.0, 0.0, FuncGroup.LIPO, default_alpha(FuncGroup.LIPO), False)

# Update selected fields; func is unchanged
p2 = p.with_fields(x=1.0, y=2.0, z=3.0, alpha=0.8)

# Or construct a new point explicitly (e.g. change feature type)
p3 = PharmacophorePoint(
    p.x, p.y, p.z,
    FuncGroup.AROM,
    default_alpha(FuncGroup.AROM),
    True,
    p.nx, p.ny, p.nz,
)
```

### `Pharmacophore` (mutable)

You can edit a pharmacophore after it is built (manually, from JSON/`.phar`, or from a molecule):

```python
from pypharao import Pharmacophore, PharmacophorePoint, FuncGroup, default_alpha

ph = Pharmacophore(name="query", points=[...])

ph.name = "renamed_query"
ph.append_point(PharmacophorePoint(1, 2, 3, FuncGroup.HACC, default_alpha(FuncGroup.HACC), True, 1, 2, 4))
ph.remove_at(0)
ph.clear()

len(ph)
p0 = ph[0]
for pt in ph:
    ...

ph.points[0] = ph.points[0].with_fields(x=1.0)
copy = ph.copy()
```

| Method / attribute | Purpose |
|--------------------|---------|
| `name` | Query or ligand label (read/write) |
| `points` | List of `PharmacophorePoint` (mutable list) |
| `append_point(p)` | Append a feature |
| `remove_at(i)` | Remove feature by index |
| `clear()` | Remove all features |
| `copy()` | Shallow copy (same point objects) |
| `to_json` / `write_json` | Serialize current state |
| `to_phar_text` / `write_phar` | Write Pharao `.phar` format |

`PerceptionOptions` only applies when **creating** a pharmacophore from a molecule. To change which features are perceived, call `pharmacophore_from_molecule()` again with new options; you cannot “re-perceive” an existing `Pharmacophore` in place.

## License

LGPL-3.0-or-later (see `LICENSE`).
