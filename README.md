# PyPharao

Python library for **3D pharmacophore** representation and **Gaussian volume alignment** matching the [Pharao](https://github.com/silicos-it/pharao) C++ implementation, with optional perception from 3D molecules (via RDKit).

## 1. Install

**pip** (NumPy only for core geometry; install the `rdkit` extra for `pharmacophore_from_molecule`):

```bash
pip install -e ".[dev,rdkit]"
```

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
db = pharmacophore_from_molecule(mol, PerceptionOptions(), conf_id=0)
result = PharmacophoreSearch().search(ref, db)
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
result = PharmacophoreSearch().search_with_molecule(ref, mol, conf_id=0)
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

`PerceptionOptions` does **not** affect manual pharmacophores, JSON/`.phar` loading, or search/alignment — only molecule perception. It does not tune Gaussian widths (`alpha`), distances, or perception thresholds beyond on/off per feature class. It is not used by `PharmacophoreSearch` itself; you pass the resulting `Pharmacophore` into search separately.

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

## License

LGPL-3.0-or-later (see `LICENSE`).
