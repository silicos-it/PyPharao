# PyPharao

Python library for **3D pharmacophore** representation and **Gaussian volume alignment** matching the [Pharao](https://github.com/silicos-it/pharao) C++ implementation, with optional **RDKit** perception from 3D molecules.

## Install

**pip** (NumPy only for core geometry; RDKit required for `pharmacophore_from_rdkit`):

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

## Quick example: pharmacophore search

```python
from rdkit import Chem
from pypharao import Pharmacophore, PharmacophoreSearch, pharmacophore_from_rdkit, PerceptionOptions

ref = Pharmacophore.from_json_file("query.json")
mol = Chem.MolFromMolFile("ligand.sdf", removeHs=False)
db = pharmacophore_from_rdkit(mol, PerceptionOptions(), conf_id=0)
result = PharmacophoreSearch().search(ref, db)
print(result.tanimoto, result.overlap_volume)
```

## Creating a pharmacophore

Feature centers are in **ångström**. For features that use a normal vector, `nx`, `ny`, and `nz` are the **absolute coordinates of the normal tip** (same convention as Pharao `.phar` files), not a unit direction on its own.

### From Python (manual)

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

### From JSON

```python
from pypharao import Pharmacophore

ph = Pharmacophore.from_json_file("query.json")
ph = Pharmacophore.from_json('{"version": 1, "name": "q", "points": []}')
```

### From Pharao `.phar`

```python
from pypharao import Pharmacophore

ph = Pharmacophore.read_phar("query.phar")
ph = Pharmacophore.from_phar_text(text)
```

### From RDKit (`pharmacophore_from_rdkit`)

Requires RDKit and a molecule with **at least one 3D conformer**. `PerceptionOptions` toggles which feature classes are perceived (`arom`, `hdon`, `hacc`, `lipo`, `posc`, `negc`, `hybh`, `hybl`).

```python
from rdkit import Chem
from pypharao import pharmacophore_from_rdkit, PerceptionOptions

mol = Chem.MolFromMolFile("ligand.sdf", removeHs=False)
opts = PerceptionOptions()
ph = pharmacophore_from_rdkit(mol, opts, conf_id=0, name="ligand_0")
```

If RDKit is not installed, `from pypharao import pharmacophore_from_rdkit` sets `pharmacophore_from_rdkit` to `None`.

## License

LGPL-3.0-or-later (see `LICENSE`).
