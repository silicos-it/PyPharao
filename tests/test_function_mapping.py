from pypharao import FuncGroup, Pharmacophore, PharmacophorePoint
from pypharao.function_mapping import FunctionMapping


def test_function_mapping_single_pair():
    ref = Pharmacophore(
        points=[
            PharmacophorePoint(0, 0, 0, FuncGroup.LIPO, 0.7, False, 0, 0, 0),
        ]
    )
    db = ref.copy()
    fm = FunctionMapping(ref, db, 0.5)
    m = fm.get_next_map()
    assert m == [(0, 0)]
