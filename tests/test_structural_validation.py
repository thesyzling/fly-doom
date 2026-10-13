import numpy as np
from scipy import sparse

from flydoom.structural_validation import rewire_relays


def test_signed_swaps_preserve_degrees_visual_rows_and_target_weight_multisets():
    rng = np.random.default_rng(19)
    rows, columns, values = [], [], []
    for target in range(40):
        pool = np.arange(8 if target < 8 else 40)
        pool = pool[pool != target]
        source = rng.choice(pool, min(5, len(pool)), replace=False)
        rows.extend([target]*len(source)); columns.extend(source)
        values.extend((.01+rng.random(len(source))*.01)*np.where(source % 2, -1, 1))
    old = sparse.csr_matrix((np.array(values, np.float32), (rows, columns)), shape=(40, 40))
    new, audit = rewire_relays(old, np.arange(8), 123)
    again, _ = rewire_relays(old, np.arange(8), 123)
    assert audit['accepted_swaps'] > 0 and audit['changed_fraction'] > .2
    assert not (new[:8] != old[:8]).nnz
    assert not new[:8, 8:].nnz
    assert not (new != again).nnz
    for row in range(40):
        np.testing.assert_array_equal(np.sort(new[row].data), np.sort(old[row].data))
    for sign in (-1, 1):
        np.testing.assert_array_equal(np.bincount(old.indices[np.sign(old.data) == sign], minlength=40),
                                      np.bincount(new.indices[np.sign(new.data) == sign], minlength=40))
