from pyinform import transferentropy as te

def test_complete_te_multiple_initial_conditions():
    '''Reproduces https://github.com/ELIFE-ASU/Inform/issues/78'''

    k = 1
    xs = [0, 1, 1, 1, 1, 0, 0, 0, 0]
    ys = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    
    cte = te.transfer_entropy(xs, ys, condition=xs, k=k)
    assert abs(cte) < 1e-6

    xs = [[1, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 1]]
    ys = [[0, 0, 1, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 1, 1, 1, 0]]
    
    cte = te.transfer_entropy(xs, ys, condition=xs, k=k)
    assert abs(cte) < 1e-6
    