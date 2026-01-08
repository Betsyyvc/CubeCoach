from cubecoach.vision.scan import _average_samples_per_cell


def test_average_samples_per_cell_simple():
    # two samples per cell
    samples = [ [(10,10,10),(20,20,20)] for _ in range(9) ]
    averaged = _average_samples_per_cell(samples)
    assert len(averaged) == 9
    # mean of (10,10,10) and (20,20,20) is (15,15,15)
    assert averaged[0] == (15,15,15)


def test_average_with_empty_cell():
    samples = [ [] for _ in range(9) ]
    samples[3] = [(0,0,0),(30,30,30)]
    averaged = _average_samples_per_cell(samples)
    assert averaged[0] == (0,0,0)  # empty cells default to zero
    assert averaged[3] == (15,15,15)
