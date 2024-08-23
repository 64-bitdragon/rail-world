from main import *
import numpy as np
import inspect

heights_expected = [
    ([0, 0, 0], [0, 0, 0]),
    ([0, 1, 0], [0, 1, 0]),
    ([0, 0, 2], [0, 1, 2]),
    ([0, 0, 2, 2], [0, 1, 2, 2]),
    ([0, 0, 2, 1], [0, 1, 2, 1]),
    ([0, 0, 2, 3], [0, 1, 2, 3]),
    ([2, 0, 0], [2, 1, 0]),
    ([3, 0, 0], [3, 2, 1]),
    ([3, 0, 1], [3, 2, 1]),
    ([0, 0, 5, 0, 0], [3, 4, 5, 4, 3]),
    ([5, 0, 0, 0, 5], [5, 4, 4, 4, 5]),
    ([5, 0, 0, 5], [5, 4, 4, 5]),
    ([5, 0, 0, 0, 0, 0, 0, 0, 5], [5, 4, 3, 2, 2, 2, 3, 4, 5]),
    ([1, 0, 1], [1, 1, 1]),
]

for heights, expected in heights_expected[:]:
    actual = adjust_hills(heights)
    if np.array_equal(expected, actual):
        print(f'test passed')
    else:
        print(f'test failed\nexpected: {expected}\nbut was: {actual}')
