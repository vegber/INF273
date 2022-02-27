from itertools import groupby

import numpy as np

from random_solution_driver import get_init
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'

problem = load_problem(path)
init_ = get_init(problem['n_vehicles'], problem['n_calls'])
print(init_)
sol = np.array([7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 0, 6, 6])


def to_list(L):
    # [list(group) for key, group in groupby(L, lambda x: x == 0)]
    lst_string = "".join([str(x) for x in L])
    lst2 = lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    lst4 = [list(map(int, z)) for z in lst3]
    print(lst4)


def one_insert(L, prob):
    # L to n + 1 lists
    list_indexed = to_list(L)
    print(list_indexed)


one_insert(sol, prob=problem)
