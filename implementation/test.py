from itertools import groupby

import numpy as np
import time
from random_solution_driver import get_init
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'

problem = load_problem(path)
init_ = get_init(problem['n_vehicles'], problem['n_calls'])
sol = np.array([7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 6, 6, 0])  # 6,6
print(sol)


def to_list(L):
    lst_string = "".join([str(x) for x in L])
    lst2 = [e + '0' for e in lst_string.split('0') if e]  # lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    lst4 = [list(map(str, z)) for z in lst3]
    lst4[-1].remove('0')
    va = list(map(int, ''.join([''.join(x) for x in lst4])))
    print(va)
    return va


def to_list2(L):
    # [list(group) for key, group in groupby(L, lambda x: x == 0)]
    lst_string = "".join([str(x) for x in L])
    lst2 = lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    return [list(map(int, z)) for z in lst3]


def one_insert(L, prob):
    # L to n + 1 lists
    var = (1, 2)
    print(bool(var))
    n_list = to_list2(L)
    print(n_list)
    print(problem['VesselCargo'])


one_insert(sol, prob=problem)
