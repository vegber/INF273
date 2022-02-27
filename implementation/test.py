from itertools import groupby

import numpy as np
import time
from random_solution_driver import get_init
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'

problem = load_problem(path)
init_ = get_init(problem['n_vehicles'], problem['n_calls'])
sol = np.array([7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 6,6, 0]) # 6,6
print(sol)


def to_list(L):
    lst_string = "".join([str(x) for x in L])
    lst2 = [e + '0' for e in lst_string.split('0') if e]  # lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    lst4 = [list(map(str, z)) for z in lst3]
    lst4[-1].remove('0')
    return np.array(map(int, ''.join([''.join(x) for x in lst4])))
    # var = [list(map(int, group)) for key, group in groupby(L, lambda x: x == 0)]
    # return var


def one_insert(L, prob):
    # L to n + 1 lists
    start = time.time()
    for x in range(1000000):
       list_indexed = to_list(init_)

    print(time.time()-start)


one_insert(sol, prob=problem)
