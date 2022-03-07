import random

from operators import find_valid_feasable_placements, to_list_v2
from random_solution_driver import get_init
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'

prob = load_problem(path)

tst = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
cycles = tst[0: 2]# random.randrange(0, len(tst), 2)]
print(cycles)
