
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt'

prob = load_problem(path)

sol = [4, 4, 15, 15, 11, 11, 16, 16, 0, 6, 6, 5, 14, 17, 5, 17, 14, 0, 8, 18, 18, 8, 13, 13, 0, 7, 7, 3, 3, 10, 1, 10, 1, 0, 9, 9, 12, 12, 0, 2, 2]
# 2376038.0
a, b = feasibility_check(sol, prob)
print(a)
print(cost_function(sol, prob))