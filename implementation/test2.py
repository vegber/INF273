import numpy as np

from utils_code.pdp_utils import *

# var = [3, 3, 0, 7, 1, 7, 1, 0, 6, 6, 0, 2, 5, 5, 4, 2, 4]
# var = [1, 1, 0, 0, 5, 5, 6, 6, 0, 7, 2, 3, 2, 4, 7, 3, 4]
# var = [7, 7, 0, 5, 5, 2, 2, 0, 1, 3, 1, 3, 0, 6, 4, 4, 6]
var = [5, 5, 2, 2, 0, 7, 7, 0, 1, 3, 3, 1, 0, 6, 4, 6, 4]
var2 = np.array([5, 5, 2, 2, 0, 7, 7, 0, 1, 3, 3, 1, 0, 6, 4, 6, 4])
problem = load_problem('../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt')
check = feasibility_check(var, problem)
cost = cost_function(var2, problem)
print(check)
print(cost)
