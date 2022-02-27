import numpy as np

from random_solution_driver import get_init, get_permutation
from utils_code.pdp_utils import *
from operators import *
from natsort import natsorted
import os

path = '../utils_code/pdp_utils/data/pd_problem/'


class LocalSearch:
    def __init__(self, problem):
        self.unpacked_problem = load_problem(path + problem)
        self.vehicle = self.unpacked_problem['n_vehicles']
        self.calls = self.unpacked_problem['n_calls']

    def run(self, operator):  # define which operator to work on - higher order func.
        init = operator(get_init(self.vehicle, self.calls), self.vehicle, self.calls)

        # set best solution
        best_solution = init # cost_function(init, self.unpacked_problem)
        best_sol_cost = cost_function(best_solution, self.unpacked_problem)
        for x in range(1):
            new_sol = operator(best_solution, self.vehicle, self.calls)
            if feasibility_check(new_sol, self.unpacked_problem) and cost_function(new_sol, self.unpacked_problem) < best_sol_cost:
                best_solution = new_sol
                best_sol_cost = cost_function(best_solution)

        print(best_solution)
        return best_solution


if __name__ == '__main__':
    file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
    local_hero = LocalSearch(file_list[0])
    local_hero.run(one_insert)
