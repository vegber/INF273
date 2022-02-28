import time

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
        self.vessel_cargo = self.unpacked_problem['VesselCargo']

    def run(self, operator):  # define which operator to work on - higher order func.
        init = get_init(self.vehicle, self.calls).tolist()
        # set best solution
        best_solution = init  # cost_function(init, self.unpacked_problem)
        best_sol_cost = cost_function(best_solution, self.unpacked_problem)
        start = time.time()
        for x in range(10000):
            new_sol = operator(best_solution, self.vehicle, self.calls, self.vessel_cargo)
            feas, lim = feasibility_check(new_sol, self.unpacked_problem)
            if feas and cost_function(new_sol, self.unpacked_problem) < best_sol_cost:
                best_solution = new_sol
                best_sol_cost = cost_function(best_solution, self.unpacked_problem)

        print(f"time was: {time.time() - start}")
        # print(best_solution)
        print(best_sol_cost)

        # beat this motherfucker!
        # print(cost_function([4, 4, 7, 7, 0, 2, 2, 0, 1, 5, 5, 3, 3, 1, 0, 6, 6], self.unpacked_problem))

        return best_solution


if __name__ == '__main__':
    file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
    local_hero = LocalSearch(file_list[5])
    for x in range(10):
        print(local_hero.run(one_insert))
