import sys
import time

from numpy import mean
from pdp_utils import *


def load_correct_file(which_mode):
    return {1: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'),
            2: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt'),
            3: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt'),
            4: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt'),
            5: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt')}[which_mode]


sol = [7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 0, 6, 6]


def get_permutation(vehicle: int):
    return np.random.permutation([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7] + [0] * vehicle)


def calculate(problem: int, perm_mode):
    avg_cost_p_round = []
    min_cost_p_round = []
    times_ = []
    improv = []
    problemos = load_correct_file(problem)

    for rounds in range(10):

        round_Cost = []             # initialize lists
        curr_diff = sys.maxsize     # init max size

        start = time.time()
        for x in range(10000): # 10 000 iterations

            curr_ = get_permutation(perm_mode)              # get random permutation
            feasiblity, c = \
                feasibility_check(curr_, problemos)         # Check if feasible sol

            if feasiblity:
                round_cost = cost_function(curr_, problemos) # find cost
                round_Cost.append(round_cost)               # add to round cost
                if round_cost < curr_diff:
                    curr_diff = round_cost

        stop = time.time()
        times_.append(stop - start)

        improv.append(curr_diff)
        avg_cost_p_round.append(mean(round_Cost))
        min_cost_p_round.append(min(round_Cost))

    return avg_cost_p_round, min_cost_p_round, times_, improv


def main_run(mode: int, swag):
    avg_cost_p_round, min_cost_p_round, times_, improv = calculate(mode, swag)
    print(f"| \t Average objective | \t Best Objective | \t Improvement | \t\t Running time |", end="\n")
    for index in range(len(avg_cost_p_round)):
        avg, best, impr, run = avg_cost_p_round[index], min_cost_p_round[index], improv[index], times_[index]

        print(
            f"| \t {round(avg, 2)} \t | \t\t {round(best, 2)} \t\t | \t\t {round(impr)} \t\t | \t {round(run, 2)} \t |",
            end="\n")


main_run(1, 3)
main_run(2, 5)
main_run(3, 20)
main_run(4, 40)
main_run(5, 90)
