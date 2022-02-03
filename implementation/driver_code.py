import multiprocessing as mp
import time

from numpy import mean

from utils_code.pdp_utils import *


def load_correct_file(which_mode):
    return {1: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'),
            2: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt'),
            3: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt'),
            4: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt'),
            5: load_problem('../utils_code/pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt')}[which_mode]


def filename(file: int):
    return {1: 'Call_7_Vehicle_3.txt',
            2: 'Call_18_Vehicle_5.txt',
            3: 'Call_80_Vehicle_20.txt',
            4: 'Call_130_Vehicle_40.txt',
            5: 'Call_300_Vehicle_90.txt'}[file]


sol = [7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 0, 6, 6]


def get_permutation(vehicle: int, call: int):
    return np.random.permutation([0] * vehicle + list(range(1, call)) * 2)


def get_init(vehicle: int, call: int):
    return np.array([0] * vehicle + list(range(1, call)) * 2)


def calculate(problem: int, perm_mode, call: int):
    avg_cost_p_round = []
    min_cost_p_round = []
    times_ = []
    improv = []
    problemos = load_correct_file(problem)

    init_sol = get_init(perm_mode, call)
    init_cost = cost_function(init_sol, problemos)

    for rounds in range(10):

        round_Costs = []  # initialize lists
        current_lowest_cost = init_cost  # sys.maxsize  # init max size

        start = time.time()
        for x in range(10000):  # 10 000 iterations

            curr_ = get_permutation(perm_mode, call)  # get random permutation
            feasiblity, c = feasibility_check(curr_, problemos)  # Check if feasible sol

            if feasiblity:
                one_round_cost = cost_function(curr_, problemos)  # find cost
                round_Costs.append(one_round_cost)  # add to round cost

                if one_round_cost < current_lowest_cost:
                    current_lowest_cost = one_round_cost

        stop = time.time()
        times_.append(stop - start)

        if len(round_Costs) == 0:
            avg_cost_p_round.append(init_cost)
            min_cost_p_round.append(init_cost)
        else:
            avg_cost_p_round.append(mean(round_Costs))
            min_cost_p_round.append(np.min(round_Costs))

        improv.append(100 * (init_cost - current_lowest_cost) / init_cost)
    return avg_cost_p_round, min_cost_p_round, times_, improv


def main_run(mode: int, swag, call):
    avg_cost_p_round, min_cost_p_round, times_, improv = calculate(mode, swag, call)
    printoutput(avg_cost_p_round, improv, min_cost_p_round, mode, times_)


def printoutput(avg_cost_p_round, improv, min_cost_p_round, mode, times_):
    # print(f"| \t Average objective | \t Best Objective | \t Improvement | \t\t Running time |", end="\n")
    print(f"Running {filename(mode)}: ")
    print("\t | \t %s \t | \t %s \t | \t %s \t | \t %s \t |".format() % (
        "Average objective", "Best Objective", "Improvement", "RunTime"))
    print("=" * 90)
    for index in range(len(avg_cost_p_round)):
        avg, best, impr, run = avg_cost_p_round[index], min_cost_p_round[index], improv[index], times_[index]
        print("%4d | \t %10s \t\t | \t %10s \t\t | \t %10s \t | \t %10s  |".format() % (
            index + 1, str(np.round(avg, 2)), str(round(best, 2)), str(round(impr, 2)), str(round(run, 2))), end="\n")
    print("\n" * 2)


def run_all(var: int):
    if var == 1:
        main_run(1, 3, 7)
    if var == 2:
        main_run(2, 5, 18)
    if var == 3:
        main_run(3, 20, 80)
    if var == 4:
        main_run(4, 40, 130)
    if var == 5:
        main_run(5, 90, 300)


if __name__ == '__main__':
    start = time.time()

    cores = mp.cpu_count()
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(processes=5)

    pool.map(run_all, range(1, 5))

    stop = time.time()
    print(f"Totall time was {stop - start}")
