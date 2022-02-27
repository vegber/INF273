import multiprocessing as mp
import time
from numpy import mean
from utils_code.pdp_utils import *
from natsort import natsorted, ns
import os

path = '../utils_code/pdp_utils/data/pd_problem/'
file_list = natsorted(os.listdir(path), key=lambda y: y.lower())


def extract_values(filename: str) -> tuple:
    vals = [s for s in filename[:-4].split("_")]
    vals = [int(i) for i in vals if i.isdigit()]
    return vals[0], vals[1]


# Original list from org rep.... dunno what it does
sol = [7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 0, 6, 6]


def get_permutation(vehicle: int, call: int):
    return np.random.permutation([0] * vehicle + list(range(1, call)) * 2)


def get_init(vehicle: int, call: int):
    return np.array([0] * vehicle + list(range(1, call)) * 2)


def calculate(problem: object, vehicle: int, call: int):
    avg_cost_p_round = []
    min_cost_p_round = []
    times_ = []
    improv = []

    init_sol = get_init(vehicle, call)
    init_cost = cost_function(init_sol, problem)

    for rounds in range(10):

        round_Costs = []  # initialize lists
        current_lowest_cost = init_cost  # init max size

        start_p10k = time.time()
        for x in range(10000):  # 10 000 iterations

            curr_ = get_permutation(vehicle, call)  # get random permutation
            feasible, error_code = feasibility_check(curr_, problem)  # Check if feasible sol
            if feasible:
                one_round_cost = cost_function(curr_, problem)  # find cost
                round_Costs.append(one_round_cost)  # add to round cost

                if one_round_cost < current_lowest_cost:
                    current_lowest_cost = one_round_cost

        stop_p10k = time.time()
        times_.append(stop_p10k - start_p10k)

        get_avg_min_cost(avg_cost_p_round, init_cost, min_cost_p_round, round_Costs)

        improv.append(100 * (init_cost - current_lowest_cost) / init_cost)
    return avg_cost_p_round, min_cost_p_round, times_, improv


def get_avg_min_cost(avg_cost_p_round, init_cost, min_cost_p_round, round_Costs):
    if len(round_Costs) == 0:
        avg_cost_p_round.append(init_cost)
        min_cost_p_round.append(init_cost)
    else:
        avg_cost_p_round.append(mean(round_Costs))
        min_cost_p_round.append(np.min(round_Costs))


def main_run(pd_problem_file: str):
    call, vehicle = map(int, extract_values(pd_problem_file))
    problem = load_problem(path + pd_problem_file)
    avg_cost_p_round, min_cost_p_round, times_, improv = calculate(problem, vehicle, call)
    print_term(avg_cost_p_round, min_cost_p_round, improv, times_, pd_problem_file)


def print_term(avg_obj, best_obj, imprv, time, file_name):
    print(f"Running {file_name}: ")
    print("\t | \t %s \t | \t %s \t | \t %s \t | \t %s \t |".format() % (
        "Average objective", "Best Objective", "Improvement", "RunTime"))
    print("=" * 90)
    for index in range(len(avg_obj)):
        avg, best, impr, run = avg_obj[index], best_obj[index], imprv[index], time[index]
        print("%10s | \t %10s \t\t | \t %10s \t\t | \t %10s \t | \t %10s  |".format() % (
            "Random strategy ", str(np.round(avg, 2)), str(round(best, 2)), str(round(impr, 2)), str(round(run, 2))), end="\n")
    print("\n" * 2)


def run_all(processor_nbr: int):
    main_run(file_list[processor_nbr])


if __name__ == '__main__':
    start_tot = time.time()

    cores = mp.cpu_count()
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(processes=len(file_list))

    pool.map(run_all, range(0, len(file_list)))

    stop_tot = time.time()
    print(f"Totall time was {stop_tot - start_tot}")
    pool.close()
