import multiprocessing as mp
import time
import pandas as pd
import numpy as np
from numpy import mean
from utils_code.pdp_utils import *
from natsort import natsorted, ns
import os
from pretty_html_table import build_table

path = '../utils_code/pdp_utils/data/pd_problem/'
file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
avg_objs = []
bst_costs = []
improvements = []
times = []
bst_solutions = []
pd_problem_file = []


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


def sort_tuple(best_sol_cost_pairs):
    return sorted(best_sol_cost_pairs, key=lambda x: x[1])


def avg_Cost(sorted_10_best_solutions):
    return sum(x[1] for x in sorted_10_best_solutions) / len(sorted_10_best_solutions)


def calculate(problem: object, vehicle: int, call: int):
    times_ = []
    best_sol_cost_pairs = []

    init_sol = get_init(vehicle, call)  # create initial worst solution
    init_cost = cost_function(init_sol, problem)  # initial worst cost

    for rounds in range(10):
        round_times = []
        current_lowest_cost = init_cost  # init max size
        solution_cost = []
        start_pr_iter = time.time()
        for x in range(10000):  # 10 000 iterations

            curr_ = get_permutation(vehicle, call)  # get random permutation
            feasible, error_code = feasibility_check(curr_, problem)  # Check if feasible sol
            if feasible:
                x_cost = cost_function(curr_, problem)  # find cost
                if x_cost < current_lowest_cost:
                    current_lowest_cost = x_cost
                    solution_cost.append((curr_, x_cost))

        stop_pr_iter = time.time()
        times_.append(stop_pr_iter - start_pr_iter)

        if len(solution_cost) == 0:
            solution_cost.append((init_sol, init_cost))  # add default if none other found
        sorted_solutions = sort_tuple(solution_cost)
        best_sol_cost_pairs.append(sorted_solutions[0])  # add the best solution of 10k'th round

        # get_avg_min_cost(avg_score, init_cost, best_score, round_Costs)
        # improv.append(100 * (init_cost - current_lowest_cost) / init_cost)

    sorted_10_best_solutions = sort_tuple(best_sol_cost_pairs)
    all_time_best_cost = np.round(sorted_10_best_solutions[0][1], 2)
    improvement = np.round(100 * (init_cost - all_time_best_cost) / init_cost, 2)
    avg_running_time = np.round(np.average(times_), 2)
    avg_objective_cost = np.round(avg_Cost(sorted_10_best_solutions), 2)
    best_solution_ = sorted_10_best_solutions[0][0]
    return avg_objective_cost, all_time_best_cost, improvement, avg_running_time, best_solution_


def store(avg, bst_cost, impr, time, bst_sol, pd_problem_file):
    avg_objs.append(avg), bst_costs.append(bst_cost), improvements.append(impr),
    times.append(time), bst_solutions.append(bst_sol)


def format_bst_sols():
    solutions_as_str = []
    for sol in bst_solutions:
        sol_str = ','.join(map(str, sol))

        """for index_ in sol:
            sol_str += str(index_)"""
        solutions_as_str.append(sol_str)
    return solutions_as_str


def to_pandas():
    best_solution_secondtry = format_bst_sols()
    data = {
        ' ': 'Random Solution',
        'Average Objective': avg_objs,
        'Best Objective': bst_costs,
        'Improbements': improvements,
        'Running Time': times,
        'Solutions': best_solution_secondtry # bst_solutions
    }
    pd.option_context('display.max_colwidth', None, "display.max_rows", None, 'display.max_columns', None)
    df = pd.DataFrame(data)
    # csv = df.to_csv()
    # html = df.to_html()
    html_table_blue_light = build_table(df, 'blue_light')
    file = open('index.html', 'w+')
    file.write(html_table_blue_light)
    file.close()


def main_run(pd_problem_file: str):
    call, vehicle = map(int, extract_values(pd_problem_file))
    problem = load_problem(path + pd_problem_file)
    avg, bst_cost, impr, time, bst_sol = calculate(problem, vehicle, call)
    store(avg, bst_cost, impr, time, bst_sol, pd_problem_file)
    print_term(avg, bst_cost, impr, time, bst_sol, pd_problem_file)
    # print(bst_sol, end="\n")


def print_term(avg_obj, best_obj, imprv, time, best_sol, file_name):
    print(f"Running {file_name}: ")
    print("\t\t\t\t | \t %s \t | \t %s \t | \t %s| \t %s \t | \t %s \t".format() % (
        "Average objective", "Best Objective", "Improvement (%) ", "RunTime", "Solution"))
    print("=" * 90)
    print("%10s | \t %10s \t\t | \t %10s \t\t | \t %10s \t | \t %10s  | \t %5s ".format() % (
        "Random strategy ", str(np.round(avg_obj, 2)), str(round(best_obj, 2)), str(round(imprv, 2)),
        str(round(time, 2)),
        str(best_sol)),
          end="\n")
    print("\n" * 2)


def run_all(processor_nbr: int):
    main_run(file_list[processor_nbr])


if __name__ == '__main__':
    for x in range(len(file_list)):
        run_all(x)
    to_pandas()

"""if __name__ == '__main__':
    start_tot = time.time()

    cores = mp.cpu_count()
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(processes=len(file_list))

    pool.map(run_all, range(0, len(file_list)))

    stop_tot = time.time()
    print(f"Totall time was {stop_tot - start_tot}")
    pool.close()
"""
