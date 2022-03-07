import multiprocessing as mp
import os
import random
import time
from collections.abc import Iterable

import pandas as pd
from natsort import natsorted
from pretty_html_table import build_table

from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/'

file_list = natsorted(os.listdir(path), key=lambda y: y.lower())
avg_objs = []
bst_costs = []
improvements = []
times = []
bst_solutions = []
pd_problem_file = []


# can be deleted
def extract_values(filename: str) -> tuple:
    vals = [s for s in filename[:-4].split("_")]
    vals = [int(i) for i in vals if i.isdigit()]
    return vals[0], vals[1]


# Original list from org rep.... dunno what it does
sol = [7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 0, 6, 6]


def get_permutation(vehicle: int, call: int):
    actual_solution = []
    solution_space = []
    available_calls = []

    available_calls.extend(range(1, call + 1))
    cars = vehicle  # random.randint(0, vehicle)

    for car in range(cars):
        # find random calls for each car
        car_choice = random.sample(available_calls, (random.randint(0, len(available_calls))))
        # append calls to solution list
        solution_space.append(car_choice)
        # remove calls in use:
        available_calls = [i for i in available_calls if i not in car_choice]

    # solution space need to be doubled, and permuted
    # then solution space need to be added together, and separated with zero

    for vehicle_call in solution_space:
        # a list of calls - could be empty
        if len(vehicle_call) == 0:
            actual_solution.append(0)
        else:
            elements = [elem for elem in vehicle_call] * 2
            random.shuffle(elements)
            vehicle_call_ = elements + [0]
            actual_solution.append(vehicle_call_)

    # if not, all cars is used, and we need to out-source the rest
    if len(available_calls) != 0:
        remaining_calls = [elem for elem in available_calls] * 2
        random.shuffle(remaining_calls)
        actual_solution.append(remaining_calls)

    return np.array(list(map(int, flatten(actual_solution))))


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def get_init(vehicle: int, call: int):
    init = [0] * vehicle
    for i in range(1, call + 1):
        init.append(i)
        init.append(i)
    return init
    # return np.array([0] * vehicle + list(range(1, call + 1)) * 2)


def sort_tuple(best_sol_cost_pairs):
    return sorted(best_sol_cost_pairs, key=lambda x: x[1])


def avg_Cost(sorted_10_best_solutions):
    return sum(x[1] for x in sorted_10_best_solutions) / len(sorted_10_best_solutions)


def calculate(problem: object):  # , vehicle: int, call: int):
    vehicle = problem['n_vehicles']
    calls = problem['n_calls']

    times_ = []
    best_sol_cost_pairs = []
    init_sol = get_init(vehicle, calls)  # create initial worst solution
    init_cost = cost_function(init_sol, problem)  # initial worst cost

    for rounds in range(10):
        current_lowest_cost = init_cost  # init max size
        solution_cost = []
        start_pr_iter = time.time()
        for it in range(10000):  # 10 000 iterations

            curr_ = get_permutation(vehicle, calls)  # get random permutation
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

    sorted_10_best_solutions = sort_tuple(best_sol_cost_pairs)  # sort 10 best
    all_time_best_cost = np.round(sorted_10_best_solutions[0][1], 2)  # choose best elem

    improvement = np.round(100 * (init_cost - all_time_best_cost) / init_cost, 2)  # improvement as per doc.
    avg_running_time = np.round(np.average(times_), 2)  # average running time

    avg_objective_cost = np.round(avg_Cost(sorted_10_best_solutions), 2)  # average cost
    best_solution_ = sorted_10_best_solutions[0][0]  # best sol - will be first elem

    return avg_objective_cost, all_time_best_cost, improvement, avg_running_time, best_solution_


def store(avg, bst_cost, impr, tme, bst_sol):
    avg_objs.append(avg), bst_costs.append(bst_cost), improvements.append(impr),
    times.append(tme), bst_solutions.append(bst_sol)


def format_bst_sols():
    return [(','.join(map(str, s))) for s in bst_solutions]


# todo
def to_pandas():
    best_solution_secondtry = format_bst_sols()
    data = {
        ' ': 'Random Solution',
        'Average Objective': avg_objs,
        'Best Objective': bst_costs,
        'Improvements': improvements,
        'Running Time': times,
        'Solutions': best_solution_secondtry  # bst_solutions
    }
    pd.option_context('display.max_colwidth', None, "display.max_rows", None, 'display.max_columns', None)
    df = pd.DataFrame(data)
    # csv = df.to_csv()
    # html = df.to_html()
    html_table_blue_light = build_table(df, 'blue_light')
    file = open('index.html', 'w+')
    file.write(html_table_blue_light)
    file.close()


def main_run(prob_file: str):
    problem = load_problem(path + prob_file)
    avg, bst_cost, improvement, tme, bst_sol = calculate(problem)  # , vehicle, call)
    store(avg, bst_cost, improvement, tme, bst_sol)  # store files
    print_term(avg, bst_cost, improvement, tme, bst_sol, prob_file)


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


"""if __name__ == '__main__':
    for x in range(len(file_list)):
        main_run(file_list[x])
    to_pandas()
    
    """


def run_all(x):
    main_run(file_list[x])


if __name__ == '__main__':
    start_tot = time.time()

    cores = mp.cpu_count()
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(processes=len(file_list))

    pool.map(run_all, range(0, len(file_list)))

    stop_tot = time.time()
    print(f"Totall time was {stop_tot - start_tot}")
    pool.close()
