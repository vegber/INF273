import math
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
from natsort import natsorted
from operators import *

path = '../utils_code/pdp_utils/data/pd_problem/'
file_list = natsorted(os.listdir(path), key=lambda y: y.lower())


class Algorithms:
    def __init__(self, problem):
        self.problem_name = problem
        self.problem = load_problem(path + problem)
        self.vehicle = self.problem['n_vehicles']
        self.calls = self.problem['n_calls']
        self.vessel_cargo = self.problem['VesselCargo']
        self.top10best_solution = []  # store solution / cost
        self.run_time = []
        self.temps = []
        self.op1 = None
        self.op2 = None
        self.op3 = None

    def set_operators(self, op1, op2, op3):
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3

    def loaded_problem(self):
        return self.problem

    def local_search(self, operator):  # define which operator to work on - higher order func.
        init = get_init(self.vehicle, self.calls)
        # set best solution
        best_solution = init  # cost_function(init, self.unpacked_problem)
        best_sol_cost = cost_function(best_solution, self.problem)
        start = time.time()
        for it in range(10000):
            new_sol = operator(best_solution)
            passed, _ = feasibility_check(new_sol, self.problem)
            if passed and cost_function(new_sol, self.problem) < best_sol_cost:
                best_solution = new_sol
                best_sol_cost = cost_function(best_solution, self.problem)
        self.run_time.append(time.time() - start)
        self.top10best_solution.append((best_solution, best_sol_cost))

    def get_op(self, obj) -> Operators:
        choices = [self.op1, self.op2, self.op3]
        # return random.choice(choices)
        op_index = [0, 1, 2]
        return choices[random.choices(op_index, weights=[70, 15, 15])[0]](obj)

    def sa(self):
        s_0 = get_init(self.vehicle, self.calls)
        fin_temp = 0.1
        incumbent = s_0
        best_solution = s_0
        delta_W = []
        start = time.time()
        while len(delta_W) == 0 or sum(delta_W) == 0:
            for w in range(100):
                new_sol = self.get_op(best_solution)
                delta_E = cost_function(new_sol, self.problem) - cost_function(incumbent, self.problem)
                passed, cause = feasibility_check(new_sol, self.problem)
                if passed and delta_E < 0:
                    incumbent = new_sol
                    if cost_function(incumbent, self.problem) < cost_function(best_solution, self.problem):
                        best_solution = incumbent
                elif passed:
                    if random.random() < 0.8:
                        incumbent = new_sol
                    delta_W.append(delta_E)
        delta_AVG = np.average(delta_W)  # sum(delta_W) / len(delta_W)
        T_0 = (-delta_AVG) / np.log(0.8)
        alfa = pow(fin_temp / T_0, 1 / 9900)
        T = T_0
        temps = []
        for e in range(1, 9900):
            temps.append(T)
            new_sol = self.get_op(incumbent)
            delta_E = cost_function(new_sol, self.problem) - cost_function(incumbent, self.problem)
            feas, _ = feasibility_check(new_sol, self.problem)
            if feas and delta_E < 0:
                incumbent = new_sol
                if cost_function(incumbent, self.problem) < cost_function(best_solution, self.problem):
                    best_solution = incumbent
            elif feas and random.random() < pow(math.e, (-delta_E / T)):
                incumbent = new_sol
            T = alfa * T
        self.run_time.append(time.time() - start)
        self.top10best_solution.append((best_solution, cost_function(best_solution, self.problem)))
        self.temps.append(temps)

    def print_stats(self, operator_name=None):
        print(self.problem_name, end="\n")

        self.top10best_solution = sort_tuple(self.top10best_solution)  # sort values
        avg_obj_cost = np.round(avg_Cost(self.top10best_solution), 2)
        best_cost = self.top10best_solution[0][1]
        init_cost = cost_function(get_init(self.vehicle, self.calls), self.problem)
        improvement = np.round(100 * (init_cost - self.top10best_solution[0][1]) / init_cost, 2)
        avg_run_time = np.round(np.average(self.run_time), 2)

        print("\t\t\t\t | \t %s \t | \t %s \t | \t %s| \t %s \t | \t"
              .format() %
              ("Average objective",
               "Best Objective",
               "Improvement (%) ",
               "RunTime"))

        print("=" * 102)

        print("%10s | \t %10s \t\t | \t %10s \t\t | \t %10s \t | \t %10s  | "
              .format() %
              (
                  operator_name,
                  str(avg_obj_cost),
                  str(best_cost),
                  str(improvement),
                  str(avg_run_time)),
              end="\n")

        print('Solution')
        print(self.top10best_solution[0][0], end="\n")
        print("\n" * 3)

    def print_temp(self):
        print("Im here")
        n_model = len(self.temps)
        fig, axes = plt.subplots(1, n_model, figsize=(7 * n_model, 5), sharey=True, squeeze=False)

        for temp, ax in zip(self.temps, axes.flat):
            ax.plot(temp, label="Temperature change", color="c")
            ax.set_title("10k")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Temperature")
            ax.legend()

        plt.show()


def run_all(i):
    """
    Method used when running multithreading
    :ignore else
    :param i:
    :return:
    """
    m = Algorithms(file_list[i])
    op = Operators(m.problem)
    m.set_operators(op.one_insert, op.two_inserter, op.max_cost_swap)
    for i in range(10):
        m.sa()
    m.print_stats("Tuned Operators: ")
    # m.print_temp()


if __name__ == '__main__':
    # Single threaded application
    # Uncomment this to run single threaded
    # [run_all(i) for i in range(6)]

    # Multithreading:
    pool = mp.Pool(processes=6)
    pool.map(run_all, range(0, 6))
