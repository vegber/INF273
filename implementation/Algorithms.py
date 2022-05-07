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
        self.operators = []
        self.operator_sum = [0] * 3
        self.operator_weight = [33, 33, 33]  # init equal weights for all
        self.solution_log = []
        self.current_operator = None
        self.R = 0.8
        self.operator_count = [0] * 3

    def set_operators(self, operators):
        [self.operators.append(x) for x in operators]

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
        # return random.choice(choices)
        index = random.choices([x for x in range(len(self.operators))], weights=self.operator_weight)[0]
        self.set_current_operator(index)
        return self.operators[index](obj)

    def get_escape_operator(self, arr) -> Operators:
        # TODO
        # change this to the most diversifying operator
        # Currently: "mostly: costing car"
        return self.operators[2](arr)

    def sa(self):
        s_0 = get_init(self.vehicle, self.calls)  # generate init solution
        fin_temp = 0.2  # 0.1
        incumbent = s_0  # s_best <- s_0
        best_solution = s_0
        delta_W = []
        start = time.time()
        self.solution_log.append(incumbent)
        # adaptiveness to warmup?
        while len(delta_W) == 0 or sum(delta_W) == 0:
            for w in range(100):
                new_sol = self.operators[0](best_solution)  # self.get_op(best_solution)
                delta_E = cost_function(new_sol, self.problem) - cost_function(incumbent, self.problem)
                passed, cause = feasibility_check(new_sol, self.problem)
                if passed and delta_E < 0:
                    self.solution_log.append(new_sol)
                    incumbent = new_sol
                    if cost_function(incumbent, self.problem) < cost_function(best_solution, self.problem):
                        best_solution = incumbent
                elif passed:
                    self.solution_log.append(new_sol)
                    if random.random() < 0.8:  # 0.8
                        incumbent = new_sol
                    delta_W.append(delta_E)
        delta_AVG = np.average(delta_W)
        T_0 = (-delta_AVG) / np.log(0.8)
        alfa = pow(fin_temp / T_0, 1 / 19900)
        # print(alfa)
        T = T_0
        temps = []
        iterations_since_best_sol = 0
        escape_condition = 400  # 500
        delta_escape = 0  # don't get stuck in a forever escape loop
        escape_counter = 0
        for e in range(1, 9900):
            temps.append(T)
            # segments of 100 -- update the weights
            if e % 300 == 0:
                self.update_weights()

            if iterations_since_best_sol >= escape_condition and delta_escape <= 50:
                # escape this local minima
                new_sol = self.get_escape_operator(incumbent)
                delta_escape += 1
                escape_counter += 1
            else:
                new_sol = self.get_op(incumbent)
                delta_escape = 0  # zero out the escape counter

            delta_E = cost_function(new_sol, self.problem) - cost_function(incumbent, self.problem)
            feasible, _ = feasibility_check(new_sol, self.problem)

            if new_sol not in self.solution_log:
                self.solution_log.append(new_sol)
                self.give_operator_points(1)

            if feasible and delta_E < 0:
                incumbent = new_sol
                # found valid solution, and it was better than the previous
                # 2 points
                self.give_operator_points(2)
                if cost_function(incumbent, self.problem) < cost_function(best_solution, self.problem):
                    # New best solution! Give four points to operator
                    best_solution = incumbent
                    iterations_since_best_sol = 0
                    self.give_operator_points(5)
            elif feasible and random.random() < pow(math.e, (-delta_E / T)):
                incumbent = new_sol
            T = alfa * T
            iterations_since_best_sol += 1
        # print(f"Escape algorithm ran: {escape_counter} times")
        # print(f"\nWeights are {''.join(str(self.operator_weight))}")
        self.run_time.append(time.time() - start)
        self.top10best_solution.append((best_solution, cost_function(best_solution, self.problem)))
        self.temps.append(temps)

    def give_operator_points(self, points: int) -> None:
        self.operator_sum[self.operators.index(self.current_operator)] += points

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
        n_model = len(self.temps)
        fig, axes = plt.subplots(1, n_model, figsize=(7 * n_model, 5), sharey=True, squeeze=False)

        for temp, ax in zip(self.temps, axes.flat):
            ax.plot(temp, label="Temperature change", color="c")
            ax.set_title(self.problem_name)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Temperature")
            ax.legend()

        plt.show()

    def update_weights(self):
        # Score / antall ganger  = X
        # (1-R)+R*(score_i/antall_ganger_i)
        self.operator_weight = [round((i / (sum(self.operator_sum))) * 100) for i in self.operator_sum]

    def set_current_operator(self, index):
        self.current_operator = self.operators[index]


def run_all(i):
    """
    Method used when running multithreading
    :ignore else
    :param i:
    :return:
    """
    m = Algorithms(file_list[i])
    if i == 0:
        op = Operators(m.problem, 0)
    elif i == 1:
        op = Operators(m.problem, 0)
    elif i == 2:
        op = Operators(m.problem, 0)
    elif i == 3:
        op = Operators(m.problem, 0.1)
    elif i == 4:
        op = Operators(m.problem, 0.1)
    else:
        op = Operators(m.problem)
    m.set_operators([op.one_insert,
                     op.smarter_one_insert,
                     op.max_cost_swap
                     ])
    for i in range(10):
        m.sa()
    m.print_stats("Tuned Op: \t\t")
    # m.print_temp()


if __name__ == '__main__':
    # Single threaded application
    # Uncomment this to run single threaded
    # [run_all(i) for i in range(6)]

    # Multithreading:
    pool = mp.Pool(processes=5)
    pool.map(run_all, range(0, 5))
