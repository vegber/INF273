import random
from collections.abc import Iterable
import numpy as np
vehicle = 5
call = 18


def get_permutation(vehicle: int, call: int):
    return np.random.permutation([0] * vehicle + list(range(1, call)) * 2)


def get_permutation(vehicle: int, call: int):
    actual_solution = []
    solution_space = []
    available_calls = []

    available_calls.extend(range(1, call + 1))
    cars = 3  # random.randint(0, vehicle)

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


for x in range(10):
    print(get_permutation(vehicle, call), end="\n")