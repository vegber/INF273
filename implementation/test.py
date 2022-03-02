import random

from operators import find_valid_feasable_placements
from random_solution_driver import get_init
from utils_code.pdp_utils import *

path = '../utils_code/pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt'

problem = load_problem(path)
init_ = get_init(problem['n_vehicles'], problem['n_calls'])
sol = np.array([7, 7, 5, 5, 0, 2, 2, 0, 3, 4, 4, 3, 1, 1, 6, 6, 0])  # 6,6


def to_list(L):
    lst_string = "".join([str(x) for x in L])
    lst2 = [e + '0' for e in lst_string.split('0') if e]  # lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    lst4 = [list(map(str, z)) for z in lst3]
    lst4[-1].remove('0')
    va = list(map(int, ''.join([''.join(x) for x in lst4])))
    print(va)
    return va


def to_list2(L):
    # [list(group) for key, group in groupby(L, lambda x: x == 0)]
    lst_string = "".join([str(x) for x in L])
    lst2 = lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    return [list(map(int, z)) for z in lst3]


def one_insert(L, prob):
    # L to n + 1 lists
    var = (1, 2)
    print(bool(var))
    n_list = to_list2(L)
    print(n_list)
    print(problem['VesselCargo'])


def two_swap(arr, vehicle, calls, prob):
    vessel_cargo = problem['VesselCargo']
    vehicle_valid_calls = feasable_placements_with_outsorce(calls, vehicle, vessel_cargo)
    arr_2 = to_list2(arr)
    print(f"vehicle valid cars {vehicle_valid_calls}")

    first_swap = random.randint(0, calls)

    # find random car, to swap with
    # legal cars to swap with
    legal_cars = [x for x in range(len(vehicle_valid_calls)) if first_swap in vehicle_valid_calls[x]]

    print(f"random swap was {first_swap} and legal cars is then {legal_cars} ")

    # where is first swap:
    first_swap_indexes = [i for i, e in enumerate(arr) if e == first_swap]
    print(first_swap_indexes)

    # pick random valid place to swap with
    # create list of all available swaps in those cars
    cars = [0]  # can always swap with zero
    available_swaps = [cars.append(car) for x in legal_cars for car in arr_2[x] if car not in cars]
    print(cars)
    return arr


def fill_2d_zero(two_dim):
    for x in range(len(two_dim)):
        if not two_dim[x]:
            two_dim[x].append(0)


def feasable_placements_with_outsorce(calls, vehicle, vessel_cargo):
    vehicle_valid_calls = [find_valid_feasable_placements(x, vessel_cargo) for x in range(vehicle)]
    outsorcevehicle = [x for x in range(1, calls + 1)]
    vehicle_valid_calls.append(outsorcevehicle)
    return vehicle_valid_calls


def swap(arr, l1, l2):
    arr[l1], arr[l2] = arr[l2], arr[l1]
    return arr


if __name__ == '__main__':
    init = get_init(3, 7)

    print(two_swap(init, 3, 7, problem))
