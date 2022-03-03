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


def get_index_1d(arr, first_swap_value):
    var = [i for i, e in enumerate(arr) if e == first_swap_value]
    # should always be two
    return var[0], var[1]


def which_cars(arr, first_swap_index, second_valid_index):
    car_index = []
    car_count = 0
    for x in range(len(arr)):
        if x == first_swap_index or x == second_valid_index:
            print("yes")
            car_index.append(car_count)
        if arr[x] == 0:
            car_count += 1
    print(car_index)
    print(f"indexes in car {car_index[0]} and {car_index[1]}")
    return car_index[0], car_index[1]


def two_swap(arr, vehicle, calls, prob):
    vessel_cargo = problem['VesselCargo']
    vehicle_valid_calls = feasable_placements_with_outsorce(calls, vehicle, vessel_cargo)
    arr_2 = fill_2d_zero(to_list2(arr))  # can have in case I need it

    first_swap_index = random.randint(0, len(arr) - 1)
    first_swap_value = arr[first_swap_index]

    # so now, this value can be swapped with all calls except zero
    second_valid_index = random.randint(0, len(arr) - 1)
    while second_valid_index == first_swap_index:
        second_valid_index = random.randint(0, len(arr) - 1)

    second_swap_value = arr[second_valid_index]
    print(first_swap_value)
    print(second_swap_value)
    print(f"before: {arr}")
    first_swap_value = 0
    second_swap_value = 0
    if first_swap_value and second_swap_value != 0:
        # both values are not zero, we can do swap
        pickup_first, deliver_first = get_index_1d(arr, first_swap_value)
        pickup_sec, deliver_sec = get_index_1d(arr, second_swap_value)
        arr = swap(arr, pickup_first, pickup_sec)
        arr = swap(arr, deliver_first, deliver_sec)
        print(f"after: {arr}")
    if (first_swap_value and second_swap_value) == 0:
        # todo
        # swap cars
        # find out car indexes, swap() --- use 2d arr
        car1, car2 = which_cars(arr, first_swap_index, second_valid_index)
        pass
    else:  # atleast one is zero

        pass
    return arr

    # 0 0 0 0 0 0 0 0 1,1 0 , 2,2,


def fill_2d_zero(two_dim):
    for x in range(len(two_dim)):
        if not two_dim[x]:
            two_dim[x].append(0)

    return two_dim


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
