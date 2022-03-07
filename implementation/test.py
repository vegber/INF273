import random

from operators import find_valid_feasable_placements, to_list_v2
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
            car_index.append(car_count)
        if arr[x] == 0:
            car_count += 1
    return car_index[0], car_index[1]


def find_zero(arr, first_swap_index, second_valid_index):
    if arr[first_swap_index] == 0:
        return first_swap_index
    else:
        return second_valid_index


def find_zero_swaps(arr):
    """
    returns exact index of possible sol. remember
    to add +1 in your insert method.
    param arr
    :return:
    """
    backlog = []
    valid_pos = []
    for i, elem in enumerate(arr):
        if elem == 0:
            valid_pos.append(i)
            continue
        backlog.append(elem)
        if backlog.count(elem) == 2:
            backlog.remove(elem)
            backlog.remove(elem)
        if len(backlog) == 0:
            valid_pos.append(i)

    return valid_pos


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

    if first_swap_value and second_swap_value != 0:
        print("1" * 120)
        # both values are not zero, we can do swap
        pickup_first, deliver_first = get_index_1d(arr, first_swap_value)
        pickup_sec, deliver_sec = get_index_1d(arr, second_swap_value)
        arr = swap(arr, pickup_first, pickup_sec)
        arr = swap(arr, deliver_first, deliver_sec)

    elif first_swap_value == 0 and second_swap_value == 0:
        print("=" * 120)
        # swap cars
        car1, car2 = which_cars(arr, first_swap_index, second_valid_index)
        arr_2 = swap(arr_2, car1, car2)
        arr = [y for x in arr_2 for y in x]

    elif first_swap_value == 0 or second_swap_value == 0:  # at least one is zero'
        print("3" * 120)
        zero_to_be_swapped_index = find_zero(arr, first_swap_index, second_valid_index)  # find out which one is zero
        valid_indexes = find_zero_swaps(arr)  # returns a list of possible insertions
        arr.pop(zero_to_be_swapped_index)
        print(arr)
        random_choice_index = random.choice(valid_indexes)
        arr.insert(random_choice_index + 1, 0)

    return arr

    # 0 0 0 0 0 0 0 0 1,1 0 , 2,2,


def fill_2d_zero(two_dim):
    for x in range(len(two_dim)):
        if not two_dim[x]:
            two_dim[x].append(0)
        elif x != len(two_dim) - 1:
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


def get_zero_indexes(arr):
    return [i for i, e in enumerate(arr) if e == 0]


def two_swap_v2(arr, vehicle, calls, vessel_cargo):
    legal_zero_swap = find_zero_swaps(arr)
    cycles = extract_good_zero_swaps(arr, legal_zero_swap)
    zero_index = get_zero_indexes(arr)
    delta = random.randint(0, 10)

    for x in range(2):
        if len(cycles) == 0:  # there is nowhere to place a zero - we have no cycles
            # we should swap elements within vehicle
            # swap two
            print(arr)
            arr_2 = fill_2d_zero(to_list_v2(arr, vehicle))
            vehicle_most_call = ([len(arr_2[i]) for i in range(len(arr_2))])
            vehicle_most_call = vehicle_most_call.index(max(vehicle_most_call))
            arr_2[vehicle_most_call] = swap(arr_2[vehicle_most_call],
                                            random.choice(arr_2[vehicle_most_call]),
                                            random.choice(arr_2[vehicle_most_call]))

            arr = [y for x in arr_2 for y in x]

        else:  # swap / insert zero
            to_insert = random.choice(cycles) + 1
            print(to_insert)
            arr.pop(random.choice(zero_index))
            arr.insert(to_insert, 0)
    return arr


def valid_car(elem, calls, vehicles, vessel_cargo):
    var = (feasable_placements_with_outsorce(calls, vehicles, vessel_cargo))


def two_swap_v3(arr, vehicle, calls, vessel_cargo):
    arr_2 = fill_2d_zero(to_list_v2(arr, vehicle))

    for i in range(2):
        vehicle_most_call = ([len(arr_2[i]) for i in range(len(arr_2))])
        vehicle_most_call = vehicle_most_call.index(max(vehicle_most_call))
        legal_zero_swap = find_zero_swaps(arr_2[vehicle_most_call])
        cycles = extract_good_zero_swaps(arr_2[vehicle_most_call], legal_zero_swap)

        # if cycles i none, then continue to swap elements
        if len(cycles) == 0:
            arr_2[vehicle_most_call] = swap(arr_2[vehicle_most_call],
                                            random.choice(arr_2[vehicle_most_call]),
                                            random.choice(arr_2[vehicle_most_call])
                                            )
            arr = [y for x in arr_2 for y in x]

        else:
            cycle = arr_2[vehicle_most_call][:cycles[-1] + 1]
            cycle = cycle[0: random.randrange(0, len(cycle), 2)]

            random_car = random.randint(0, vehicle)
            # random_car = valid_car(random.choice(cycle), calls, vehicle, vessel_cargo)
            while random_car == vehicle_most_call:
                random_car = random.randint(0, vehicle)

            for call in cycle:
                arr_2[random_car].insert(0, call)

            arr_2[vehicle_most_call] = arr_2[vehicle_most_call][len(cycle):]
            arr = [y for x in arr_2 for y in x]

    return arr


def extract_good_zero_swaps(arr, legal_zero_swap):
    return [x for x in legal_zero_swap if arr[x] != 0]  # and x != (len(arr) - 1)]


if __name__ == '__main__':
    init = get_init(3, 7)
    # init = [1, 1, 0, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

    print(init)
    out = two_swap_v3(init, 3, 7, problem)
    for x in range(10000):
        out = two_swap_v3(out, 3, 7, problem)

    print(out)
    # print(out)
    #    one, two = feasibility_check(out, problem)
