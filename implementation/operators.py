import random
from utils_code.pdp_utils import *
import numpy as np


def one_insert(arr, vehicle, calls, vessel_cargo):
    sol_vehicle_arr = to_list_v2(arr, vehicle)
    random_vehicle = random.randint(0, vehicle)  # zero indexed

    outsource = False

    if random_vehicle == vehicle:  # cargo will be outsourced
        outsource = True

    if not outsource:
        valid_placements = find_valid_feasable_placements(random_vehicle, vessel_cargo)
    else:
        valid_placements = [x for x in range(1, calls + 1)]  # outsource can take all calls

    random_call = random.choice(valid_placements)

    # check if random vehicle is "itself"
    origin_vehicle, origin_pickup, origin_delivery = find_indexes(sol_vehicle_arr, random_call)
    """
    Two cases: 
    if call to new vehicle - delivery must also follow

    if call to same vehicle - only change index of pickup """
    # if call is to be places in same vehicle - change index
    if random_vehicle == origin_vehicle:  # call in same vehicle - change index
        # only change index of pickup
        sol_vehicle_arr[random_vehicle].pop(origin_pickup)  # remove origin pickup

        # find new pickup index
        index = random.randint(0, len(sol_vehicle_arr[random_vehicle]))
        sol_vehicle_arr[random_vehicle].insert(index, random_call)

    else:  # call not in same vehicle
        # remove pickup
        sol_vehicle_arr[origin_vehicle].pop(origin_pickup)
        sol_vehicle_arr[origin_vehicle].pop(origin_delivery - 1)

        # insert pickup & delivery
        pickup_index = random.randint(0, len(sol_vehicle_arr[random_vehicle]))
        sol_vehicle_arr[random_vehicle].insert(pickup_index, random_call)

        delivery_index = random.randint(0, len(sol_vehicle_arr[random_vehicle]))
        sol_vehicle_arr[random_vehicle].insert(delivery_index, random_call)

    # change sol_vehicle_arr back to normal form
    arr = list_format(sol_vehicle_arr)
    return arr


def two_exchange(arr, vehicle, calls, vessel_cargo):
    arr_2 = fill_2d_zero(to_list_v2(arr, vehicle))
    vehicle_most_call = ([len(arr_2[i]) for i in range(len(arr_2))])
    vehicle_most_call = vehicle_most_call.index(max(vehicle_most_call))

    for i in range(2):
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
            random_car = random.randint(0, vehicle)
            while random_car == vehicle_most_call:
                random_car = random.randint(0, vehicle)

            for call in cycle:
                arr_2[random_car].insert(0, call)

            arr_2[vehicle_most_call] = arr_2[vehicle_most_call][len(cycle):]
            arr = [y for x in arr_2 for y in x]

    return arr



def get_index_1d(arr, first_swap_value):
    var = [i for i, e in enumerate(arr) if e == first_swap_value]
    # should always be two
    return var[0], var[1]


def swap(arr, l1, l2):
    arr[l1], arr[l2] = arr[l2], arr[l1]
    return arr


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
    :returns exact index of possible sol. remember
    to add +1 in your insert method.
    :param arr:
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


def three_exchange(arr, vehicle, calls, prob):
    return arr


def to_list_v2(arr, vehicle):
    out = [[] * x for x in range(vehicle + 1)]  # change six by vehicle +1
    counter = 0

    L = list(map(str, arr))
    content = ""

    for elem in range(len(L)):
        if L[elem] == "0":
            out[counter] = list(content.split())
            content = ""
            counter += 1
        else:
            content += L[elem] + ' '
    out[counter] = list(content.split())

    for outer in range(len(out)):
        for inner in range(len(out[outer])):
            if out[outer][inner] == '':
                out.pop(outer)
                out.insert(outer, [])
            else:
                out[outer][inner] = int(out[outer][inner])

    return out


def find_valid_feasable_placements(vehicle, vessel_comp):
    return [i for i, e in enumerate(vessel_comp[vehicle], 1) if e == 1.]


def feasable_placements_with_outsorce(calls, vehicle, vessel_cargo):
    vehicle_valid_calls = [find_valid_feasable_placements(x, vessel_cargo) for x in range(vehicle)]
    outsorcevehicle = [x for x in range(1, calls + 1)]
    vehicle_valid_calls.append(outsorcevehicle)
    return vehicle_valid_calls


def fill_2d_zero(two_dim):
    for x in range(len(two_dim)):
        if not two_dim[x]:
            two_dim[x].append(0)

    return two_dim


def find_indexes(sol_vehicle_arr, random_call):
    indexes = 0
    vehicle = 0
    for x in range(len(sol_vehicle_arr)):
        var = tuple(i for i, e in enumerate(sol_vehicle_arr[x]) if e == random_call)
        if var:
            vehicle = x  # vehicle is not zero index
            indexes = var
    return vehicle, indexes[0], indexes[1]


def list_format(sol_vehicle_arr):
    arr = []
    for x in sol_vehicle_arr:
        if not x:
            arr.append(0)
        else:
            for elem in x:
                arr.append(elem)
            arr.append(0)

    # edge case
    if arr[-1] == 0:
        arr.pop()
    return arr


def extract_good_zero_swaps(arr, legal_zero_swap):
    return [x for x in legal_zero_swap if arr[x] != 0 and x != (len(arr) - 1)]
