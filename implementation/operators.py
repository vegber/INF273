import random
from utils_code.pdp_utils import *

import numpy as np


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


def one_insert(arr, vehicle, calls, vessel_cargo):
    sol_vehicle_arr = to_list_v2(arr, vehicle)  # to_list(arr)
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


def find_valid_feasable_placements(vehicle, vessel_comp):
    return [i for i, e in enumerate(vessel_comp[vehicle], 1) if e == 1.]


def two_exchange(arr, vehicle, calls, prob):
    return arr


def three_exchange(arr, vehicle, calls, prob):
    return arr


def to_list(arr):
    # [list(group) for key, group in groupby(L, lambda x: x == 0)]

    lst_string = "".join([str(x) for x in arr])
    lst2 = lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    var = [list(map(int, z)) for z in lst3]
    return var


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
