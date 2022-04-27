import random
import numpy as np


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


def sort_tuple(best_sol_cost_pairs):
    return sorted(best_sol_cost_pairs, key=lambda x: x[1])


def avg_Cost(sorted_10_best_solutions):
    return sum(x[1] for x in sorted_10_best_solutions) / len(sorted_10_best_solutions)


def get_init(vehicle: int, call: int):
    init = [0] * vehicle
    for i in range(1, call + 1):
        init.append(i)
        init.append(i)
    return init


def fill_2d_zero(two_dim):
    for x in range(len(two_dim)):
        if not two_dim[x]:
            two_dim[x].append(0)
        elif x != len(two_dim) - 1:
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


def list_format(sol_vehicle_arr, random_perm=False):
    arr = []
    for x in sol_vehicle_arr:
        if not x:
            arr.append(0)
        else:
            if random_perm:
                random.shuffle(x)
            for elem in x:
                arr.append(elem)
            arr.append(0)

    # edge case
    if arr[-1] == 0:
        arr.pop()
    return arr


def find_zero(arr, first_swap_index, second_valid_index):
    if arr[first_swap_index] == 0:
        return first_swap_index
    else:
        return second_valid_index


def extract_good_zero_swaps(arr, legal_zero_swap):
    return [x for x in legal_zero_swap if arr[x] != 0 and x != (len(arr) - 1)]


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


def place_at_insert_positions(call, car_index, insert_positions, arr):
    if len(insert_positions) > 0:
        arr.insert(insert_positions[0], call)  # inserts at call and to the right
        arr.insert(insert_positions[1], call)
    else:
        # no inserts found, place at the end
        arr.insert(car_index[-1], call)
        arr.insert(car_index[-1], call)
    return arr.copy()


def get_vehicle_indexes(org_arr_without_calls_to_move):
    car_index = [i for i, x in enumerate(org_arr_without_calls_to_move) if x == 0]
    return car_index


def get_sol_without_calls_to_move(calls, solution):
    base_solution = [x for x in solution if x not in calls]
    return base_solution


def get_upper_lower_bound(car_index, vehicle):
    """
    Returns a tuple of the valid insert start/end index of vehicle.

    :param car_index:
    :param vehicle:
    :return:
    """
    if vehicle == 0:
        return 0, 0
    else:
        return car_index[vehicle - 1] + 1, car_index[vehicle]


def remove_call(origin_delivery, origin_pickup, origin_vehicle, sol_vehicle_arr):
    sol_vehicle_arr[origin_vehicle].pop(origin_pickup)
    sol_vehicle_arr[origin_vehicle].pop(origin_delivery - 1)


def insert_call(car_to_place, random_call, sol_vehicle_arr):
    # insert pickup & delivery
    pickup_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
    sol_vehicle_arr[car_to_place].insert(pickup_index, random_call)
    delivery_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
    sol_vehicle_arr[car_to_place].insert(delivery_index, random_call)


def is_within_arrival_time(calls_length, current_time, route_travel_time, time_windows):
    """
    Checks if call's route time is on point or not. If exceed the arrival time, return
    false.
    :param calls_length:
    :param current_time:
    :param route_travel_time:
    :param time_windows:
    :return:
    """
    time = np.zeros(calls_length)
    for i in range(calls_length):
        time[i] = max((current_time + route_travel_time[i], time_windows[0, i]))
        if time[i] > time_windows[1, i]:
            return False
    return True


def get_zeroed_indexes(calls_to_change):
    return [i - 1 for i, e in enumerate(calls_to_change)]
