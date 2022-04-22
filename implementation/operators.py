import os
import random

import numpy as np
from natsort import natsorted

from random_solution_driver import get_init
from utils_code.pdp_utils import *


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


def place_at_insert_positions(call, car_index, insert_positions, org_arr_without_calls_to_move):
    if len(insert_positions) > 0:
        org_arr_without_calls_to_move.insert(insert_positions[0], call)  # inserts at call and to the right
        org_arr_without_calls_to_move.insert(insert_positions[1], call)
    else:
        # no inserts found, place at the end
        org_arr_without_calls_to_move.insert(car_index[-1], call)
        org_arr_without_calls_to_move.insert(car_index[-1], call)
    return org_arr_without_calls_to_move.copy()


def get_vehicle_indexes(org_arr_without_calls_to_move):
    car_index = [i for i, x in enumerate(org_arr_without_calls_to_move) if x == 0]
    return car_index


def get_sol_without_calls_to_move(calls, solution):
    base_solution = [x for x in solution if x not in calls]
    return base_solution


def get_upper_lower_bound(car_index, vehicle):
    """
    Returns a tuple of the valid insert start/end index of vehicle
    :param car_index:
    :param vehicle:
    :return:
    """
    if vehicle == 0:
        lower_bound, upper_bound = 0, 0
    else:
        lower_bound, upper_bound = car_index[vehicle - 1] + 1, car_index[vehicle]
    return lower_bound, upper_bound


def remove_call(origin_delivery, origin_pickup, origin_vehicle, sol_vehicle_arr):
    sol_vehicle_arr[origin_vehicle].pop(origin_pickup)
    sol_vehicle_arr[origin_vehicle].pop(origin_delivery - 1)


def insert_call(car_to_place, random_call, sol_vehicle_arr):
    # insert pickup & delivery
    pickup_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
    sol_vehicle_arr[car_to_place].insert(pickup_index, random_call)
    delivery_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
    sol_vehicle_arr[car_to_place].insert(delivery_index, random_call)


class Operators:
    def __init__(self, loaded_problem):
        self.vehicle = loaded_problem['n_vehicles']
        self.calls = loaded_problem['n_calls']
        self.vessel_cargo = loaded_problem['VesselCargo']
        self.cargo = loaded_problem['Cargo']
        self.travel_time = loaded_problem['TravelTime']
        self.first_travel_time = loaded_problem['FirstTravelTime']
        self.vessel_capacity = loaded_problem['VesselCapacity']
        self.loading_time = loaded_problem['LoadingTime']
        self.unloading_time = loaded_problem['UnloadingTime']
        self.TravelCost = loaded_problem['TravelCost']
        self.FirstTravelCost = loaded_problem['FirstTravelCost']
        self.PortCost = loaded_problem['PortCost']

    def one_insert(self, arr):
        """
        One insert operator:
        inputs a one dim array, outputs a one dim arr.
        Simply takes random vehicles and puts them in another semi - random vehicle (valid cars)
        :param arr:
        :return:
        """
        sol_vehicle_arr = self.to_list_v2(arr)
        random_vehicle = random.randint(0, self.vehicle)  # zero indexed

        valid_placements = self.get_valid_calls(random_vehicle)

        random_call = random.choice(valid_placements)

        # check if random vehicle is "itself"
        origin_vehicle, origin_pickup, origin_delivery = find_indexes(sol_vehicle_arr, random_call)
        """
        Two cases: 
        if call to new vehicle - delivery must also follow
    
        if call to same vehicle - only change index of pickup 
        """
        # if call is to be places in same vehicle - change index
        if random_vehicle == origin_vehicle:  # call in same vehicle - change index
            # only change index of pickup
            sol_vehicle_arr[random_vehicle].pop(origin_pickup)  # remove origin pickup

            # find new pickup index
            index = random.randint(0, len(sol_vehicle_arr[random_vehicle]))
            sol_vehicle_arr[random_vehicle].insert(index, random_call)

        else:  # call not in same vehicle
            # remove pickup
            remove_call(origin_delivery, origin_pickup, origin_vehicle, sol_vehicle_arr)

            # insert pickup & delivery
            insert_call(random_vehicle, random_call, sol_vehicle_arr)

        # change sol_vehicle_arr back to normal form
        arr = list_format(sol_vehicle_arr)
        return arr

    def change_car_insert(self, arr, car):
        """
        One insert operator:
        Simply takes random vehicles and puts them in another semi - random vehicle (valid cars)
        Will only swap calls to valid "places" / cars
        :param car: car to remove call from
        :param arr:
        :return:
        """
        sol_vehicle_arr = self.to_list_v2(arr)

        cars_ = [i for i, x in enumerate(sol_vehicle_arr)]
        cars_.remove(car)

        car_to_place = random.choice(cars_)

        valid_placements = self.get_valid_calls(car_to_place)

        random_call = random.choice(valid_placements)

        # check if random vehicle is "itself"
        origin_vehicle, origin_pickup, origin_delivery = find_indexes(sol_vehicle_arr, random_call)
        remove_call(origin_delivery, origin_pickup, origin_vehicle, sol_vehicle_arr)
        insert_call(car_to_place, random_call, sol_vehicle_arr)

        # change sol_vehicle_arr back to normal form
        arr = list_format(sol_vehicle_arr)
        return arr

    def max_cost_swap(self, arr):
        """
        Naive operator. Uses a custom cost function to find the vehicle with most
        costly calls. Idea is to spread cost evenly.
        :param arr:
        :return:
        """
        sol_vehicle_arr = self.to_list_v2(arr)
        car_cost = self.car_cost(sol_vehicle_arr)
        car_to_change = car_cost.index(max(car_cost)) - 1  # select vehicle with the highest cost not zero indexed.
        out = self.change_car_insert(arr, car_to_change)
        return out

    def smarter_insert(self, arr):
        return self.k_insert(arr, k_val=1)

    def k_insert(self, solution, k_val=random.choice([2, 3, 4])):

        calls = random.sample(range(1, self.calls + 1), k=k_val)
        org_arr_without_calls_to_move = get_sol_without_calls_to_move(calls, solution)

        for call in calls:
            car_index = get_vehicle_indexes(org_arr_without_calls_to_move)
            compatible_vehicles = self.get_compatible_vehicle(call)
            random.shuffle(compatible_vehicles)  # add a layer of random - ness
            insert_positions = self.find_best_pos(org_arr_without_calls_to_move, call, car_index, compatible_vehicles)
            place_at_insert_positions(call, car_index, insert_positions, org_arr_without_calls_to_move)

        arr = org_arr_without_calls_to_move.copy()
        return arr

    def car_cost(self, arr):
        """
        Input two dim per car
        :param arr:
        :return:
        """
        return [self.call_cost(i, x) for i, x in enumerate(arr)]

    def feasible_vehicle(self, car, calls_to_place):
        calls_length = len(calls_to_place)
        if not calls_length: return True  # can apply if not available car

        load_capacity, current_time, total = 0, 0, 0
        calls = calls_to_place.copy()
        calls = [x - 1 for x in calls]  # zero index calls
        index = [i for i, e in enumerate(calls)]
        load_capacity -= self.cargo[calls, 2]  # matrix (y, x) fetch cargo at call
        load_capacity[::2] = self.cargo[calls[::2], 2]  # eval calls
        # check if any vessel is overloaded
        if not np.any(self.vessel_capacity[car] - [total := total + x for x in load_capacity[index]] < 0):
            time_windows = np.zeros((2, calls_length))
            time_windows[0] = self.cargo[calls, 6]
            time_windows[0, ::2] = self.cargo[calls[::2], 4]
            time_windows[1] = self.cargo[calls, 7]
            time_windows[1, ::2] = self.cargo[calls[::2], 5]
            time_windows = time_windows[:, index]

            port_index = self.cargo[calls, 1].astype(int)
            port_index[::2] = self.cargo[calls[::2], 0]
            port_index = port_index[index] - 1

            # Todo port cost

            loading_time = self.unloading_time[car, calls]
            loading_time[::2] = self.loading_time[car, calls[::2]]
            loading_time = loading_time[index]
            travel_time_port = self.travel_time[car, port_index[:-1], port_index[1:]]
            first_visit_time = self.first_travel_time[car, int(self.cargo[calls[0], 0] - 1)]
            route_travel_time = np.hstack((first_visit_time, travel_time_port.flatten()))
            arrive_time = np.zeros(calls_length)
            return self.is_within_arrival_time(arrive_time,
                                               calls_length,
                                               current_time,
                                               loading_time,
                                               route_travel_time,
                                               time_windows)
        return False

    def is_within_arrival_time(self, arrive_time, calls_length, current_time, lu_time, route_travel_time, time_windows):
        for i in range(calls_length):
            arrive_time[i] = np.max((current_time + route_travel_time[i], time_windows[0, i]))
            if arrive_time[i] > time_windows[1, i]:
                return False
            current_time = arrive_time[i] + lu_time[i]
        return True

    def call_cost(self, car, route):
        """
        Modifiable cost function. Returns the cost current calls.
        Uses the cargo, travel cost, FirstTravelCost, PortCost from the problem.
        :param route:
        :return:
        """
        car = car - 1

        if len(route) == 0:
            return 0

        calls_to_change = route.copy()
        calls_to_change = np.sort([x - 1 for x in calls_to_change])
        calls_to_change_indexes = [i for i, e in enumerate(calls_to_change)]

        cargo_matrix = self.cargo[calls_to_change, 1].astype(int)
        cargo_matrix[::2] = self.cargo[calls_to_change[::2], 0]  # step - wise two and two fetch at cargo list.comp
        cargo_matrix = cargo_matrix[calls_to_change_indexes] - 1

        travel_cost_at_curr_car = self.TravelCost[car, cargo_matrix[:-1], cargo_matrix[1:]]
        FirstVisitCost = self.FirstTravelCost[car, (self.cargo[calls_to_change[0], 0] - 1).astype(int)]
        return np.sum(np.hstack((FirstVisitCost, travel_cost_at_curr_car.flatten()))) + np.sum(
            self.PortCost[car, calls_to_change]) / 2

    def find_best_pos(self, base_solution, call, car_index, compatible_vehicles):
        """
        Iterates over compatible vehicles.

        :param base_solution: solution without calls to change:
        :param call: current call to change :
        :param car_index: start index of cars :
        :param compatible_vehicles: valid cars to place call :
        :return:
        """

        insert_position = []
        for vehicle in compatible_vehicles:
            lower_bound, upper_bound = get_upper_lower_bound(car_index, vehicle)
            valid_car_indexes = base_solution[lower_bound:upper_bound].copy()

            best_position = (-1, -1)
            min_cost = -1

            for i in range(len(valid_car_indexes)):
                first_insert_route = valid_car_indexes.copy()
                first_insert_route.insert(i, call)
                for j in range(i, len(valid_car_indexes)):
                    second_insert_route = first_insert_route.copy()
                    second_insert_route.insert(j, call)
                    if not self.feasible_vehicle(vehicle, second_insert_route):
                        continue

                    cost = self.call_cost(vehicle, second_insert_route)
                    if cost < min_cost:
                        min_cost = cost
                        best_position = (i, j)

            if len(valid_car_indexes) > 0 and best_position == (-1, -1):
                continue
            elif len(valid_car_indexes) == 0:
                best_position = (0, 0)

            insert_position = (lower_bound + best_position[0], lower_bound + best_position[1])
            break
        return insert_position

    def get_compatible_vehicle(self, call):
        return [i for i in range(self.vehicle) if self.vessel_cargo[i, call - 1]]

    def get_valid_calls(self, random_vehicle):
        """
        :param random_vehicle:
        :return:
        """
        outsource = None
        if random_vehicle == self.vehicle:  # cargo will be outsourced
            outsource = True
        if not outsource:  # get valid calls that can be places in this vehicle
            valid_placements = self.get_valid_feasible_placements(random_vehicle)
        else:
            valid_placements = [x for x in range(1, self.calls + 1)]  # outsource can take all calls
        return valid_placements

    def two_exchange(self, arr, ):
        return self.swap_op(arr, 2)

    def three_exchange(self, arr, ):
        return self.swap_op(arr, 3)

    def swap_op(self, arr, swap_count):
        arr_2 = fill_2d_zero(self.to_list_v2(arr))
        for i in range(swap_count):
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
                cycle = cycle[0: 2]  # random.randrange(0, len(cycle), 2)]
                call = cycle[0]

                valid_cars = []
                for x in range(self.vehicle):
                    curr = self.get_valid_feasible_placements(x)
                    if call in curr:
                        valid_cars.append(x)

                random_car = random.choice(valid_cars)

                while random_car == vehicle_most_call:
                    random_car = random.randint(0, self.vehicle)

                for call in cycle:
                    arr_2[random_car].insert(0, call)

                arr_2[vehicle_most_call] = arr_2[vehicle_most_call][len(cycle):]
                arr = [y for x in arr_2 for y in x]
        return arr

    def to_list_v2(self, arr):
        """
        Converts a one dimensional array to a
        two dimensional list

        problem specific method.
        Todo runtime
        :param arr:
        :return:
        """
        out = [[]] * (self.vehicle + 1)
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

    def get_valid_feasible_placements(self, x):
        """
        returns valid indexes of call (must be iterated - vessel cargo is 2d - arr)

        :param x:
        :return:
        """
        return [i for i, e in enumerate(self.vessel_cargo[x], 1) if e == 1.]

    def feasible_placements_with_outsource(self):
        vehicle_valid_calls = [self.get_valid_feasible_placements(x) for x in range(self.vehicle)]
        outsourcing = [x for x in range(1, self.calls + 1)]
        vehicle_valid_calls.append(outsourcing)
        return vehicle_valid_calls


"""path = '../utils_code/pdp_utils/data/pd_problem/'
file_list = natsorted(os.listdir(path), key=lambda y: y.lower())

sol = [23, 23, 1, 1, 0, 11, 11, 17, 17, 0, 16, 16, 24, 24, 5, 5, 2, 2, 31, 31, 0, 6, 6, 13, 13, 0, 8, 8, 26, 32, 32, 26,
       0, 14, 27, 14, 27, 0, 35, 35, 10, 10, 0, 25, 33, 7, 19, 15, 18, 3, 30, 25, 18, 21, 15, 9, 20, 22, 20, 3, 34, 28,
       4, 29, 29, 4, 12, 19, 28, 33, 21, 12, 34, 9, 22, 30, 7]
prob = load_problem(path + file_list[0])
# a, b = feasibility_check(sol, prob)

op = Operators(prob)

output = op.k_insert(get_init(3, 7))

print(output)
"""
