import os
import random
from natsort import natsorted
import copy
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


def find_vehicle_positions(base_solution):
    """
    Find vehicle positions
    :param base_solution:
    :return:
    """
    return [i for i, x in enumerate(base_solution) if x == 0]


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

    def change_car_insert(self, arr, car):
        """
        One insert operator:
        inputs a one dim array, outputs a one dim arr.
        Simply takes random vehicles and puts them in another semi - random vehicle (valid cars)
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
        # if call is to be places in same vehicle - change index
        sol_vehicle_arr[origin_vehicle].pop(origin_pickup)
        sol_vehicle_arr[origin_vehicle].pop(origin_delivery - 1)

        # insert pickup & delivery
        pickup_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
        sol_vehicle_arr[car_to_place].insert(pickup_index, random_call)

        delivery_index = random.randint(0, len(sol_vehicle_arr[car_to_place]))
        sol_vehicle_arr[car_to_place].insert(delivery_index, random_call)

        # change sol_vehicle_arr back to normal form
        arr = list_format(sol_vehicle_arr)
        return arr

    def one_insert_v2(self, arr):
        """
        NAIVE !!!!
        type cost for each vehicle, modify one insert to change curr max vehicle
        :param arr:
        :return:
        """
        sol_vehicle_arr = self.to_list_v2(arr)
        # find which car has most cost
        # best_model = models[(model_score.index(max(model_score)))]
        car_cost = self.car_cost(sol_vehicle_arr)
        car_to_change = car_cost.index(max(car_cost))
        # Todo
        # Call one insert to change this vehicle
        out = self.change_car_insert(arr, car_to_change)
        # print(f"Cargo {[len(x) for x in self.cargo]}")
        # print(f"Vessel Cargo: {self.vessel_cargo}")
        # print(f"Vessel cap: {self.vessel_capacity}")
        return out

    def car_cost(self, arr):
        """
        Input two dim per car
        :param arr:
        :return:
        """
        return [self.route_cost(x) for x in arr]

    def feasible_vehicle(self, route):
        len_route = len(route)

        if len_route == 0:
            return True

        load_size, current_time = 0, 0
        route_aux = route.copy()
        route_aux = [x - 1 for x in route_aux]
        sorted_route = np.sort(route_aux, kind='mergesort')
        I = np.argsort(route_aux, kind='mergesort')
        index = np.argsort(I, kind='mergesort')

        load_size -= self.cargo[sorted_route, 2]
        load_size[::2] = self.cargo[sorted_route[::2], 2]
        load_size = load_size[index]
        if np.any(self.vessel_capacity[self.vehicle - 1] - np.cumsum(load_size) < 0):
            return False

        time_windows = np.zeros((2, len_route))
        time_windows[0] = self.cargo[sorted_route, 6]
        time_windows[0, ::2] = self.cargo[sorted_route[::2], 4]
        time_windows[1] = self.cargo[sorted_route, 7]
        time_windows[1, ::2] = self.cargo[sorted_route[::2], 5]
        time_windows = time_windows[:, index]

        port_index = self.cargo[sorted_route, 1].astype(int)
        port_index[::2] = self.cargo[sorted_route[::2], 0]
        port_index = port_index[index] - 1

        lu_time = self.unloading_time[self.vehicle - 1, sorted_route]
        lu_time[::2] = self.loading_time[self.vehicle - 1, sorted_route[::2]]
        lu_time = lu_time[index]
        diag = self.travel_time[self.vehicle - 1, port_index[:-1], port_index[1:]]
        first_visit_time = self.first_travel_time[self.vehicle - 1, int(self.cargo[route_aux[0], 0] - 1)]
        route_travel_time = np.hstack((first_visit_time, diag.flatten()))

        arrive_time = np.zeros(len_route)
        for i in range(len_route):
            arrive_time[i] = np.max((current_time + route_travel_time[i], time_windows[0, i]))
            if arrive_time[i] > time_windows[1, i]:
                return False
            current_time = arrive_time[i] + lu_time[i]

        return True

    def route_cost(self, route):
        if len(route) == 0:
            return 0

        currentVPlan = route.copy()
        currentVPlan = [x - 1 for x in currentVPlan]

        sortRout = np.sort(currentVPlan, kind='mergesort')
        I = np.argsort(currentVPlan, kind='mergesort')
        Indx = np.argsort(I, kind='mergesort')

        PortIndex = self.cargo[sortRout, 1].astype(int)
        PortIndex[::2] = self.cargo[sortRout[::2], 0]
        PortIndex = PortIndex[Indx] - 1

        Diag = self.TravelCost[self.vehicle - 1, PortIndex[:-1], PortIndex[1:]]

        FirstVisitCost = self.FirstTravelCost[self.vehicle - 1, int(self.cargo[currentVPlan[0], 0] - 1)]
        RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
        CostInPorts = np.sum(self.PortCost[self.vehicle - 1, currentVPlan]) / 2

        return RouteTravelCost + CostInPorts

    def smarter_insert(self, arr):
        """

        :param arr:
        :return:
        """
        k_val = random.choice([2, 3, 4])  # choose how many calls to change
        # Any more than 5 and my PC dies!
        calls = random.sample(range(1, self.calls + 1), k=k_val)
        arr_without_calls_to_change = [x for x in arr if x not in calls]

        for call in calls:
            insert_position = []
            car_sep = find_vehicle_positions(arr_without_calls_to_change)
            compatible_vehicles = self.get_compatible_vehicles(call)

            for vehicle in compatible_vehicles:
                # find valid places for compat vehicle. lower_bound index <==> upper_bound_index
                lower_bound_index, upper_bound_index = self.get_upper_lower_bound(car_sep, vehicle)
                route = arr_without_calls_to_change[lower_bound_index:upper_bound_index].copy()
                min_cost = float(-1)
                curr_best = (-1, -1)

                for i in range(len(route)):
                    first_insert_route = route.copy()
                    first_insert_route.insert(i, call)
                    for j in range(i, len(route)):
                        second_insert_route = first_insert_route.copy()
                        second_insert_route.insert(j, call)
                        if not self.feasible_vehicle(second_insert_route):
                            continue

                        cost = self.route_cost(second_insert_route)
                        if cost < min_cost:
                            min_cost = cost
                            curr_best = (i, j)

                if len(route) > 0 and curr_best == (-1, -1):
                    continue
                elif len(route) == 0:
                    curr_best = (0, 0)

                insert_position = (lower_bound_index + curr_best[0], lower_bound_index + curr_best[1])
                break

            if len(insert_position) > 0:
                arr_without_calls_to_change.insert(insert_position[0], call)
                arr_without_calls_to_change.insert(insert_position[1], call)
            else:
                arr_without_calls_to_change.insert(car_sep[-1], call)
                arr_without_calls_to_change.insert(car_sep[-1], call)

        return arr_without_calls_to_change

    def get_compatible_vehicles(self, call, shuffle=True):
        compatible_vehicles = [i for i in range(self.vehicle) if self.vessel_cargo[i, call - 1]]
        if shuffle:
            random.shuffle(compatible_vehicles)
            return compatible_vehicles
        else:
            return compatible_vehicles

    def get_upper_lower_bound(self, car_sep, vehicle):
        lower_bound_index, upper_bound_index = 0 if vehicle == 0 else car_sep[vehicle - 1] + 1, car_sep[vehicle]
        return lower_bound_index, upper_bound_index

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
                cycle = cycle[0: 2]
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
        outsorcevehicle = [x for x in range(1, self.calls + 1)]
        vehicle_valid_calls.append(outsorcevehicle)
        return vehicle_valid_calls


"""path = '../utils_code/pdp_utils/data/pd_problem/'
file_list = natsorted(os.listdir(path), key=lambda y: y.lower())

prob = load_problem(path + file_list[2])
# a, b = feasibility_check(sol, prob)

op = Operators(prob)

output = op.k_insert(get_init(7, 35))

print(output)
"""
