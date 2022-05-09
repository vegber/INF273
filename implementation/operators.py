import random

from implementation.operator_helper_class import *
from utils_code.pdp_utils import *


class Operators:
    def __init__(self, loaded_problem, global_probability=1):
        self.problem = loaded_problem
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
        self.glob_ind = 1 - global_probability  # - (0.1 * global_probability)

    def one_insert(self, arr):
        """
        One insert operator:
        inputs a one dim array, outputs a one dim arr.
        Simply takes random vehicles and puts them in another semi - random vehicle (valid cars)
        :param arr:
        :return:
        """
        while True:
            sol_vehicle_arr = self.to_list_v2(arr)
            random_vehicle = random.randint(0, self.vehicle)  # zero indexed
            valid_placements = self.get_valid_calls(random_vehicle)
            random_call = random.choice(valid_placements)
            # check if random vehicle is "itself"
            origin_vehicle, origin_pickup, origin_delivery = find_indexes(sol_vehicle_arr, random_call)
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

            feas, _ = feasibility_check(arr, self.problem)
            if feas or self.glob_ind > random.random():
                return arr

    def max_cost_swap(self, arr):
        """
        Naive operator. Uses a custom cost function to find the vehicle with most
        costly calls. Idea is to spread cost evenly.
        :param arr:
        :return:
        """
        while True:
            sol_vehicle_arr = self.to_list_v2(arr)
            car_cost = self.car_cost(sol_vehicle_arr)
            car_to_change = car_cost.index(max(car_cost))  # select vehicle with the highest cost not zero indexed.
            out = self.change_car_insert(arr, car_to_change)
            passed, _ = feasibility_check(out, self.problem)
            if passed or random.random() > self.glob_ind:
                return out

    def k_insert(self, arr, K):
        """
        Two insert method that removes two random delivery calls.
        Insertion is helped by checking problem specific data at
        the removed calls. Efficient at pointing the search at
        the right "track"...
        :param K:
        :param arr:
        :return:
        """
        calls = random.sample(range(1, self.calls + 1), k=K)
        org_arr_without_calls_to_move = get_sol_without_calls_to_move(calls, arr)

        for call in calls:
            car_index = get_vehicle_indexes(org_arr_without_calls_to_move)
            compatible_vehicles = self.get_compatible_vehicle(call)
            insert_positions = self.find_best_pos(org_arr_without_calls_to_move, call, car_index, compatible_vehicles)
            place_at_insert_positions(call, car_index, insert_positions, org_arr_without_calls_to_move)
        return org_arr_without_calls_to_move.copy()

    def two_inserter(self, arr):
        return self.k_insert(arr, 2)

    def more_insert(self, arr):
        return self.k_insert(arr, random.choice([2, 3, 4]))

    def smarter_one_insert(self, arr):
        return self.k_insert(arr, 1)

    def car_cost(self, arr):
        """
        Input two dim per car
        :param arr:
        :return:
        """
        return [self.call_cost(i, x) for i, x in enumerate(arr)]

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

    def valid_insertion(self, car, calls_to_place):
        """
        Problem specific validity checker. Boolean "check if" calls can be places at vehicle
        # Todo room for more checks!
        :param car:
        :param calls_to_place:
        :return:
        """
        calls_length = len(calls_to_place)
        if not calls_length: return True  # can apply if not available car
        call_size, current_time, total = 0, 0, 0
        calls = np.sort(calls_to_place.copy())
        calls = [x - 1 for x in calls]  # zero index calls
        i = [i for i, e in enumerate(calls)]
        call_size = self.get_call_size(calls, call_size)
        # check vessel capacity against call size
        if not np.any(self.vessel_capacity[car] - [total := total + x for x in call_size[i]] < 0):
            cargo_t_windows = self.get_call_upper_lower_t_window(calls, calls_length)
            origin_nodes = self.get_start_loc(calls, i)
            total_travel_time = self.get_travel_time(calls, car, origin_nodes)
            return is_within_arrival_time(calls_length, current_time, total_travel_time, cargo_t_windows)
        return False

    def get_call_size(self, calls, call_size):
        """
        Returns call size from cargo list
        :param calls:
        :param call_size:
        :return:
        """
        call_size -= self.cargo[calls, 2]  # Cargo size for all calls
        call_size[::2] = self.cargo[calls[::2], 2]  # eval calls
        return call_size

    def get_start_loc(self, calls, index):
        """
        Returns the origin node for calls
        NB - must be zero indexed
        :param calls:
        :param index:
        :return:
        """
        origin_nodes = self.cargo[calls, 1].astype(int)
        origin_nodes[::2] = self.cargo[calls[::2], 0]
        return origin_nodes[index] - 1

    def get_travel_time(self, calls, car, origin_nodes):
        """
        Find the total travel time for calls.
        Included first time visit and travel time from origin node
        to origin node.
        :param calls:
        :param car:
        :param origin_nodes:
        :return:
        """
        first_visit_time = self.first_travel_time[car, int(self.cargo[calls[0], 0] - 1)]  # convert float to
        travel_time_visited = self.travel_time[car, origin_nodes[:-1], origin_nodes[1:]]
        return [x for x in first_visit_time.flatten()] + [y for y in travel_time_visited.flatten()]

    def get_call_upper_lower_t_window(self, calls, calls_length):
        """
        Finds, and returns the upper/lower time window for nodes.
        :returns at format
        [
            [A_lower_pickup, A_lower_delivery, B_lower_pickup, B_lower_delivery]
            [A_upper_pickup, A_upper_pickup, B_upper_pickup, B_upper_delivery]
        ]
        :param calls:
        :param calls_length:
        :return:
        """
        cargo_t_windows = np.zeros((2, calls_length))
        # (0) origin node,
        # (1) destination node,
        # (2) size,
        # (3) cost of not transporting,
        # (4) lower-bound time window for pickup,
        # (5) upper_time window for pickup,
        # (6) lower-bound time window for delivery,
        # (7) upper_time window for delivery
        cargo_t_windows[0] = self.cargo[calls, 6]  # upper_time window for pickup for first call
        cargo_t_windows[0, ::2] = self.cargo[calls[::2], 4]  # lower-bound time window for pickup -- A & B node
        cargo_t_windows[1] = self.cargo[calls, 7]  # fill cargo delivery schedule with upper_time window for delivery
        cargo_t_windows[1, ::2] = self.cargo[calls[::2], 5]  # upper_time window for pickup
        return cargo_t_windows

    def call_cost(self, car, calls):
        """
        Modifiable cost function. Returns the cost current calls.
        Uses the cargo, travel cost, FirstTravelCost, PortCost from the problem.
        Currently, returns the sum of the modules implemented. Can be changed at a later stage
        :param car:
        :param calls:
        :return:
        """
        car = car - 1

        if len(calls) == 0:
            return 0
        calls_to_change = np.sort([x - 1 for x in calls.copy()])
        calls_to_change_i = get_zeroed_indexes(calls_to_change)
        origin_dest_node = self.get_start_loc(calls_to_change, calls_to_change_i)
        travel_cost_at_curr_car = self.TravelCost[car, origin_dest_node[:-1], origin_dest_node[1:]]
        FirstVisitCost = self.FirstTravelCost[car, (self.cargo[calls_to_change[0], 0] - 1).astype(int)]
        return np.sum([x for x in FirstVisitCost.flatten()] + [y for y in travel_cost_at_curr_car.flatten()]) \
               + np.sum(self.PortCost[car, calls_to_change]) / 2

    def find_best_pos(self, arr, call, car_index, comp_vehicles):
        """
        Goes through compatible vehicles and searches for the best
        insertion points. Uses problem specific information to
        narrow search. Best ranked position is returned

        :param arr: solution without calls to change:
        :param call: current call to change :
        :param car_index: start index of cars :
        :param comp_vehicles: valid cars to place call :
        :return:
        """

        insert_positions = []
        for v in comp_vehicles:
            lower_bound, upper_bound = get_upper_lower_bound(car_index, v)
            valid_car_indexes = arr[lower_bound:upper_bound].copy()
            current_min_position, curr_min = (-1, -1), -1
            for pickup in range(len(valid_car_indexes)):
                car_indexes = valid_car_indexes.copy()
                car_indexes.insert(pickup, call)
                for deliver in range(pickup, len(valid_car_indexes)):
                    potential_call_index = car_indexes.copy()
                    potential_call_index.insert(deliver, call)
                    if self.valid_insertion(v, potential_call_index):
                        cost = self.call_cost(v, potential_call_index)
                        if cost < curr_min: current_min_position, curr_min = (pickup, deliver), cost
                    else:
                        continue
            if len(valid_car_indexes) > 0 and current_min_position == (-1, -1):
                continue
            elif len(valid_car_indexes) == 0:
                current_min_position = (0, 0)

            insert_positions = (
                lower_bound +
                current_min_position[0],
                lower_bound +
                current_min_position[1])  # shift to insert: pickup, delivery
            break
        return insert_positions

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
