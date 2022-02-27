import random


def one_insert(arr, vehicle, calls):
    to_liz = to_list(arr)
    print(to_liz)
    random_call = random.randint(0, vehicle+1)

    return arr


def two_exchange(arr, vehicle, calls):
    return arr


def three_exchange(arr, vehicle, calls):
    return arr


def to_list(arr):
    lst_string = "".join([str(x) for x in arr])
    lst2 = [e + '0' for e in lst_string.split('0') if e]  # lst_string.split('0')
    lst3 = [list(y) for y in lst2]
    lst4 = [list(map(str, z)) for z in lst3]
    lst4[-1].remove('0')
    va = list(map(int, ''.join([''.join(x) for x in lst4])))
    return va
