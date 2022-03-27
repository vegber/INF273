import time

from random_solution_driver import get_init


def to_list_v2(arr):
    """
    Converts a one dimensional array to a
    two dimensional list

    problem specific method.
    Todo runtime
    :param arr:
    :return:
    """

    out = [[]] * (3 + 1)  # two dim n vehicle list

    counter = 0

    print(arr)
    L = list(map(str, arr))
    print(L)
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


sol = get_init(3, 7)

st = time.time()
for x in range(100000):
    out = [[]] * 3
    # out = ([() * x for x in range(3 + 1)])  # two dim n vehicle list
print(f"time:  {time.time() - st}")

for x in range(1):
    to_list_v2(sol)
