arr = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]

var = [1, 2, 1, 2, 3, 3]


def zeros(arr):
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
    # add the last position of list

    return valid_pos


valid = zeros(arr)
print(valid)
print(arr)
arr.insert(valid[0], 9)
print(arr)
