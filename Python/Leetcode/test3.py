#'''
def func():
    tree_full = input().strip(' []').split(',')

    tree_all = []

    # record all 2 level tree
    for i in range(7):
        tree_single = []
        tree_single.append(tree_full[i])
        tree_single.append(tree_full[(i + 1) * 2 - 1])
        tree_single.append(tree_full[(i + 1) * 2])
        tree_all.append(tree_single)

    # record all 3 level tree
    for i in range(3):
        tree_single = []
        tree_single.append(tree_full[i])
        tree_single.append(tree_full[(i + 1) * 2 - 1])
        tree_single.append(tree_full[(i + 1) * 2])
        tree_single.append(tree_full[i * 4 + 3])
        tree_single.append(tree_full[i * 4 + 4])
        tree_single.append(tree_full[i * 4 + 5])
        tree_single.append(tree_full[i * 4 + 6])
        tree_all.append(tree_single)

    # record all same tree
    tree_all.sort()
    tree_same = []
    for i in range(len(tree_all) - 1):
        if (tree_all[i] == tree_all[i + 1]) and (tree_all not in tree_same):
            tree_same.append(tree_all[i])

    for i in range(len(tree_same)):
        if tree_same[i][1] == 'null' and tree_same[i][2] == 'null':
            del tree_same[i]

    # print
    if tree_same == []:
        print('-1')
        return

    for i in tree_same:
        if len(i) == 7:
            print('[', end='')
            print(','.join(i), end='')
            print(']')
            return

    for i in tree_same:
        print('[', end='')
        print(','.join(i), end='')
        print(']', end=',')

    return

if __name__ == "__main__":
    func()
'''
arr = ['null', 'null', 'null']
arr.pop(['null', 'null', 'null'])
print(arr)
#'''