#'''
def func(filenames):
    try:
        filename, row = input().strip().split()
    except:
        flag = False
        return flag, filenames

    filename = filename.split('\\')[-1]
    row = int(row)

    flag_same = False

    for i in range(len(filenames)):
        if filenames[i][0] == filename and filenames[i][1] == row:
            filenames[i][2] += 1

            flag_same = True
            break

    if not flag_same:
        filenames.append([filename, row, 1])

    flag = True
    return flag, filenames


if __name__ == "__main__":
    filenames = [[[]] * 3]
    flag = True
    while flag:
        flag, filenames = func(filenames)

    filenames.remove([[], [], []])
    for i in range(len(filenames) - 1):
        for j in range(len(filenames) - i - 1):
            if filenames[j][2] < filenames[j + 1][2]:
                filenames[j], filenames[j + 1] = filenames[j + 1], filenames[j]

    count = 0
    for i in filenames:
        count += 1
        if count > 8:
            break
        if len(i[0]) > 16:
            i[0] = i[0][-16:]
        print(i[0], i[1], i[2])
'''
x = input().split()
for i in range(len(x) - 1):
    for j in range(len(x) - i - 1):
        if int(x[j]) < int(x[j + 1]):
            x[j], x[j + 1] = x[j + 1], x[j]
print(x)
'''