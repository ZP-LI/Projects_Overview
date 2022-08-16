'''
def func():
    try:
        n, m = map(int, input().strip().split())
    except:
        flag = False
        return flag

if __name__ == "__main__":
    flag = True
    while flag == True:
        flag = func()
'''

#'''
row = 10
col = 10
test_list = [1000] * row
for i in range(row):
    test_list[i] = [1000] * col
test_list[6][3] = 0
test_list[5][2] = -1
test_list[6][4] = -1
test_list[3][4] = -1
test_list[4][5] = -1
test_list[5][6] = -1
test_list[6][7] = -1

for loop in range(row + col):
    for r in range(row):
        for c in range(col):
            if test_list[r][c] == 0 or test_list[r][c] == -1:
                continue

            sur = []
            if r - 1 >= 0:
                if test_list[r - 1][c] != -1:
                    sur.append(test_list[r - 1][c])
            if r + 1 < row:
                if test_list[r + 1][c] != -1:
                    sur.append(test_list[r + 1][c])
            if c - 1 >= 0:
                if test_list[r][c - 1] != -1:
                    sur.append(test_list[r][c - 1])
            if c + 1 < col:
                if test_list[r][c + 1] != -1:
                    sur.append(test_list[r][c + 1])
            test_list[r][c] = min(sur) + 1

#print(test_list)
inv_buffer = []
inv_buffer.append([2, 7])
count = [0] * (test_list[2][7] + 1)
count[-1] = 1
while inv_buffer:
    cur_point = inv_buffer.pop(0)
    if test_list[cur_point[0]][cur_point[1]] == 0:
        break
    if cur_point[0] - 1 >= 0:
        if test_list[cur_point[0] - 1][cur_point[1]] == test_list[cur_point[0]][cur_point[1]] - 1:
            inv_buffer.append([cur_point[0] - 1, cur_point[1]])
            count[test_list[cur_point[0] - 1][cur_point[1]]] += 1
    if cur_point[0] + 1 < row:
        if test_list[cur_point[0] + 1][cur_point[1]] == test_list[cur_point[0]][cur_point[1]] - 1:
            inv_buffer.append([cur_point[0] + 1, cur_point[1]])
            count[test_list[cur_point[0] + 1][cur_point[1]]] += 1
    if cur_point[1] - 1 >= 0:
        if test_list[cur_point[0]][cur_point[1] - 1] == test_list[cur_point[0]][cur_point[1]] - 1:
            inv_buffer.append([cur_point[0], cur_point[1] - 1])
            count[test_list[cur_point[0]][cur_point[1]] - 1] += 1
    if cur_point[1] + 1 < col:
        if test_list[cur_point[0]][cur_point[1] + 1] == test_list[cur_point[0]][cur_point[1]] - 1:
            inv_buffer.append([cur_point[0], cur_point[1] + 1])
            count[test_list[cur_point[0]][cur_point[1] + 1]] += 1
print(max(count))