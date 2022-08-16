'''
def func():

    # please define the python3 input here. For example: a,b = map(int, input().strip().split())
    try:
        n, m = map(int, input().strip().split())
    except:
        flag = False
        return flag
    grade = map(int, input().strip().split())
    grade = list(grade)
    task = [[] * 3] * m
    # n, m -> int, id, task -> str

    global ret
    for i in range(m):
        task[i] = input().strip().split()
        if task[i][0] == 'Q':
            max_grade = 0
            if int(task[i][1]) > int(task[i][2]):
                task[i][1], task[i][2] = task[i][2], task[i][1]
            if int(task[i][1]) > n:
                break
            elif int(task[i][2]) > n:
                task[i][2] = str(n)
            for i in range(int(task[i][1]) - 1, int(task[i][2])):
                max_grade = max(max_grade, grade[i])
            ret.append(max_grade)
        elif task[i][0] == 'U':
            grade[int(task[i][1]) - 1] = int(task[i][2])
        else:
            break

    #print(grade, ret)
    flag = True
    return flag

if __name__ == "__main__":
    global ret
    ret = []
    flag = True
    while flag == True:
        flag = func()

    for i in ret:
        print(i)
'''
x = 'fawegewrhgerhegwgfwezteheqg'
print(x[-16:])
#'''