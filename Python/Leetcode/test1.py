#'''
def func():
    List_in = input().strip()

    List_hw = []
    for i in List_in:
        if i.isalpha() == True or i.isdigit() == True:
            List_hw.append(i)

    if len(List_hw) == 0:
        print('0,0')
        return

    for i in range(len(List_hw) // 2):
        if List_hw[i] != List_hw[-i-1]:
            print('0,0')
            return

    print('1,', end='')
    print(len(List_in))

if __name__ == "__main__":
    func()
'''
List_hw = 'fwgerhgeh'
print('1,', end='')
print(len(List_hw))
#'''