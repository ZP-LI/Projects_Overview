#'''
def func():

    l_in = input().strip()

    left = 0
    right = 0
    l_mit = []
    max_len = 0

    for i in l_in:
        if i not in l_mit:
            l_mit.append(i)
            right += 1
        else:
            while i in l_mit:
                l_mit.pop(0)
                left += 1
            l_mit.append(i)
            right += 1

        max_len = max(max_len, len(l_mit))

    print(max_len)

if __name__ == "__main__":
    func()
'''

#'''