#'''
def func():

    list = input().strip()
    # print(list)

    nums = [0] * (len(list) - 1)
    # print(nums)

    for i in list:
        if nums[int(i) - 1] == 1:
            print(i)
            break
        else:
            nums[int(i) - 1] = 1

if __name__ == "__main__":
    func()
'''

'''