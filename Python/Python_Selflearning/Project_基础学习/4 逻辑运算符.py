# and
name = 'dahai'
num = 20

print(num > 18 and num < 26 and name == 'dahai' and 1 > 3)

# or
print(1 > 3 or 2 > 4 or 'x' == 'y' or 1 == 1)

# not
print(not 1 > 3)

'''
not的优先级最高
如果都是一种，那从左到右计算
如果and和or混用，先用括号把and两边两个条件括起来，然后进行运算
'''

print(not 3 > 1 or 3 > 1)  # True

res = not False and True or False or False or True  # True 最好用括号把and两端括起来，便于阅读
print(res)

print((3 > 4 and 4 > 3) or (1 == 3 and ('x' == 'x' or 3 > 3)))
