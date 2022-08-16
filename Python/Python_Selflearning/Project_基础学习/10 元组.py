# 一.基本使用：tuple
# 1 用途：记录多个值，当多个值没有改的需求，此时用元组更合适
# 2 定义方式：在（）内用括号分割开多个任意类型的值

t = (1,2,'dahai',(2,2,),[1,2,3])
print(t)
print(type(t))
print(t[0])

# t[0] = 8 会报错，元组是不能改的
# 但是元组里面的列表是可以改的
t[4][0] = 8
print(t)

# 如果想修改，可以转换成列表形式
t = list(t)
t[0] = 8
print(tuple(t))

print(type(t))
print(type(tuple(t)))