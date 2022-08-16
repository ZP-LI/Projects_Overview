# 一:基本使用
# 1 用途: 存放多个值，可以根据索引存取值
# 2 定义方式: 在[]内用都好分隔开多个任意类型的值

L = ['dahai',1,1,2,[1,'xiaohai']]
print(L)

# 1.按索引取值（正向/反向）：即可存也可取
# 索引是从0开始 相当于书本的页码
print(L[0])
print(L[-1])
print(L[3][1])
L[0] = 'honghai'

# 2.切片（顾头不顾尾，步长）
# 查找列表当中的一段值 [起始值：终止值：步长]
print(L[0:3:1])

# 3.len长度 列表元素的多少个
print(len(L))

# 4.成员运算in和not in
print('honghai' in L)
print('honghai' not in L)
# 查看列表某个元素的个数 count/index
print(L.count('honghai'))
print(L.index('honghai',2,3))
# 没有找到会报错

# 增
# append() 往列表的末尾追加一个元素
L.append('lanhai')
# insert(index,element) 往指定索引位置前插入一个元素
L.insert(0,'huanghai')
# extend() 往列表当中添加多个元素 括号里面放列表
L.extend(['lvhai','zihai'])

# 删
# del L[0]
# L.remove('lvhai') 指定删除
# pop #从列表里面拿走一个值，按照索引删除值，默认是删除最后一个
L.pop() # 相当于L.pop(-1)
L.pop(1)
# L.clear()

# 改
L[0] = 'baihai'

# 反序
L.reverse()

# 排序
list_num = [1,3,2,5]
list_num.sort(reverse=True) # True代表倒序/False为正序