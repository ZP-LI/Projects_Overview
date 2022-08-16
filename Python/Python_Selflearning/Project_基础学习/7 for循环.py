# while 遍历列表
# names = ['dahai', 'xialuo', 'guan', 'xishi']
# i = 0;
# while i < len(names):
#     print(names[i])
#     i += 1
#
# # for 遍历列表
# for n in names:
#     print(n)

# for 遍历字典

# names= {'name1':'dahai', 'name2':'xialuo', 'name3':'xishi'}
# # 默认遍历key值的两种方法
# for i in names:
#     print(i)
#
# for i in names.keys():
#     print(i)
# # 遍历value值
# for i in names:
#     print(names[i])
#
# for i in names.values():
#     print(i)
# # 遍历键值对
# for i in names.items():
#     print(i)

# range
# a = range(0,5) 起始索引，结束索引，可能还有步长 range(0,5,2)
# # 它是一个迭代器
# print(type(a))
# print(list(a))
# 为什么不直接变成列表，因为会浪费内存
# 一般和for循环连用，循环一次取一次
# 相当于母鸡下蛋，一次下一个，下了0 1 2 3 4这5个鸡蛋
# for i in range(0,5):
#     print(i)
#
# for i in [0,1,2,3,4]:
#     print(i)

# names = ['dahai', 'xialuo', 'guan', 'xishi']
# for i in range(0,len(names)):
#     print(names[i])

# for + break/continue
# names = ['dahai', 'xialuo', 'guan', 'xishi']
# for n in names:
#     if n == 'guan':
#         continue
#         # break
#     print(n)

# for + else
# for i in range(10):
#     if i == 4:
#         #break
#         continue
#     print(i)
# else:
#     print('======')

# for循环嵌套
for i in range(1,10): #控制9行
    for j in range(1,i+1):
        print('%s*%s=%s' %(i,j,i*j),end=' ')
    print()
# print还有一个参数end，默认是\n
# print(1,end=' ')
