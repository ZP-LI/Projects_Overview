db_user = 'dahai'
db_password = '123'

# while + break: break代表结束本层循环

'''
while True:
    input_user = input('请输入用户名:')
    input_password = input('请输入密码:')
    if input_user != db_user or input_password != db_password:
        print('登陆失败')
    else:
        print('登陆成功')
        break
'''

# while + 一个条件范围 不满足范围的条件结束循环
# while + continue：continue代表结束本次循环（本次循环continue之后的代码不在运行），直接进入下一次循环
'''
start = 0;
while start < 8:
    start += 1
    if start == 4:
        continue
    print(start)
'''

# 断点
# F9 绿色的三角形是跳到下一个断点
# F8 蓝色朝下的箭头是单步走
# Alt + F9 移动到光标处

# while + else
n = 0
while n < 6:
    n += 1
    if n == 6:
        break
    print(n)
else:
    # else的代码会在while循环没有break打断的情况下最后运行
    print('==============')
print('------------')