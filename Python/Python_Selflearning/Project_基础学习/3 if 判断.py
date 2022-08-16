# if的语法 1
tag = False  # 或者是tag = 1==1
if tag:
    print('条件满足')

# if的语法 2
if tag:
    print('条件满足')

else:
    print('条件不满足')

# if判断加上逻辑运算符
cls = 'human'
sex = 'female'
age = 18

if cls == 'human' and sex == 'female' and 16 < age < 22:
    print('开始表白。。。以下省略一万字')
else:
    print('阿姨好')

# 三目运算 满足条件的结果 if 条件 else 不满足条件的结果
# 只能对if 。。。 else
a = 6
print('满足条件') if a > 5 else print('不满足条件的结果')

# if的语法 3 多分枝 elif
score = input('>>')
score = int(score) # input进来是字符串，但是if只能比较相同类型，所以需要将score变量转换成数字类型
if score >= 90:
    print('优秀')
elif score >= 80:
    print('良好')
elif score >= 70:
    print('普通')
else:
    print('极差')

'''
if并列 的情况，与elif对比，逻辑有所不同
if并列是每个if都是独立的，一定会执行，而elif是在上一个if或者elif不满足的情况下才会执行
if score >= 90:
    print('优秀')
if 90 > score >= 80:
    print('良好')
if 80 > score >= 70:
    print('普通')
else:
    print('极差')
'''

# if嵌套
if cls == 'human' and sex == 'female' and 16 < age < 22:
    print('开始表白。。。以下省略一万字')
    is_success = input('女孩输入我愿意')
    if is_success == '我愿意':
        print('在一起')
    else:
        print('我逗你玩呢...')
else:
    print('阿姨好')
