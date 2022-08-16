user = 'dahai'
pwd = 123
balance = 5000
tag = True
while True:
    # while Tag 是登录程序
    while tag:
        user1 = input('输入用户名:')
        if user1 != user:
            print('你输入的用户名错误，请重新输入:')
            continue
        pwd1 =int(input('请输入密码:'))
        if pwd1 == pwd:
            print('登陆成功')
            tag = False
            break
        else:
            print('输入错误')
    money = int(input('输入你的取款金额:'))
    if balance > money:
        balance = balance - money
        print('恭喜你取走了%s' %money)
        print('还剩%s' %balance)
        break
    else:
        print('余额不足')