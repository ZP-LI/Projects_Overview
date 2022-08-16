# def fun():
#     print('run')

# 无代码的函数，用来测试，先做好项目的需求
# def fun():
#     pass

# def gongchang(a,b):
#     c = a + b
#     print(c)

# 不定长的参数
def gongcang(*args,**kwargs):
    print(args)
    print(kwargs)
gongcang(*[1,2,3],**{'x':3,'y':4})

# 返回值 return
def gongchang():
    print('zhengzaizhizaoshouji')
    return None
# return是一个函数结束的标志，函数内可以有多个return，但只要执行一次，整个函数就会结束运行
# return的返回值无类型和个数限制，0个 返回None，1个 返回该值，多个 返回元组类型
