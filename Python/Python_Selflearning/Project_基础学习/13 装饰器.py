# 装饰器就是一个特殊的闭包函数，这个函数不是给自己使用的，而是给其他函数添加功能的
'''
为什么要用装饰器？
软件一旦上线运行后对修改源代码是封闭的，对扩展功能是开放的，这就用到了装饰器
不修改被装饰对象的源代码和调用方式
'''

# 测试type和isinstance两个函数，哪个速度更加的快
# 思路：先分别定义type和isinstance两个函数运行相同的次数
# type 查看类型， isinstance 判断类型

from datetime import datetime

def run_time(func): # 时间装饰器，func就是要被装饰的函数
    def new_func():
        start_time = datetime.now()
        print('开始时间为%s' %start_time)
        func()
        end_time = datetime.now()
        print('结束时间为%s' %end_time)
        total_time = end_time - start_time
        print('总共花费的时间%s' %total_time)
    return new_func

@run_time # 等同于：
# my_type = new_func = run_time(my_type)
# my_type()
def my_type():
    print('---type---')
    for i in range(10000):
        type('hello')

@run_time
def my_instance():
    print('---instance---')
    for i in range(10000):
        isinstance('hello',str)

my_type()
my_instance()