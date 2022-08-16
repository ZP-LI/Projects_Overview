'''
字典类型：dict
作用：记录多个key：value值，优势是每一个值value都有其对应关系/映射关系key，而key对value有描述性作用
定义：在（）内用括号分割开多个key：value元素，其中value可以是任意的数据类型，而key通常应该是字符串类型
'''

info = {'name':'dahai','age':18}
print(info)

print(info['name'])
print(info['age'])

# 列表和字典的区别：列表是依靠索引，字典是依靠键值对，key描述性的信息
# 注意字典的key必须是不可变类型，key无法提取
# 错误示范：(列表不能用作key，但是元组可以)
# info1 = {[1,2,3]:'heihai'}
info2 = {(1,2,3):'heihai'}
print(info2[(1,2,3)])

# 生成字典的方式 2
dic = dict(x=1,y=2,z=3)
print(dic)

# 字典的添加操作
print(info)
info['addr']='changsha'
print(info)

# 字典里的len可以查看键值对的个数
print(len(info))
#成员运算in和not in字典的成员运算判断的是key，返回值是布尔类型
print('name' in info)

# 删 info.clear() / del info['name'] / pop 实际上是拿走value值，有返回值
res = inof.pop('addr')
# 删除不存在的key则报错
# info.popitem() 最后一对键值对删除，字典无序，返回的是一个元组

# 改
info['name'] = 'honghai'
info.update({'name':'dahai'})
# setdefault 有则不动，无则添加/返回新值