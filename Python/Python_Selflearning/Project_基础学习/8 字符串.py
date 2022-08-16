res = 'dahai' # 单引号双引号三引号都可以

# 优先掌握的操作:
# 1.按索引取值（正向取值/反向取值）：从0开始
mag = 'hello world'
print(mag[0])
print(mag[-1])
print(mag)

# 2.切片（顾头不顾尾，步长）查找字符串当中的一段值[起始值：终止值：步长] -> 0是第一个字符
print(mag[0:5])
print(mag[0:5:2])
# 负数的步长
print(mag[10:5:-1])

# 3.长度len方法 可以计算长度
print(len(mag))

# 4.成员运算in和not in：判断一个子字符串是否存在于一个大的字符串中
# 返回布尔类型 True/False
print('dahai' in 'dahai is dashuaibi')
print('xialuo' in 'dahai is dashuaibi')

# 增 字符串拼接
print('dahai' + 'dsb')

# %s 占位符 可以接受所有的数据类型 %d 只能接受数字 有局限性
# 多个值的话直接放到 %后面 要有括号
name = input('请输入名字')
print('my name is %s' %name)

# 字符串的删除
name1 = 'dahai'
del name1
# print(name1)

# 字符串的修改
# 字符串字母变大写和变小写 lower，upper
mag2 = 'abc'
mag3 = mag2.upper()
print(mag3)
print(mag2)
# 把第一个字母转换为大写 capitalize
letter = 'abcd'
print(letter.capitalize())
# 把字符串切分成列表 split 默认空格字符切分
magg = 'hello world python'
magg1 = magg.split()
print(magg1)
# 可以切分你想要的字符 比如*
maggg = 'hello*world*python'
maggg1 = maggg.split('*')
print(maggg1)
# 切分split的作用：针对按照某种分隔符组织的字符串，可以用split将其切分成列表，进而进行取值
magww = 'root:123456'
print(magww[5:11])
res3 = magww.split(':')
print(res3)
print(res3[1])
# 去掉字符串左右两边的字符strip 不写默认是空格字符，不管中间的其它字符
user1 = '      dahai          '
userl = user1.strip()
print(userl)

name = input('输入名字').strip()
print(name)

# 查找
# 1. find，rfind，index，rindex 查找子字符串在大字符串的那个位置（起始索引）
magee = 'hello dahai is dsb'
print(magee.find('dahai'))
# 找到字符串的起始索引，默认约束的范围是所有字符
print(magee.find('xialuo'))
# 约束范围是0开始到3结束，所以找不到，但是找不到会返回-1，不会报错
# 与find不同 print(magee.index('xialuo')) 找不到会报错
# 2. count 统计一个字符串在大写字符串中出现的次数
magrr = 'dahai dierge dahai'
print(magrr.count('dahai'))
# 3.isdigit 判断一个字符串里的数据是不是都是数字
mun = '14'
print(mun.isdigit())
# 4.isalpha 判断每个数据是不是字母
letter1 = 'abc'
print(letter1.isalpha())

# 比较开头的元素是否相同 startswith
# 比较结尾的元素是否相同 endswith
msg = 'dahai is dsb'
print(msg.startswith('dahai'))
print(msg.endswith('sb'))
print(msg.endswith('b'))

# 判断字符串中的值是否全是小写的 islower
# 判断字符串中的值是否全是大写的 isupper
letter1 = 'Abc'
print(letter1.islower())
letter2 = 'abc'
print(letter2.islower())
letter3 = 'ABC'
print(letter3.isupper())