import requests
from pyquery import PyQuery
url="https://pvp.qq.com/web201605/herolist.shtml"
html=requests.get(url).content
# print(html)
doc=PyQuery(html)
items=doc('.herolist > li').items()
# 循环遍历
for item in items:
    # 继续下找，查找img标签，attr属性功能，src
    url=item.find('img').attr('src')
    # print(url)
    urls="http:"+url
    name=item.find("a").text()
    # print(name)
    url_content=requests.get(urls).content
    with open('./picture/'+name+'.jpg','wb')as file:
        file.write(url_content)
        print("正在下载: %s.....%s" %(name,urls))
print("下载完毕")
