from bs4 import *
import requests 
import os


url = "https://www.wikiart.org/en/zdzislaw-beksinski/all-works/text-list"
prefix = "https://www.wikiart.org"
r = requests.get(url).content
soup = BeautifulSoup(r, 'html.parser')
ul = soup.find("body").find(class_='painting-list-text')
links = ul.find_all('a',href=True)
total = len(links)
for i,link in enumerate(links):
  r = requests.get(prefix+link['href'])
  soup = BeautifulSoup(r.text,'html.parser')
  image = soup.find('img')
  imgLink = image['src']
  r = requests.get(imgLink).content
  with open("./images/{0}.jpg".format(i+1), 'wb+') as f:
    f.write(r)
  if i % 50 == 0:
    print("Downloaded {0}/{1} Pictures".format(i,total))
