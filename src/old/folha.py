import sys
import requests
import textwrap
from bs4 import BeautifulSoup as bs

# From file
#filename = sys.argv[1]
#with open(filename, 'r') as fd:
#    text = fd.readlines()
#soup = bs(text, 'html.parser')

# From web
url = sys.argv[1]
page = requests.get(url)
page.encoding = 'utf-8'
soup = bs(''.join(page.text), 'html.parser')


materia = soup.find('div', class_='c-news__body')
materia.find('div', class_="c-advertising__banner-area").decompose()
materia.find('div', class_="js-gallery-widget").decompose()

for a in materia.findAll('a'):
    a.replaceWithChildren()

for d in materia.findAll('div'):
    d.replaceWithChildren()

print(soup.title.text)
print(len(soup.title.text) * '=')
print()
for p in materia.findAll('p'):
    for l in textwrap.wrap(p.text, width=79):
        print(l)
    print()

# lines = textwrap.wrap(materia.get_text(), width=79)
# 
# for l in lines:
#     print(l)
# 
