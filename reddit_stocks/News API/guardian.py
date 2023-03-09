from requests.models import to_key_val_list
from bs4 import BeautifulSoup
import requests
import sqlite3

GUARDIAN_API_URL = "https://content.guardianapis.com/search?"
DATE = "2015-06-27"
CORPORATION_NAME = "Apple"
API_KEY = "0a873f97-b5d9-4e12-82cb-4005c0ea887c"
HELP_TAG = "&tag=technology/apple"


response = requests.get(f'{GUARDIAN_API_URL}{HELP_TAG}&q={CORPORATION_NAME}$show-fields=body&api-key={API_KEY}&from-date={DATE}')
if response.status_code == 404:
  print(f'404 error: {s}')
else:
  response.raise_for_status()

  data_list = []
  json_data = response.json()['response']

  pages = json_data['pages']

  # print(f'json_data: {json_data}')
  articles = json_data['results']
  for article in articles: 
    date = article['webPublicationDate'][:10]
    title = article['webTitle']
    url = article['webUrl']
    # body = article['body']
    data_list.append([date, title, url])



# for row in data_list: 
response = requests.get(data_list[0][2])
response.raise_for_status() # will throw an error if status is not OK
data = response.content

html_dump = BeautifulSoup(data, 'html.parser')

# TODO: Save data below.
paragraphs = html_dump.find_all('dcr-s23rjr')
print(len(paragraphs))
text = ""
for paragraph in paragraphs:
  paragraph = paragraph[1:]
  print("paragraph: " + paragraph)
  href = False
  while len(paragraph) != 0: 
    ind = paragraph.find('"')
    if href: 
      text_start = paragraph.find('"in body link">')
      text_end = paragraph.find('</a>')
      text += paragraph[text_start:text_end]
      href = False
    else: 
      text += paragraph[:ind]
      href = True
    paragraph = paragraph[ind + 1:]
    print("text: " + text)
data_list[0].append(text)
print(data_list[0])








# # Create connection to database
# conn = sqlite3.connect('data.db')
# c = conn.cursor()

# # Delete tables if they exist
# c.execute('DROP TABLE IF EXISTS "companies";')
# c.execute('DROP TABLE IF EXISTS "quotes";')

# #TODO: Create tables in the database and add data to it. REMEMBER TO COMMIT

# companes_creation = 'CREATE TABLE companies(symbol TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL, location TEXT NOT NULL);'

# quotes_creation = 'CREATE TABLE quotes (symbol TEXT PRIMARY KEY NOT NULL, price FLOAT NOT NULL, avg_price FLOAT, num_articles INT, volume INT NOT NULL, change_pct FLOAT NOT NULL);'

# c.execute(companes_creation)
# c.execute(quotes_creation)
# conn.commit()

# for row in all_data: 
#   comp_name = row[0]
#   sym = row[1]
#   price = row[2]
#   change_perc = row[3]
#   vol = row[4]
#   hq_state = row[5]
#   five_avg = row[6]
#   news = row[7]

#   if five_avg != None: 
#     c.execute('INSERT INTO companies VALUES (?, ?, ?)', (sym, comp_name, hq_state))
#     c.execute('INSERT INTO quotes VALUES (?, ?, ?, ?, ?, ?)', (sym, price, five_avg, news, vol, change_perc))
# conn.commit()
# c.close()
# conn.close()