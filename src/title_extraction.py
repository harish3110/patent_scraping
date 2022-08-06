import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

base_url = 'https://patents.google.com/patent/US'

parser = argparse.ArgumentParser()
parser.add_argument('start', help='Starting index of patent scraping', type=int)
parser.add_argument('end', help='Starting index of patent scraping', type=int)
args = parser.parse_args()

patents = []
titles = []
dates = []

for i in tqdm(range(args.start, args.end)):
    try:
        url = f'{base_url}{i}'
        # get url instance
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        text = soup.find('title').get_text().split(' - Google Patents')[0]
        patent = text.split(' - ')[0]
        title = text.split(' - ')[1]
        date = soup.time.attrs['datetime']
        patents.append(patent)
        titles.append(title)
        dates.append(date)
    except:
        continue

path = '../data/'
df = pd.DataFrame(list(zip(patents, dates, titles)), columns =['patent_number', 'date', 'title'])
df['date'] =  pd.to_datetime(df['date'])
df.to_csv(f'{path}{args.start//1000}k-{args.end//1000}k.csv', index=False)


# logistic regression log loss: minimize summation of (log(1 + exp^-(yi* Wtxi))) + wt*w