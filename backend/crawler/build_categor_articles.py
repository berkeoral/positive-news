import sys

import newspaper
import csv

from tqdm import tqdm

from backend.utils.txtops import TextOps
from newspaper import Article

source_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/uci-news-aggregator.csv"
write_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/crawled_category_news.txt"

txt_ops = TextOps()


def read_csv(path):
    data = []
    keep = [0, 1, 2, 4]
    with open(path, "r") as file:
        reader = csv.reader(file)
        data = [[row[_i] for _i in keep] for row in reader]
        for row in data:
            row[2] = row[2].split("\\")[0]
    data.pop(0)
    return data


data = read_csv(source_path)


def crawl_articles(_data, start_offset=0):
    for i in tqdm(range(start_offset, len(_data)), file=sys.stdout):
        article = Article(_data[i][2])
        try:
            article.download()
            article.parse()
            _data[i].append(article.text)
            if data[i][-1] != "":
                txt_ops.append_records_v2(write_path, data[i])
        except newspaper.ArticleException:
            tqdm.write("Article exception at id:%s" % i)


# crawl_articles(data, 13400)

test = txt_ops.records_as_list(write_path)

sss = []
t = []
b = []
e = []
m = []

for article in test:
    if article[3] == "t":
        t.append(article)
    elif article[3] == "b":
        b.append(article)
    elif article[3] == "e":
        e.append(article)
    elif article[3] == "m":
        m.append(article)


print("debug")
