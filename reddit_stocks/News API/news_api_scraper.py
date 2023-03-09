# run pip install --user -U nltk
# then in python run 
# >>> import nltk
# >>> nltk.downloader.download('vader_lexicon')
# if that isn't working try second answer from https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
from nltk.sentiment import SentimentIntensityAnalyzer
import requests

API_KEY = "660b3a6611e649b8885d901f453b9457"

def get_company_news(company: str) -> list:
    """Takes in company name or ticker, returns JSON with news related to the company"""
    query_params = {
            "apiKey": API_KEY,
            "qInTitle": company,
            "from": "2021-06-07"
            }
    base_url = "https://newsapi.org/v2/everything"
    r = requests.get(base_url, params=query_params)
    data = r.json()
    if (r.status_code != requests.codes.ok):
        #Throw Exception
        pass
    #Have to analyze the title or short description since every url will be different structure - impossible to scrape
    articles = data["articles"]
    res = []
    for i in articles:
        res.append(i["title"] + " " + i["description"])
    #print(res)
    return res

def sentiment_scorer(articles: list) -> list:
    """Takes in list of strings, returns list of number representing the sentiment score of the respective string"""
    sia = SentimentIntensityAnalyzer()
    res = []
    for string in articles: 
        res.append(sia.polarity_scores(string)["compound"])
    # print(res)
    return res

if __name__ == "__main__":
    """Retrieves news for all companies of interest"""
    company_list = ["Apple"]
    articles = []
    for company in company_list:
        articles = get_company_news(company)
    senti_list = sentiment_scorer(articles)