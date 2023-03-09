# Ligma Stonks

# Data Deliverable Report - Enigma Equities

**Team members**: Alyssa Cong, Roy Kim, Sejin Park, Benjamin Shih, Alexander Zheng

## Step 1

1. **Your data collection description goes here (at most around 200 words)** 

   

   We acquired a csv containing the QQQ constituents and used pandas to read/extract the data from the CSV (stock symbol and company name). In order to scrape equity data, we used AlphaVantage API to request historical OHCLV data passing in the list of stock symbols we had previously acquired. 

   

   For Twitter, we used Twitter API to query and scrape tweets from a given time frame for a query in the form of company names/symbols. The tweet query, content, date, text, and sentiment are stored in a csv file.

   

   We used Guardian API to find article URLs within a timeframe about companies. Queries are used with company names when they are unambiguous (non-ex. Apple is ambiguous because it can refer to the fruit). Queries that are ambiguous are found using tags (ex. Apple is found with the tag “technology/Apple”). These article URLs are then webscraped for the body/content of the article. The article title, date, URL, and sentiment measurements are stored in a csv file. 

   

   To get NYT data, we use the NYT API to scrape snippets of information summarizing each article published in a given time frame. The data was collected by using the common name of the company and restricted to articles labeled as relevant to the financial sector to avoid unrelated articles.The tweets and news articles are run through sentiment packages for a sentiment measure.

2. **Your data commentary goes here (at most around 200 words)** 

   

   The QQQ constituent data is reputable as it is public information and is objectively accurate.

   

   The reputation of Twitter is generally a reliable place to collect public sentiment as one of the largest social networking sites. The data from Twitter is public but requires an application for granting access to the API. The data may be potentially skewed towards younger people who use Twitter more frequently.

   

   We used the Guardian data based on its reputability as a trustworthy news source that does not align severely on any side of the political spectrum. It has many readers, which means that it is more likely to have an impact if an impact is to be found in news articles/social media. The Guardian’s readership demographic tends to reach older people much more than younger people. (credit: https://www.statista.com/statistics/380687/the-guardian-the-observer-monthly-reach-by-demographic-uk/)

   

   We used New York Times data based on its reputation as a news source which helps to reduce the influence that highly emotional pieces of objective articles would have on the aggregate data. In the data collection process, the querying was limited to financially associated articles, so the data may be skewed towards a more professional audience. 

## Step 2

**Your data schema description goes here (at most around 300 words)**

## Step 3

1. Your observation of the data prior to cleaning the data goes here (limit: 250 words)
2. Your data cleaning process and outcome goes here (at most around 250 words)

## Step 4

Your train-test data descriptions goes here (at most around 250 words)

## Step 5

Your socio-historical context & impact report goes here

## Step 6

5. **What are the major technical challenges that you anticipate with your project?** 

   

   In regards to sentiment analysis, we expect there to be difficulty in aggregating the data (i.e. Tweets) since different entities and individuals refer to companies in different ways (e.g. by ticker symbol, full name, etc.). We are also concerned about the number of independetnt variables and speicifically which to integrate in our analysis. Additionally, we anticipate a challenge in our strategy of analysis; for example, which models we will employ, since we have a wide variety of potential choices ranging from k-Nearest-Neighbors to Long-Short-Term-Memory models for time series.

6. **How have you been distributing the work between group members? What are your plans for work distribution in the future?** 

   

   We have been coming together as a group regularly in order to discuss current topics of action and decide how to proceed with our project. Currently we've spent time discussing project ideas and specific data sources as well as strategies we will employ for analysis. Our plan for future work distributiion is to come away from our regular meetings with courese of action and to distribute those among our members. For example, our current plan is to give each member a data source to gather data from so we can gather together and work on the complete data together.

