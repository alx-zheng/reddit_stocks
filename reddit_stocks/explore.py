import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import csv

subreddit = "AAPL"
df = pd.read_csv('explore/%s_joined_full.csv' % subreddit)

df["Date"] = df["Date"].astype(str)
df["datetime"] = pd.to_datetime(df['Date'])
df["Avg. Close"] = df["Avg. Close"].astype(float)
df["Avg Joint Pos Sentiment"] = ((df["Reddit Avg. Pos"] + df["Guardian Avg. Pos"] + df["NYT Avg. Pos"])/3).astype(float)
df["Avg Joint Neu Sentiment"] = ((df["Reddit Avg. Neu"] + df["Guardian Avg. Neu"] + df["NYT Avg. Neu"])/3).astype(float)
df["Avg Joint Neg Sentiment"] = ((df["Reddit Avg. Neg"] + df["Guardian Avg. Neg"] + df["NYT Avg. Neg"])/3).astype(float)


before = (df['datetime'] > dt.datetime(2010,1,15)) & (df['datetime'] <= dt.datetime(2017,1,1))
after = (df['datetime'] > dt.datetime(2017,1,1)) & (df['datetime'] <= dt.datetime(2021,6,30))


june26df = df.loc[before]
after2019df = df.loc[after]

june26df_cleaned = june26df[~june26df.isin([np.nan, np.inf, -np.inf]).any(1)]
after2019df = after2019df[~after2019df.isin([np.nan, np.inf, -np.inf]).any(1)]




def scatter(table, xaxis, yaxis, title, xlabel, ylabel):
    x = table[xaxis]
    y = table[yaxis]
    plot = plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #manipulate x-axis
    plt.xticks(np.arange(0, len(x), 18))
    plt.xticks(rotation = 90)

    plt.gcf().subplots_adjust(bottom=0.25)
    return plot


'''
CLOSING PRICES
'''

plot = scatter(df, 'Date', 'Avg. Close', 'Closing Prices', 'date', 'Closing Price')
plt.savefig('explore/graphs/closing_prices_time.png')
plt.show(block = True)


'''
INDIVIDUAL SENTIMENTS PER SOURCE
'''

'''
REDDIT
'''

##reddit pos
plot = scatter(june26df_cleaned, 'Date', 'Reddit Avg. Pos', 'Reddit_Pos', 'date', 'reddit_pos_sentiment')

# plt.show(block = True)

##reddit neu
plot = scatter(june26df_cleaned, 'Date', 'Reddit Avg. Neu', 'Reddit_Neu', 'date', 'reddit_neu_sentiment')
# plt.show(block = True)

##reddit neg
plot = scatter(june26df_cleaned, 'Date', 'Reddit Avg. Neg', 'Reddit_Neg', 'date', 'reddit_neg_sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.savefig('explore/graphs/reddit_sentiment_time.png')
plt.show(block = True)

'''
GUARDIAN
'''

plot = scatter(june26df_cleaned, 'Date', 'Guardian Avg. Pos', 'Guardian Pos', 'date', 'positive sentiment')
# plt.show(block = True)

plot = scatter(june26df_cleaned, 'Date', 'Guardian Avg. Neu', 'Guardian Neu', 'date', 'neutral sentiment')
# plt.show(block = True)

plot = scatter(june26df_cleaned, 'Date', 'Guardian Avg. Neg', 'Guardian Neg', 'date', 'negative sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.savefig('explore/graphs/guardian_sentiment_time.png')
plt.show(block = True)


'''
NYT
'''

plot = scatter(june26df_cleaned, 'Date', 'NYT Avg. Pos', 'NYT Pos', 'date', 'positive sentiment')
# plt.show(block = True)

plot = scatter(june26df_cleaned, 'Date', 'NYT Avg. Neu', 'NYT Neu', 'date', 'neutral sentiment')
# plt.show(block = True)

plot = scatter(june26df_cleaned, 'Date', 'NYT Avg. Neg', 'NYT Neg', 'date', 'negative sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.savefig('explore/graphs/nyt_sentiment_time.png')
plt.show(block = True)


'''
joint pos, neu, neg
'''
plot = scatter(june26df_cleaned, 'Date', 'Avg Joint Pos Sentiment', 'joint sentiment', 'date', 'positive joint')
plt.yticks(np.arange(0.0, 1.0, 0.1))
# plt.show(block = True)


plot = scatter(june26df_cleaned, 'Date', 'Avg Joint Neu Sentiment', 'joint sentiment', 'date', 'neutral joint')
plt.yticks(np.arange(0.0, 1.0, 0.1))
# plt.show(block = True)

plot = scatter(june26df_cleaned, 'Date', 'Avg Joint Neg Sentiment', 'joint sentiment', 'date', 'negative joint')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.savefig('explore/graphs/joint_sentiment_time.png')
plt.show(block = True)


# for row in june26df.iterrows():
#     print(row)

'''
regression
'''

cleaned_table = june26df_cleaned[['Date', 'Avg. Close', 'Avg Joint Pos Sentiment', 'Avg Joint Neu Sentiment', 'Avg Joint Neg Sentiment']]


'''
linear regression
'''

model = sm.OLS(cleaned_table['Avg. Close'], cleaned_table[['Avg Joint Pos Sentiment','Avg Joint Neu Sentiment','Avg Joint Neg Sentiment']])
result = model.fit()
print(result.summary())
prediction = result.predict(after2019df[['Avg Joint Pos Sentiment','Avg Joint Neu Sentiment','Avg Joint Neg Sentiment']])


'''
plotting regression 
'''
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(after2019df['Date'], prediction)
ax.plot(after2019df['Date'], after2019df['Avg. Close'])
plt.xticks(np.arange(0, len(after2019df['Date']), 18))
plt.xticks(rotation = 90)
plt.xlabel('date')
plt.ylabel('stock price')
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig('explore/graphs/ols_prediction1_2018.png')
plt.show(block = True)


# '''logistic regression'''
# X_train, X_test, y_train, y_test = train_test_split(june26df_cleaned, june26df_cleaned['Avg. Close'], test_size = 0.2)

# log_model = LogisticRegression(random_state = 0).fit(X_train, y_train)
# print(log_model.summary())





