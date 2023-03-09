import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.api as sm
from statsmodels.tools import eval_measures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor


import csv

subreddit = "GOOGL"
df = pd.read_csv('explore/%s_joined_full.csv' % subreddit)

df["Date"] = df["Date"].astype(str)
df["datetime"] = pd.to_datetime(df['Date'])
df["Avg. Close"] = df["Avg. Close"].astype(float)
df["Avg Joint Pos Sentiment"] = ((df["Reddit Avg. Pos"] + df["Guardian Avg. Pos"] + df["NYT Avg. Pos"])/3).astype(float)
df["Avg Joint Neu Sentiment"] = ((df["Reddit Avg. Neu"] + df["Guardian Avg. Neu"] + df["NYT Avg. Neu"])/3).astype(float)
df["Avg Joint Neg Sentiment"] = ((df["Reddit Avg. Neg"] + df["Guardian Avg. Neg"] + df["NYT Avg. Neg"])/3).astype(float)


before = (df['datetime'] > dt.datetime(2010,1,15)) & (df['datetime'] <= dt.datetime(2018,1,1))
after = (df['datetime'] > dt.datetime(2018,1,1)) & (df['datetime'] <= dt.datetime(2021,6,30))


train = df.loc[before]
test = df.loc[after]

train = train[~train.isin([np.nan, np.inf, -np.inf]).any(1)]
test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]



# print(df["Guardian Avg. Pos"].mean())
# print(df["Guardian Avg. Neu"].mean())
# print(df["Guardian Avg. Neg"].mean())
# print(df["NYT Avg. Pos"].mean())
# print(df["NYT Avg. Neu"].mean())
# print(df["NYT Avg. Neg"].mean())
# print(df["Reddit Avg. Pos"].mean())
# print(df["Reddit Avg. Neu"].mean())
# print(df["Reddit Avg. Neg"].mean())


def scatter(table, xaxis, yaxis, title, xlabel, ylabel):
    x = table[xaxis]
    y = table[yaxis]
    plot = plt.plot(x,y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #manipulate x-axis
    plt.xticks(np.arange(0, len(x), 30))
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
plot = scatter(df, 'Date', 'Reddit Avg. Pos', 'Reddit_Sentiment', 'date', 'nltk_sentiment')

# plt.show(block = True)

##reddit neu
plot = scatter(df, 'Date', 'Reddit Avg. Neu', 'Reddit_Sentiment', 'date', 'nltk_sentiment')
# plt.show(block = True)

##reddit neg
plot = scatter(df, 'Date', 'Reddit Avg. Neg', 'Reddit_Sentiment', 'date', 'nltk_sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.05))
plt.savefig('explore/graphs/reddit_sentiment_time.png')
plt.show(block = True)

'''
GUARDIAN
'''

plot = scatter(df, 'Date', 'Guardian Avg. Pos', 'Guardian_Sentiment', 'date', 'nltk_sentiment')
# plt.show(block = True)

plot = scatter(df, 'Date', 'Guardian Avg. Neu', 'Guardian_Sentiment', 'date', 'nltk_sentiment')
# plt.show(block = True)

plot = scatter(df, 'Date', 'Guardian Avg. Neg', 'Guardian_Sentiment', 'date', 'nltk_sentiment')
plt.yticks(np.arange(0.0, 1.0,0.05))
plt.savefig('explore/graphs/guardian_sentiment_time.png')
plt.show(block = True)


'''
NYT
'''

plot = scatter(df, 'Date', 'NYT Avg. Pos', 'NYT_Sentiment', 'date', 'nltk_sentiment')
# plt.show(block = True)

plot = scatter(df, 'Date', 'NYT Avg. Neu', 'NYT_Sentiment', 'date', 'nltk_sentiment')
# plt.show(block = True)

plot = scatter(df, 'Date', 'NYT Avg. Neg', 'NYT_Sentiment', 'date', 'nltk_sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.05))
plt.savefig('explore/graphs/nyt_sentiment_time.png')
plt.show(block = True)


'''
joint pos, neu, neg
'''
plot = scatter(df, 'Date', 'Avg Joint Pos Sentiment', 'joint sentiment', 'date', 'nltk_sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.1))
# plt.show(block = True)


plot = scatter(df, 'Date', 'Avg Joint Neu Sentiment', 'joint sentiment', 'date', 'nltk_sentiment')
plt.yticks(np.arange(0.0, 1.0, 0.1))
# plt.show(block = True)

plot = scatter(df, 'Date', 'Avg Joint Neg Sentiment', 'joint sentiment', 'date', 'negative joint')
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.savefig('explore/graphs/joint_sentiment_time.png')
plt.show(block = True)


# for row in june26df.iterrows():
#     print(row)

'''
regression
'''

# training_table = june26df[['Date', 'Avg. Close', 'Guardian Avg. Pos', 'Guardian Avg. Neu', 'Guardian Avg. Neg', 'Reddit Avg. Pos', 'Guardian Avg. Neu', 'Guardian Avg. Neg', 'NYT Avg. Pos', 'NYT Avg. Neu', 'NYT Avg. Neg']]


'''
linear regression
'''

model = sm.OLS(train['Avg. Close'], sm.add_constant(train[['Guardian Avg. Pos', 'Guardian Avg. Neu', 'Guardian Avg. Neg', 'Reddit Avg. Pos', 'Guardian Avg. Neu', 'Guardian Avg. Neg', 'NYT Avg. Pos', 'NYT Avg. Neu', 'NYT Avg. Neg']]))
result = model.fit()
print(result.summary())
prediction = result.predict(sm.add_constant(test[['Guardian Avg. Pos','Guardian Avg. Neu','Guardian Avg. Neg','Reddit Avg. Pos','Reddit Avg. Neu','Reddit Avg. Neg','NYT Avg. Neg','NYT Avg. Neu','NYT Avg. Pos']]))

print(mean_squared_error(test['Avg. Close'], prediction))



'''
plotting regression 
'''
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(test['Date'], prediction, label = 'Prediction')
ax.plot(test['Date'], test['Avg. Close'], label = 'True value')
plt.legend()
plt.title('GOOGL Stock Price calculated on Sentiment Scores from Reddit and News Sources')
plt.xticks(np.arange(0, len(test['Date']), 30))
plt.xticks(rotation = 90)
plt.xlabel('date')
plt.ylabel('GOOGL Stock Price in Dollars')
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig('explore/graphs/GOOGL_prediction_2018.png')
plt.show(block = True)


# '''logistic regression'''
# X_train, X_test, y_train, y_test = train_test_split(june26df_cleaned, june26df_cleaned['Avg. Close'], test_size = 0.2)

# log_model = LogisticRegression(random_state = 0).fit(X_train, y_train)
# print(log_model.summary())



'''
dummy regression
'''

dummy_model = DummyRegressor(strategy = 'mean')
dummy_model.fit(sm.add_constant(test[['Guardian Avg. Pos','Guardian Avg. Neu','Guardian Avg. Neg','Reddit Avg. Pos','Reddit Avg. Neu','Reddit Avg. Neg','NYT Avg. Neg','NYT Avg. Neu','NYT Avg. Pos']]), test['Avg. Close'])
dummy_prediction = dummy_model.predict(sm.add_constant(test[['Guardian Avg. Pos','Guardian Avg. Neu','Guardian Avg. Neg','Reddit Avg. Pos','Reddit Avg. Neu','Reddit Avg. Neg','NYT Avg. Neg','NYT Avg. Neu','NYT Avg. Pos']]))
print(mean_squared_error(test['Avg. Close'], dummy_prediction))


'''
plotting dummy 
'''
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(test['Date'], dummy_prediction, label = 'prediction')
ax.plot(test['Date'], test['Avg. Close'], label = 'True value')
plt.legend()
plt.title('GOOGL Stock Price calculated on Dummy Model Predicting using mean value')
plt.xticks(np.arange(0, len(test['Date']), 30))
plt.xticks(rotation = 90)
plt.xlabel('date')
plt.ylabel('GOOGL Stock Price in Dollars')
plt.gcf().subplots_adjust(bottom=0.2)
plt.savefig('explore/graphs/GOOGL_dummy_prediction_2018.png')
plt.show(block = True)



