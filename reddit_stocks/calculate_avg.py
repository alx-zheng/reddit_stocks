import pandas as pd
import datetime as dt
import numpy as np


# converts datetime string to datetime object
def return_date(x):
    return dt.datetime.strptime(x, '%Y-%m-%d')

# read raw dataframe in
subreddit = "teslamotors"
df = pd.read_csv('data/%s_data.csv' % subreddit, index_col=0)

# start, end, increment (ten day sliding window, add 5 days to each in loop)
NINE_DAYS = dt.timedelta(days=9)
FIVE_DAYS = dt.timedelta(days=5)
start = dt.datetime(2010, 1, 10)
end = dt.datetime(2021, 7, 1)
increment = start + NINE_DAYS



# changes all values in time from string to datetime
df['Time Created'] = df['Time Created'].apply(return_date)

# drop all data from before jan 10th, 2010, and count how many pieces of data that is
dropped = pd.DataFrame(columns=["Subreddit", "Number of Data Points Dropped"])

num_dropped = df.loc[df["Time Created"] < start]

row = {"Subreddit": subreddit,
       "Number of Data Points Dropped": num_dropped.shape[0]}

dropped = dropped.append(row, ignore_index=True)

dropped.to_csv("dropped_data.csv", mode='a')

# create dataframe to store average values
average_columns = ["start_date", "end_date", "avg_neg", "avg_neu", "avg_pos", "num_data"]
average = pd.DataFrame(columns=average_columns)

# loop through raw data
while start < end:

    # check if "next week" needs to be reduced to end
    if increment > end:
        increment = end

    sum_neg = 0
    sum_neu = 0
    sum_pos = 0
    size = 0

    # pulls out rows that are within this 10 day window
    week_df = df.loc[(df['Time Created'] >= start) & (df['Time Created'] <= increment)]

    size += week_df.shape[0]

    # if there were no submissions/comments in this window, put empty in rows,
    # otherwise put in average of the values
    if size == 0:
        avg_row = {"start_date": start,
                   "end_date": increment,
                   "avg_neg": np.nan,
                   "avg_neu": np.nan,
                   "avg_pos": np.nan,
                   "num_data": 0}
        average = average.append(avg_row, ignore_index=True)
    else:
        sum_neg += week_df['neg'].sum()
        sum_neu += week_df['neu'].sum()
        sum_pos += week_df['pos'].sum()
        avg_row = {"start_date": start,
                   "end_date": increment,
                   "avg_neg": sum_neg / size,
                   "avg_neu": sum_neu / size,
                   "avg_pos": sum_pos / size,
                   "num_data": size}
        average = average.append(avg_row, ignore_index=True)

    # moves sliding window 5 days forward
    start += FIVE_DAYS
    increment = start + NINE_DAYS

average.to_csv("averages/%s_average.csv" % subreddit)
