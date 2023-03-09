import pandas as pd
from pmaw import PushshiftAPI
import datetime as dt
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def get_neg(txt):
    return sia.polarity_scores(txt)['neg']


def get_neu(txt):
    return sia.polarity_scores(txt)['neu']


def get_pos(txt):
    return sia.polarity_scores(txt)['pos']


def nltk_analysis(df):
    df['neg'] = df['Text'].apply(get_neg)
    df['neu'] = df['Text'].apply(get_neu)
    df['pos'] = df['Text'].apply(get_pos)
    return df


api = PushshiftAPI()
WEEK_IN_SECONDS = 604800
subreddit = "google"
limit = 500

start = int(dt.datetime(2010, 1, 1, 0, 0).timestamp())
end = int(dt.datetime(2021, 7, 1, 0, 0).timestamp())
week_after_current = start + WEEK_IN_SECONDS

print_headers = True

while start < end:

    # check if "next week" needs to be reduced to end
    if week_after_current > end:
        week_after_current = end

    # scrape comments
    comments = api.search_comments(q=subreddit, subreddit=subreddit,
                                   limit=limit,
                                   after=start, before=week_after_current)
    comments_df = pd.DataFrame(
        data=[[dt.datetime.fromtimestamp(comment["created_utc"]).strftime(
            '%Y-%m-%d'), comment["body"]]
              for comment in comments],
        columns=['Time Created', 'Text'])
    comments_df = nltk_analysis(comments_df)
    comments_df.to_csv("%s_data.csv" % subreddit, mode='a',
                       header=print_headers)
    print_headers = False

    # scrape submissions
    time_created = []
    text = []
    submissions = api.search_submissions(subreddit=subreddit, limit=limit,
                                         after=start, before=week_after_current)
    for submission in submissions:
        time_created.append(
            dt.datetime.fromtimestamp(submission["created_utc"]).strftime(
                '%Y-%m-%d'))
        try:
            text.append(submission["title"] + submission["selftext"])
        except KeyError:
            text.append(submission["title"])
    submissions_df = pd.DataFrame(list(zip(time_created, text)),
                                  columns=['Time Created', 'Text'])
    submissions_df = nltk_analysis(submissions_df)
    submissions_df.to_csv("%s_data.csv" % subreddit, mode='a',
                          header=print_headers)


    start += WEEK_IN_SECONDS
    week_after_current += WEEK_IN_SECONDS
