import requests
import pandas as pd

API_KEY = "GTUVILTORCGMA9MO"
BASE_URL = "https://www.alphavantage.co/query"

def get_equity_data(symbol: str, df: pd.DataFrame):
    """Modify df to include the historical adjusted close price of symbol since 2010-01-01"""
    query_params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": API_KEY
            }
    r = requests.get(BASE_URL, params=query_params)
    data = r.json()
    for c,i in enumerate(data["Time Series (Daily)"]):
        #Add data to dataframe for every date since 2010
        if i == "2010-01-04":
            break
        df.loc[c] = [i, data["Time Series (Daily)"][i]["5. adjusted close"]]

if __name__ == "__main__":
    symbols = pd.read_csv("./QQQ_constituents.csv")
    symbols = symbols['Symbol'].values.tolist()
    symbols = symbols[:-2]
    for i in symbols:
        df = pd.DataFrame(columns=["Date", "Adj. Close"])
        get_equity_data(i, df)
        path = './equity_data/' + i + '.csv'
        df.to_csv(path, index=False, header=True)
