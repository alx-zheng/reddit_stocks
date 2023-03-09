from datetime import datetime
import os
import math
from time import strptime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import DateFormatter
import pandas as pd
import statsmodels.api as sm
import sklearn.metrics

feature_columns = ["Date", "Avg. Close"]#, "Guardian Avg. Pos", "Guardian Avg. Neu", "Guardian Avg. Neg", "Reddit Avg. Pos", "Reddit Avg. Neu", "Reddit Avg. Neg", "NYT Avg. Pos", "NYT Avg. Neu", "NYT Avg. Neg"]

def create_dir(name): 
    '''
    creates stock directory if it does not exist

    :param name: name of the directory
    '''
    if not os.path.exists(name):
        os.makedirs(name)

def plot_data(predict_Y, true_Y, title, short_title, stock):
    '''
    plots the predicted values against the true values and saves the plot under the stock directory

    :param predict_Y: predicted values
    :param true_Y: true values
    :param stock: stock name
    '''

    pred_dates = [datetime.strptime(d, '%m/%d/%Y') for d in predict_Y[:,1]]
    true_dates = [datetime.strptime(d, '%m/%d/%Y') for d in true_Y[:,1]]

    plt.figure()
    fig, ax = plt.subplots()
    plt.plot_date(pred_dates, predict_Y[:,0], label = 'Predicted', linestyle='solid', marker='None')
    plt.plot_date(true_dates, true_Y[:,0], label="Actual", linestyle='solid', marker='None')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.legend()
    plt.xlabel("Date")
    plt.xticks(rotation = 60)
    plt.ylabel(title)
    plt.title(stock + ': Autoregression Prediction for ' + title)
    plt.savefig(stock + "_percent/lin_reg_"+short_title+"_prediction.png", bbox_inches="tight")
    plt.show()

def read_data(path):
    """
    Reads the data at the provided files path. 

    :param path: path to dataset
    :return: raw dates, raw numeric data
    """
    # Load the data set into a 2D numpy array
    with open(path) as data_file:
        data = pd.read_csv(data_file)[feature_columns].to_numpy()

    dates = data[:, 0].astype(datetime)
    
    num_data = np.delete(data, 0, 1).astype('float64')

    return num_data, dates

def divide_by_date(date, data, dates): 
    find_date = 0
    for i in range(dates.shape[0]): 
        if datetime.strptime(dates[i], '%m/%d/%Y') >= date: 
            find_date = i
            break

    train_data = data[:find_date, :]
    test_data = data[find_date:, :]

    return train_data, test_data, dates[:find_date], dates[find_date:]

def elbow_point_plot(windows, error, error_label, error_lbl_short, stock):
    """
    This function helps create and save a plot representing the tradeoff between the
    number of windows and the error values. 

    :param windows: 1D np array that represents the number of windows
    :param error: 1D np array that represents the error values
    """
    plt.clf()
    plt.plot(windows, error)
    plt.xlabel('Number of Windows')
    plt.ylabel(error_label)
    plt.title('Elbow Plot for Number of Windows and ' + error_label + " for " + stock)
    plt.savefig(stock + "_percent/xWindow_vs_" + error_lbl_short + ".png")
    plt.show()

def min_max_scale(data):
    """
    Pre-processes the data by performing MinMax scaling.

    MinMax scaling prevents different scales of the data features from
    influencing distance calculations.

    MinMax scaling is performed by
        X_new = (X - X_min) / (X_max - X_min),

    where X_new is the newly scaled value, X_min is the minimum and X_max is the
    maximum along a single feature column.

    :param data: 2D numpy array of raw data
    :return: preprocessed data
    """
    for i in range(data.shape[1]): 
        min = data[0][i]
        max = data[0][i]
        for j in range(data.shape[0]):
            if data[j][i] < min: 
                min = data[j][i]
            elif data[j][i] > max: 
                max = data[j][i]
        for j in range(data.shape[0]):
            data[j][i] = (data[j][i] - min) / (max - min)
    return data

def change_to_perc(data): 
    Y = data[:, 0]
    np.reshape(Y[:Y.shape[0] - 1], (data.shape[0] - 1, -1))
    new = np.reshape(Y[1:], (Y.shape[0] - 1, -1))
    last = np.reshape(Y[:Y.shape[0] - 1], (Y.shape[0] - 1, -1))
    Y = np.concatenate((last, new), axis = 1)
    Y = list(map(lambda row: ((row[1] - row[0]) / row[0]), Y))
    return np.insert(Y, 0, 0)


def create_lin_reg(num_wind, data):
    '''
    First, it creates an np array that holds num_wind windows of data per day. In doing so, we can get data for num_wind number of rows of data we have. 
    Then it runs a linear regression on this array against the prices of the next window. 

    :param num_wind: an integer number of windows to looks at
    :param data: a 2D np array of data on which to work on
    :return: sm.OLS object
    '''
    clean_data, Y, skipped = data_to_windows(data[0], num_wind)
    for train_data in data[1:]:
        temp_clean_data, temp_Y, temp_skipped = data_to_windows(train_data, num_wind)
        skipped = np.concatenate((skipped, [x + clean_data.shape[0] for x in temp_skipped]))
        clean_data = np.concatenate((clean_data, temp_clean_data), axis = 0)
        Y = np.concatenate((Y, temp_Y), axis = 0)

    clean_data = sm.add_constant(clean_data)
    return sm.OLS(Y, clean_data), skipped

def data_to_windows(data, num_wind):
    length = data.shape[0]
    num_data = data[num_wind - 1:length - 1, 1:]
    Y = data[num_wind:, 0]

    for i in range(1, num_wind): 
        next_wind = data[num_wind - i - 1:length - i - 1, 1:]
        num_data = np.concatenate((num_data, next_wind), axis = 1)

    clean_data = num_data
    skipped = []
    for i in reversed(range(length - num_wind)): 
        if np.isnan(np.sum(num_data[i, :])):
            clean_data = np.delete(clean_data, i, 0)
            Y = np.delete(Y, i, 0)
            skipped.append(i)

    return clean_data, Y, skipped

def find_best_model(num, train_data, stock): 
    '''
    Creates error graphs for adjusted r^2 values, MSE values, and MSE residual values of linear regression models using window size of 1 to num. 

    :param num: an integer representing up to how many windows to look at
    :param data: a 2D np array of data on which to work on
    '''
    xWindows = range(1, num)
    r_sq_list = []
    rmse_list = []

    for i in xWindows: 
        res = create_lin_reg(i, train_data)[0].fit()

        test_data, true_value = data_to_windows(train_data[0], i)[:2]
        for temp_train_data in train_data[1:]:
            temp_clean_data, temp_Y = data_to_windows(temp_train_data, i)[:2]
            test_data = np.concatenate((test_data, temp_clean_data), axis = 0)
            true_value = np.concatenate((true_value, temp_Y), axis = 0)
        prediction = res.predict(sm.add_constant(test_data))

        r_sq_list.append(sklearn.metrics.r2_score(true_value, prediction))
        rmse_list.append(math.sqrt(sklearn.metrics.mean_squared_error(true_value, prediction)))
    
    elbow_point_plot(np.array(range(1, len(r_sq_list) + 1)), np.array(r_sq_list), "R-Squared Value", "r_sq", stock)
    elbow_point_plot(np.array(range(1, len(rmse_list) + 1)), np.array(rmse_list), "Root Mean Square Error", "rmse", stock)

def plot_prediction(train_data, test_data, prices, dates, num_wind, using_test): 
    best_fit, skipped = create_lin_reg(num_wind, train_data)
    best_fit = best_fit.fit()
    # print(best_fit.summary())

    clean_test_data, true_val, skipped = data_to_windows(test_data, num_wind)
    clean_test_data = sm.add_constant(clean_test_data)
    prediction = best_fit.predict(clean_test_data)

    result = np.zeros(prediction.shape[0])

    result[0] = prices[num_wind - 1]
    for i in range(1, result.shape[0]): 
        result[i] = (1 + prediction[i]) * result[i-1]

    dates = dates[num_wind:]
    mod_dates = dates
    cor_prices = prices[num_wind:]
    for i in skipped:
        mod_dates = np.delete(mod_dates, i, 0)
        cor_prices = np.delete(cor_prices, i, 0)

    with open(stock + "_percent/statistics.txt", "a") as f:
        f.seek(4)
        if using_test:
            f.write("Root Mean Squared Error of Prices of Test Data on " + str(num_wind) + " windows: %f\n" % math.sqrt(sklearn.metrics.mean_squared_error(cor_prices, result)))
            f.write("R squared of Prices of Test Data on " + str(num_wind) + " windows: %f\n" % sklearn.metrics.r2_score(cor_prices, result))
            f.write("Root Mean Squared Error of Percent Change of Test Data on " + str(num_wind) + " windows: %f\n" % math.sqrt(sklearn.metrics.mean_squared_error(true_val, prediction)))
            f.write("R squared of Percent Change of Test Data on " + str(num_wind) + " windows: %f\n" % sklearn.metrics.r2_score(true_val, prediction))
            f.write("%f\n")
        else: 
            f.write("Root Mean Squared Error of Prices on " + str(num_wind) + " windows: %f\n" % math.sqrt(sklearn.metrics.mean_squared_error(cor_prices, result)))
            f.write("R squared of Prices on " + str(num_wind) + " windows: %f\n" % sklearn.metrics.r2_score(cor_prices, result))
            f.write("Root Mean Squared Error of Percent Change on " + str(num_wind) + " windows: %f\n" % math.sqrt(sklearn.metrics.mean_squared_error(true_val, prediction)))
            f.write("R squared of Percent Change on " + str(num_wind) + " windows: %f\n" % sklearn.metrics.r2_score(true_val, prediction))
        f.close()

    mod_date_len = mod_dates.shape[0]
    date_len = dates.shape[0]

    mod_fit_val = np.concatenate((np.reshape(prediction, (mod_date_len, -1)), np.reshape(mod_dates, (mod_date_len, -1))), axis = 1)

    mod_orig_val = np.concatenate((np.reshape(test_data[num_wind:, 0], (date_len, -1)), np.reshape(dates, (date_len, -1))), axis = 1)

    if using_test:
        plot_data(mod_fit_val, mod_orig_val, "Percent Change with " + str(num_wind) + " windows ", "per_ch_test" + str(num_wind), stock)
    else: 
        plot_data(mod_fit_val, mod_orig_val, "Percent Change with " + str(num_wind) + " windows ", "per_ch" + str(num_wind), stock)

    mod_fit_price = np.concatenate((np.reshape(result, (mod_date_len, -1)), np.reshape(mod_dates, (mod_date_len, -1))), axis = 1)
    mod_orig_price = np.concatenate((np.reshape(prices[num_wind:], (date_len, -1)), np.reshape(dates, (date_len, 1))), axis = 1)
    if using_test:
        plot_data(mod_fit_price, mod_orig_price, "Price with " + str(num_wind) + " windows ", "price_test_" + str(num_wind), stock)
    else: 
        plot_data(mod_fit_price, mod_orig_price, "Price with " + str(num_wind) + " windows ", "price" + str(num_wind), stock)

if __name__ == '__main__':
    stocks = ["AAPL", "MSFT", "GOOGL"]
    for stock in stocks: 
        create_dir(stock + "_percent")
        path = "joined/" + stock + "_joined_full.csv"
        num_data, dates = read_data(path)
        num_data[:, 1:] = min_max_scale(num_data[:, 1:])
        perc_change = change_to_perc(num_data)

        model_data = np.concatenate((np.reshape(perc_change, (perc_change.shape[0], -1)), num_data[:, 1:]), axis = 1)
        pre_2015_data, post_2015_data, pre_2015_dates, post_2015_dates = divide_by_date(datetime(2015, 1, 1), num_data, dates)
        price = divide_by_date(datetime(2016, 7, 1), post_2015_data, post_2015_dates)[0][:, 0]
        pre_2015_data, post_2015_data, pre_2015_dates, post_2015_dates = divide_by_date(datetime(2015, 1, 1), model_data, dates)
        pre_2016_7_data, post_2016_7_data, pre_2016_7_dates, post_2016_7_dates = divide_by_date(datetime(2016, 7, 1), model_data, post_2015_dates)

        find_best_model(50, [pre_2015_data, post_2016_7_data], stock)

        if stock == "AAPL": 
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 28, True)
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 44, True)
            plot_prediction([model_data], model_data, num_data[:, 0], dates, 28, False)
        elif stock == "MSFT": 
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 28, True)
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 44, True)
            plot_prediction([model_data], model_data, num_data[:, 0], dates, 28, False)
        elif stock == "GOOGL": 
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 28, True)
            # plot_prediction([pre_2015_data, post_2016_7_data], pre_2016_7_data, price, pre_2016_7_dates, 44, True)
            plot_prediction([model_data], model_data, num_data[:, 0], dates, 28, False)
    