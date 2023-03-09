from datetime import datetime
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import DateFormatter
import pandas as pd
import statsmodels.api as sm
import sklearn.metrics

feature_columns = ["Date", "Avg. Close", "Guardian Avg. Pos", "Guardian Avg. Neu", "Guardian Avg. Neg", "Reddit Avg. Pos", "Reddit Avg. Neu", "Reddit Avg. Neg", "NYT Avg. Pos", "NYT Avg. Neu", "NYT Avg. Neg"]
COLORS = matplotlib.cycler('color', ['#8fbcbb', '#81a1c1', '#bf616a', '#d08770', '#ebcb8b', '#a3be8c', '#b48ead'])
matplotlib.rcParams['axes.prop_cycle'] = COLORS

def create_dir(name): 
    '''
    creates stock directory if it does not exist

    :param name: name of the directory
    '''
    if not os.path.exists(name):
        os.makedirs(name)

def plot_data(predict_Y, true_Y, stock):
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
    plt.ylabel("Price")
    plt.title('Linear Regression Prediction for '+stock)
    plt.savefig(stock + "_xWind/lin_reg_prediction.png")
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
    plt.savefig(stock + "_xWind/xWindow_vs_" + error_lbl_short + ".png")
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

def create_lin_reg(num_wind, data):
    '''
    First, it creates an np array that holds num_wind windows of data per day. In doing so, we can get data for num_wind number of rows of data we have. 
    Then it runs a linear regression on this array against the prices of the next window. 

    :param num_wind: an integer number of windows to looks at
    :param data: a 2D np array of data on which to work on
    :return: sm.OLS object
    '''
    length = data.shape[0]
    num_data = data[num_wind - 1:length - 1, :]
    Y = data[num_wind:, 0]

    for i in range(1, num_wind - 1): 
        next_wind = data[num_wind - i - 1:length - i - 1, :]
        num_data = np.concatenate((num_data, next_wind), axis = 1)
    
    clean_data = num_data
    skipped = []
    for i in reversed(range(length - num_wind)): 
        if np.isnan(np.sum(num_data[i, :])):
            clean_data = np.delete(clean_data, i, 0)
            Y = np.delete(Y, i, 0)
            skipped.append(i)

    clean_data = sm.add_constant(clean_data)
    return sm.OLS(Y, clean_data), skipped

def find_best_model(num, data, stock): 
    '''
    Creates error graphs for adjusted r^2 values, MSE values, and MSE residual values of linear regression models using window size of 1 to num. 

    :param num: an integer representing up to how many windows to look at
    :param data: a 2D np array of data on which to work on
    '''
    xWindows = range(1, num)
    adj_r_sq_list = []
    r_sq_list = []
    mse_total = []
    mse_resid = []

    for i in xWindows: 
        res = create_lin_reg(i, data)[0].fit()
        adj_r_sq_list.append(res.rsquared_adj)
        r_sq_list.append(res.rsquared)
        mse_total.append(res.mse_total)
        mse_resid.append(res.mse_resid)
    
    elbow_point_plot(np.array(xWindows), np.array(adj_r_sq_list), "R-Squared Value", "r_sq", stock)
    elbow_point_plot(np.array(xWindows), np.array(adj_r_sq_list), "Adjusted R-Squared Value", "adj_r_sq", stock)
    elbow_point_plot(np.array(xWindows), np.array(mse_total), "MSE Value", "mse_tot", stock)
    elbow_point_plot(np.array(xWindows), np.array(mse_resid), "MSE Residual Value", "mse_res", stock)


if __name__ == '__main__':
    stocks = ["AAPL", "MSFT", "GOOGL"]
    for stock in stocks: 
        create_dir(stock + "_xWind")
        path = "joined/" + stock + "_joined_full.csv"
        num_data, dates = read_data(path)
        num_data = min_max_scale(num_data)
        find_best_model(50, num_data, stock)

        best_fit, skipped = create_lin_reg(1, num_data)

        best_fit = best_fit.fit()
        cor_true_val = num_data[1:, 0]
        # print(best_fit.summary())

        dates = np.delete(dates, 0, 0)
        mod_dates = dates
        for i in skipped: 
            mod_dates = np.delete(mod_dates, i, 0)
            cor_true_val = np.delete(cor_true_val, i, 0)

        predict_Y = best_fit.fittedvalues

        with open(stock + "_xWind/statistics.txt", "w") as f:
            f.seek(0)
            f.write("Root Mean Squared Error: %f\n" % math.sqrt(sklearn.metrics.mean_squared_error(cor_true_val, predict_Y)))
            f.write("R squared: %f" % sklearn.metrics.r2_score(cor_true_val, predict_Y))
            f.close()

        mod_date_len = mod_dates.shape[0]
        date_len = dates.shape[0]

        mod_fit_val = np.concatenate((np.reshape(predict_Y, (mod_date_len, -1)), np.reshape(mod_dates, (mod_date_len, -1))), axis = 1)
        mod_orig_val = np.concatenate((np.reshape(num_data[1:, 0], (date_len, -1)), np.reshape(dates, (date_len, 1))), axis = 1)
        plot_data(mod_fit_val, mod_orig_val, stock)
    