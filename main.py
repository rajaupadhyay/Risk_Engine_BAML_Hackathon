import pandas as pd
import numpy as np
from data_formulation import *
from itertools import combinations
import csv
import sys


# list_of_stock_files = [(googleDataFrame, Ticker), (microsoftDataFrame, Ticker)]

# list_of_stock_files = ["daily_GOOG.csv", "daily_MSFT-1.csv"]


# Function returns Sharpes ratio, Alpha, Beta portfolio,  standard deviation

def get_risk_indicators(list_of_stock_files):
    beta_mean_variance_and_quantity_values = {}
    benchmark_mean = 0
    stocks_change_values = {}
    spindex_file = list_of_stock_files[-1][0]
    stocks = [i[1] for i in list_of_stock_files]


    for dataframe, ticker, quantity in list_of_stock_files[:-1]:
        df, stock_var, stock_mean, benchmark_var, benchmark_mean = stock_calculator(spindex_file, dataframe)

        print("Benchmark variance",benchmark_var)

        # df = pd.read_csv(file.split(".")[0] + "_percent_change.csv")

        stocks_values = df["%change_stocks"].values
        stocks_change_values[ticker] = stocks_values
        benchmark_values = df["%change_benchmark"].values
        cov = np.cov(stocks_values, benchmark_values)
        covariance = cov[0][1]

        print("Stock mean",stock_mean)
        print("Benchmark mean", benchmark_mean)

        # Beta calculation
        print("Beta calculation")
        beta = covariance/benchmark_var
        # print("BETA for {}".format(file), beta)
        beta_mean_variance_and_quantity_values[ticker] = (beta, stock_mean, stock_var, quantity)


    RISK_FREE_RATE = 0.0126/365
    # print(beta_mean_and_variance_values)


    total_capital = sum([i[1]*i[3] for i in beta_mean_variance_and_quantity_values.values()])

    beta_portfolio = sum([i[0]*((i[3]*i[1])/total_capital) for i in beta_mean_variance_and_quantity_values.values()])
    print(beta_portfolio)

    portfolio_mean = sum([i[1]*((i[3]*i[1])/total_capital) for i in beta_mean_variance_and_quantity_values.values()])

    # Alpha calculation
    alpha = portfolio_mean-RISK_FREE_RATE-(beta_portfolio*(benchmark_mean-RISK_FREE_RATE))
    print("ALPHA", alpha)

    weight_array = [(i[3]*i[1])/total_capital for i in beta_mean_variance_and_quantity_values.values()]

    dict_of_weights = {}

    for i in stocks:
        dict_of_weights[i] = weight_array[0]

    # Variance of portfolio

    combinations_of_two = list(combinations(stocks, 2))
    # print("STOCKS ARRAY", stocks_change_values)
    print(combinations_of_two)
    portfolio_of_variance = 0
    for i in combinations_of_two:
        size1 = stocks_change_values[i[0]].size
        size2 = stocks_change_values[i[1]].size
        size = min(size1, size2)

        cov = np.cov(stocks_change_values[i[0]][:size], stocks_change_values[i[1]][:size])
        covariance = cov[0][1]
        print("COVARIANCE:", cov)
        current_pV = dict_of_weights[i[0]]**2 * beta_mean_variance_and_quantity_values[i[0]][2] + dict_of_weights[i[1]]**2 * beta_mean_variance_and_quantity_values[i[1]][2] \
        + 2 * covariance * dict_of_weights[i[0]] * dict_of_weights[i[1]]
        portfolio_of_variance += current_pV

    print("standard deviation of portfolio", math.sqrt(portfolio_of_variance))


    # Sharpes ratio

    sharpes_ratio = (portfolio_mean - RISK_FREE_RATE)/math.sqrt(portfolio_of_variance)
    # sharpes_ratio += 0.5

    standard_deviation = math.sqrt(portfolio_of_variance)

    print(sharpes_ratio)
    return sharpes_ratio, alpha, beta_portfolio,  standard_deviation