import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
from scipy.stats import norm
import random
import pandas as pd

'''
def plot_bell(mean, variance):
    sigma = math.sqrt(variance)
    x = np.linspace(mean-3*sigma, mean+3*sigma,100)
    plt.plot(x,mlab.normpdf(x, mean, sigma))
    plt.show()


# data = np.random.normal(loc=5.0, scale=2.0, size=1000)
# mean,std=norm.fit(data)

x = [random.randint(1,100) for _ in range(20)]

# print(np.var(x))
'''



def stock_calculator(sp_index, dataframe):
    # print("Reading sp_index file into pandas Dataframe")
    # df_spindex = pd.read_csv(sp_index)

    # print("Reading stockprice file into dataframe")
    # df = pd.read_csv(csvfile)
    df_spindex = sp_index
    df = dataframe
    df.insert(6, "closeNew", df_spindex["close"])
    df = df[["close", "closeNew"]]
    df.columns = ["close_stocks", "close_benchmark"]

    print("Raveling columns")
    close_stocks_values = df.close_stocks.values
    close_benchmark_values = df.close_benchmark.values

    close_stocks_values = close_stocks_values[::-1]
    close_benchmark_values = close_benchmark_values[::-1]

    # print(" Max Benchmark Values", max(close_benchmark_values))
    # print(" Min Benchmark Values", min(close_benchmark_values))

    # print(close_stocks_values[:100])

    # print("Calculating mean and variance of stock prices")
    # # Variance and mean calculations
    # # stock_var = np.var(close_stocks_values)
    # stock_mean = sum(list(close_stocks_values)) / len(close_stocks_values)
    # stock_var = sum([(i-stock_mean)**2 for i in close_stocks_values])/(len(close_stocks_values)-1)
    #
    # print("Calculating mean and variance of benchmark index")
    # # benchmark_var = np.var(close_benchmark_values)
    # benchmark_mean = sum(list(close_benchmark_values)) / len(close_benchmark_values)
    # benchmark_var = sum([(i-benchmark_mean)**2 for i in close_benchmark_values])/(len(close_benchmark_values)-1)


    close_stocks_change = []
    close_benchmark_change = []

    print("Calculating % change in stock prices")
    for i in range(1, len(close_stocks_values)):
        close_stocks_change.append(
            (close_stocks_values[i] - close_stocks_values[i - 1]) / close_stocks_values[i - 1])

    print("Calculating % change in benchmark values")
    for i in range(1, len(close_benchmark_values)):
        close_benchmark_change.append(
            (close_benchmark_values[i] - close_benchmark_values[i - 1]) / close_benchmark_values[i - 1])

    close_stocks_change = np.array(close_stocks_change)

    close_benchmark_change = np.array(close_benchmark_change)

    print("Calculating mean and variance of close stock prices")
    # Variance and mean calculations
    stock_var = np.var(close_stocks_change)
    stock_mean = sum(list(close_stocks_change)) / len(close_stocks_change)
    # stock_var = sum([(i - stock_mean) ** 2 for i in close_stocks_values]) / (len(close_stocks_values) - 1)

    print("Calculating mean and variance of close benchmark index")
    benchmark_var = np.var(close_benchmark_change)
    benchmark_mean = sum(list(close_benchmark_change)) / len(close_benchmark_change)
    # benchmark_var = sum([(i - benchmark_mean) ** 2 for i in close_benchmark_values]) / (len(close_benchmark_values) - 1)


    df_change = pd.DataFrame(close_stocks_change)
    # print(df_change.head())

    df_change.insert(1, "change_benchmark", close_benchmark_change)

    df_change.columns = ["%change_stocks", "%change_benchmark"]

    # print("Writing % changes to CSV")
    # df_change.to_csv(csvfile.split(".")[0]+ "_percent_change" + ".csv", index=False)

    # print("Calculation on {} complete".format(csvfile))
    return df_change, stock_var, stock_mean, benchmark_var, benchmark_mean

# res = stock_calculator("daily_SPY-1.csv", ["daily_GOOG.csv", "daily_MSFT-1.csv"])


# print(res)



'''

dfMSFT = pd.read_csv("daily_MSFT-1.csv")

# dfMSFT = dfMSFT["close"]

dfSPY = pd.read_csv("daily_SPY-1.csv")

dfMSFT.insert(6,"closeNew",dfSPY["close"])

# dfMSFT['close'] = dfSPY['close']
# print(dfMSFT.head())

dfMSFT = dfMSFT[["close", "closeNew"]]

dfMSFT.columns = ["close_stocks", "close_benchmark"]

print(dfMSFT.head())



close_stocks_values = dfMSFT.close_stocks.values

close_benchmark_values = dfMSFT.close_benchmark.values

# variance and mean calculations
stock_var = np.var(close_stocks_values)
stock_mean = sum(list(close_stocks_values))/len(close_stocks_values)

benchmark_var = np.var(close_benchmark_values)
benchmark_mean = sum(list(close_benchmark_values))/len(close_benchmark_values)




close_stocks_change = []
close_benchmark_change = []

for i in range(1, len(close_stocks_values)):
    close_stocks_change.append((close_stocks_values[i]-close_stocks_values[i-1])/close_stocks_values[i-1])


for i in range(1, len(close_benchmark_values)):
    close_benchmark_change.append((close_benchmark_values[i]-close_benchmark_values[i-1])/close_benchmark_values[i-1])

close_stocks_change = np.array(close_stocks_change)

close_benchmark_change = np.array(close_benchmark_change)

df_change = pd.DataFrame(close_stocks_change)
print(df_change.head())

df_change.insert(1,"change_benchmark",close_benchmark_change)

df_change.columns = ["%change_stocks", "%change_benchmark"]

print(df_change.head())

df_change.to_csv("df_change.csv", index=False)

'''