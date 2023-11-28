from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar

plt.style.use("seaborn-v0_8-white")
pd.options.mode.chained_assignment = None
# original pandas datareader does not work, use yfinance to override
yf.pdr_override()


class backtest:

    def __init__(self, ticker, commission, shares_per_trade=2, start=1, end="now"):
        """
        Args:
            ticker (str): ticker symbol from yahoo finance
            commission (float): commission cost per trade
            shares_per_trade (int): number shares purchased per trade
            start (int): number years from end date
            end (int): default "now" = current date, or enter specific year, i.e., 2021 (year end)
        """
        self.ticker = ticker
        self.end = datetime.now() if end == "now" else datetime(end, 12, 31)
        self.start = datetime(self.end.year-start, self.end.month, self.end.day)
        self.commission = commission
        self.shares_per_trade = shares_per_trade

        self.df = web.get_data_yahoo(ticker, self.start, self.end)
        self.current_price = self.df["Close"].iloc[-1]


    def dollar_cost_averaging(self):
        """DCA, buy once per month on first non-holiday weekday"""
        
        # get first business day of month, excluding holidays
        cal = USFederalHolidayCalendar()
        buy_dates = []
        for date in pd.date_range(self.start, self.end, freq="BMS"):
            while date.isoweekday() > 5 or date in cal.holidays():
                date += timedelta(days=1)    
            buy_dates.append(date.date().strftime("%Y-%m-%d"))

        buy_df = self.df[self.df.index.isin(buy_dates)]
        buy_df["date"] = buy_df.index
        buy_df["shares"] = self.shares_per_trade

        buy_df = buy_df[["Close", "date", "shares"]]
        return buy_df


    def simple_moving_average(self, sma=30, rm_shares=False):
        """SMA, buy once a month when price drop below specified SMA, SMA acts as a support"""
        
        sma_df = self.df.copy()
        col_name = "SMA" + str(sma)
        sma_df[col_name] = sma_df["Close"].rolling(sma).mean()

        # get instances where price < SMA
        sma_df["buy"] = sma_df.apply(lambda x: True if x["Close"] < x[col_name] else False, axis=1)
        sma_df = sma_df[sma_df["buy"]==True]
        
        # get first SMA drop per month
        sma_df["date"] = sma_df.index
        dfg = sma_df.groupby(pd.Grouper(freq="M")).agg(min_date=("date", "min")).reset_index()
        buy_dates = dfg["min_date"].dt.strftime("%Y-%m-%d").tolist()
        
        # if month has zero buys, will have nan
        buy_dates = [date for date in buy_dates if date is not np.nan]
        sma_df = sma_df[sma_df.index.isin(buy_dates)]
        
        sma_df["shares"] = self.shares_per_trade
        sma_df = sma_df[["Close", "date", "shares", col_name]]

        # for sma_multi() strategy to work
        if rm_shares:
            del sma_df["shares"]

        return sma_df
    
    
    def sma_crossing(self, sma1=30, sma2=50):
        """Buy when sma1 crosses sma2"""

        sma_df = self.df.copy()
        col_sma1 = "SMA" + str(sma1)
        col_sma2 = "SMA" + str(sma2)
        sma_df[col_sma1] = sma_df["Close"].rolling(sma1).mean()
        sma_df[col_sma2] = sma_df["Close"].rolling(sma2).mean()
        
        def cross_validation(x):
            if pd.isnull(x[col_sma2]):
                return np.nan
            elif x[col_sma1] > x[col_sma2]:
                return True
            else:
                return False
                
        sma_df["cross"] = sma_df.apply(lambda x: cross_validation(x), axis=1)
        sma_df["shift_down"] = sma_df["cross"].shift(periods=1)
        sma_df["buy"] = sma_df.apply(lambda x: True if x["cross"] == True and x["shift_down"] == False \
                                                 else False, axis=1)
        sma_df = sma_df[sma_df["buy"]==True]
        sma_df["date"] = sma_df.index
        sma_df["shares"] = self.shares_per_trade
        sma_df = sma_df[["Close", "date", "shares", col_sma1, col_sma2]]

        return sma_df

    
    def sma_multi(self, sma_list=[30,50,100,200,300], share_list=[2,2,4,4,6]):
        """Buy if price cross below each SMA
        specify shares for each SMA drop"""
        
        if len(sma_list) != len(share_list):
            raise ValueError("sma_list & share_list must have the same length")

        df_list = [self.simple_moving_average(sma, rm_shares=True) for sma in sma_list]
        df = pd.concat(df_list, axis=1)
        del df["Close"]
        del df["date"]
        df = df.merge(self.df["Close"], left_index=True, right_index=True)

        # remove buy of same day for each SMA crossover, retain largest sma
        def none_cal(x, col_sma, col_last):
            if not pd.isnull(x[col_sma]) and not pd.isnull(x[col_last]):
                return np.nan
            else:
                return x[col_sma]

        # remove sma if have another longer sma on same day
        sma_list_ = sma_list
        while len(sma_list_) > 1:
            col_last = "SMA" + str(sma_list_[-1])
            for sma in sma_list_[:-1]:
                col_sma = "SMA" + str(sma)
                df[col_sma] = df.apply(lambda x: none_cal(x, col_sma, col_last), axis=1)
            sma_list_ = sma_list_[:-1]

        df["buy"] = True
        df["date"] = df.index

        # add num shares
        def weight(x):
            for sma, share in zip(sma_list, share_list):
                col_sma = "SMA" + str(sma)
                if not pd.isnull(x[col_sma]):
                    return share
        df["shares"] = df.apply(lambda x: weight(x), axis=1)
    
        return df

    
    def mixed(self, strategy1, strategy2):
        """buy through 2 mixed strategies of choice"""
        df = pd.merge(strategy1, strategy2, how="outer", on=["Close", "date", "shares"], indicator=True)
        return df


    def profit_loss(self, buy_df):
        """P&L, accounting for total trades & commission"""

        buy_df["profit"] = buy_df.apply(lambda x: (self.current_price-x["Close"]) * x["shares"], axis=1)        
        buy_df["cost"] = buy_df.apply(lambda x: x["shares"] * x["Close"], axis=1)
        cost = (buy_df["cost"].sum())
        total_trades = len(buy_df)
        commission_total = total_trades * self.commission
        profit = buy_df["profit"].sum() - commission_total
        
        profit_rounded = f"{int(profit):,}"
        margin = round(profit / cost * 100, 2)
        return margin, profit_rounded, total_trades
    
    
    def result(self, strategy):
        results = self.profit_loss(strategy)
        return results
    
    
    def plot(self, strategy, plottext=False, display=True, plotname=False):
        """plot buys in closing price graph

        Args:
            strategy (dataframe): output from each strategy method
            plottext (str): any text you wish to add in chart
            display (bool): display chart as popout
            plotname (str): if given, save plot in path as png"""
        
        plt.figure(figsize=(15,10))
        self.df["Close"].plot(legend=True, title=self.ticker, label="Closing Price")
        
        # add SMA plot if present
        sma_cols = [col for col in strategy.columns if "SMA" in col]
        for col_name in sma_cols:
            sma_days = int(col_name.replace("SMA", ""))
            self.df[col_name] = self.df["Close"].rolling(sma_days).mean()
            self.df[col_name].plot(label=col_name)
        
        # add buy dates
        dates = strategy["date"].tolist()
        for date in dates:
            plt.axvline(date, color="grey", lw=0.5, linestyle="--")
        
        plt.legend()
        if plottext:
            list_close = self.df["Close"].tolist()
            min_close = min(list_close)
            y = ((max(list_close) - min_close) / 2) + min_close
            x = self.df.index.tolist()[0]
            plt.text(x, y, plottext, backgroundcolor="white")
        if plotname:
            plt.savefig(f"{plotname}.png", dpi=300)
        if display:
            plt.show()


if __name__ == "__main__":
    # ticker = "QQQ"
    ticker = "SPY"
    
    start = 10
    commission = 1.99

    shares = 2
    bt = backtest(ticker, commission, shares, start)
    dca = bt.dollar_cost_averaging()
    margin, profit, trades = bt.result(dca)
    print("DCA", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(dca, plottext=text, display=False, plotname=f"results/{ticker}_DCA")
    
    sma_day = 100
    shares = 5
    bt = backtest(ticker, commission, shares, start)
    sma = bt.simple_moving_average(sma_day)
    margin, profit, trades = bt.result(sma)
    print("SMA-CLOSE", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(sma, plottext=text, display=False, plotname=f"results/{ticker}_SMA{sma_day}")

    bt = backtest(ticker, commission, shares, start)
    multi = bt.sma_multi(sma_list=[100,200,400,500,600,700,800], share_list=[2,2,2,5,5,10,10])
    margin, profit, trades = bt.result(multi)
    print("MULTI-SMA", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(multi, plottext=text, display=False, plotname=f"results/{ticker}_MULTI_SMA")
    
    sma_day1 = 30
    sma_day2 = 60
    shares = 5
    bt = backtest(ticker, commission, shares, start)
    crossing = bt.sma_crossing(sma_day1, sma_day2)
    margin, profit, trades = bt.result(crossing)
    print("SMA-CROSSING", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(crossing, plottext=text, display=False, plotname=f"results/{ticker}_SMA{sma_day1}-{sma_day2}")

    sma_day1 = 50
    sma_day2 = 100
    shares = 10
    bt = backtest(ticker, commission, shares, start)
    crossing = bt.sma_crossing(sma_day1, sma_day2)
    margin, profit, trades = bt.result(crossing)
    print("SMA-CROSSING", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(crossing, plottext=text, display=False, plotname=f"results/{ticker}_SMA{sma_day1}-{sma_day2}")
    
    mixed = bt.mixed(crossing, dca)
    margin, profit, trades = bt.result(mixed)
    print("DCA & SMA-CROSSING", margin, profit, trades)
    text = f"Profit: ${profit}\nMargin: {margin}%\nTotal Trades: {trades}"
    bt.plot(mixed, plottext=text, display=False, plotname=f"results/{ticker}_DCA-SMA{sma_day1}-{sma_day2}")



