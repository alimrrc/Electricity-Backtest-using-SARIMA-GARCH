import warnings # used to raise warnings with possible missing data
import SARIMA_GARCH_Article as sg # contains the functions of our SARIMA GARCH model
from dataclasses import dataclass # is used for the class Quote
from datetime import datetime, timedelta # is used as index and key in the dict of Quote and pd.series
from typing import Dict, List, Union, Set, Any # is used for typing PositionGenerator
from enum import Enum # is used for the class Frequency
import matplotlib.pyplot as plt # is used to diplay the details of the Backtest and/or if disp = True the autocorrelation of the quotes
import pandas as pd # is used to retrieve the quotes in a dataframe, and to use pd.Series objects
import numpy as np # is used for various computing
import yfinance as yf # is used to automatically retrieve data from Yahoo Finance


class Frequency(Enum):
    """
    Enumerated class for frequency of data sampling
    """
    HALF_HOURLY = "Half Hourly"  # New
    HOURLY = "Hourly"
    WEEKLY = "Weekly"  # New
    MONTHLY = "Monthly"
    DAILY = "Daily"


@dataclass
class Quote:
    """
    This class is a quote at a certain ts and for a certain underlying
    """
    symbol: str = None
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    adj_close: float = None
    volume: float = None
    ts: datetime = None
    
    
@dataclass
class Position:
    """
    This class is a Position, its value is 1 if long, -1 if short or 0 if neutral at a certain ts
    and for a certain underlying
    """
    value: int = None
    ts: datetime = None
    underlying: str = None


class Config:
    """
    Configuration class, contains the parameters for the model and the configuration of the Backtest
    """
    data_auto = False  # Determines if the data is retrieved from yahoo finance (True) or from a csv
    universe: str = 'ERCOT_2011_2018_yf'  # File name or yahoo finance ticker
    # Backtest from start_ts to end_ts : Don't forget to tell the hour if hourly
    start_ts: datetime = datetime(2015, 6, 1, 0)  
    end_ts: datetime = datetime(2015, 12, 31, 23)
    frequency: Frequency = Frequency.HOURLY # Frequency of the dataset
    basis: int = 100  # Base value of the portfolio in the backtest
    model_parameters: dict = {'train_start': datetime(2015, 1, 1, 0),
                              # Training date of the SARIMA-GARCH model : before start_ts
                              'min_p': 1, # Minimum p order of the ARIMA to look for
                              'min_q': 1, # Minimum q order of the ARIMA to look for
                              'max_p': 1, # Maximum p order of the ARIMA to look for
                              'max_q': 1, # Maximum q order of the ARIMA to look for
                              'seasonal_lag': 0, 
                              # Predetermine the seasonal lag, if 0 or no value, the value will be searched automatically
                              'disp': True,  # Allows to display all the details, useful to follow the progress
                              'short': True,  # Allows short position in the Backtest
                              'up_triger': 0.1,  # Triger above which a signal will become a long position
                              'low_triger': -0.1}  # Triger below which a signal will become a short position

    def __post_init__(self):
        """
        Raising Error if logical order of 'train_start', start_ts and end_ts is not respected
        """
        if self.start_ts >= self.end_ts:
            raise ValueError("self.start_ts must be before self.end_ts")
        if self.start_ts <= self.model_parameters['train_start']:
            raise ValueError("self.start_ts must be after self.model_parameters['train_start']")
        if len(self.universe) == 0:
            raise ValueError("self.universe should contain at least one element")

    @property
    def timedelta(self):
        """
        Returns the timedelta based on the chosen frequency
        """
        if self.frequency == Frequency.HALF_HOURLY:
            return timedelta(hours=0.5)
        if self.frequency == Frequency.HOURLY:
            return timedelta(hours=1)
        if self.frequency == Frequency.DAILY:
            return timedelta(days=1)
        if self.frequency == Frequency.WEEKLY:
            return timedelta(days=7)
        if self.frequency == Frequency.MONTHLY:
            return timedelta(months=1)

    @property
    def calendar(self) -> List[datetime]:
        """
        Returns the list of dates between start_ts and end_ts
        ------
        Returns
        calendar_ : List[datetime]
        """
        timedelta_ = self.timedelta
        tmp = self.start_ts
        calendar_ = []
        while tmp <= self.end_ts:
            calendar_.append(tmp)
            tmp += timedelta_
        return calendar_

    def get_quotes(self) -> Dict[Any, Quote]:
        """
        Retrieving data from yahoofinance in a dict of Quote

        Returns
        -------
        res : Dict[Any, Quote]
        """
        res = dict()

        data = pd.DataFrame()
        data = yf.download(self.universe,
                           start=self.model_parameters['train_start'],
                           end=self.end_ts)

        if len(data) == 0:
            raise ValueError("Download from yahoo finance failed")

        # Collect data from yahoofinance
        data = data.rename(
            columns={"Index": "ts",
                     "Open": "open",
                     "High": "high",
                     "Low": "low",
                     "Adj Close": "adj_close",
                     "Close": "close",
                     "Volume": "volume"})
        to_dict = data.to_dict(orient='index')

        # Convertion in Quote
        for key, value in to_dict.items():
            quote = Quote(
                self.universe,
                value['open'],
                value['high'],
                value['low'],
                value['close'],
                value['adj_close'],
                value['volume'],
                key)
            res[key] = quote

        list_dates = list(res.keys())

        # Error condition
        if list_dates[0] > self.model_parameters['train_start']:
            warnings.warn("Train set might be smaller than specified in input")
        if list_dates[-1] < self.end_ts:
            warnings.warn("Backtest set might be smaller than specified in input")
        if list_dates[-1] < self.start_ts:
            raise Warning("Backtest set has no data")
        if list_dates[0] > self.start_ts:
            raise Warning("Train set has no data")

        return res

    def get_quotes_from_excel(self) -> Dict[datetime, Quote]:
        """
        Retrieving data from a csv file in a dict of Quote

        Returns
        -------
        res : Dict[datetime, Quote]
        """
        df = pd.read_csv(f'{self.universe}.csv', parse_dates=True)

        # Similar formatting to yahoofinance
        df = df.rename(
            columns={"Index": "ts",
                     "Open": "open",
                     "High": "high",
                     "Low": "low",
                     "Adj Close": "adj_close",
                     "Close": "close",
                     "Volume": "volume"})

        res = dict()
        to_dict = df.to_dict(orient='index')

        new_dict = {}

        for key, value in to_dict.items():
            if self.frequency == Frequency.DAILY:
                new_dict[datetime.strptime(value['Date'], "%Y-%m-%d")] = to_dict[key]
            elif self.frequency == Frequency.HOURLY:
                new_dict[datetime.strptime(value['Date'], "%Y-%m-%d %H:%M")] = to_dict[key]
        # Convertion in Quote
        for key, value in new_dict.items():
            if key >= self.model_parameters['train_start'] and key <= self.end_ts:
                quote = Quote(
                    self.universe,
                    value['open'],
                    value['high'],
                    value['low'],
                    value['close'],
                    value['adj_close'],
                    value['volume'],
                    key)
                res[key] = quote

        list_dates = list(res.keys())

        # Error condition
        if list_dates[0] > self.model_parameters['train_start']:
            warnings.warn("Train set might be smaller than specified in input")
        if list_dates[-1] < self.end_ts:
            warnings.warn("Backtest set might be smaller than specified in input")
        if list_dates[-1] < self.start_ts:
            raise Warning("Backtest set has no data")
        if list_dates[0] > self.start_ts:
            raise Warning("Train set has no data")

        return res

    def get_index_of_ts(self, d: Dict, key: datetime) -> int:
        """
        Know the position index of a date in a dict of Quotes or any dict with datetime keys

        Parameters
        ----------
        d : Dict
            dictionary of ts
        key : datetime
            key to find

        Returns
        -------
        keys.index(k) : int
        """
        keys = list(d.keys())

        if key in keys:
            return keys.index(key)
        else:
            k = key
            while (k not in keys) and k <= self.end_ts:
                k = k + self.timedelta

            if k not in keys:
                raise ValueError("Key not found and no nearest key found")

            return keys.index(k)

    def get_training_ts(self, list_quotes: List[Quote]) -> List[datetime]:
        """
        Retrieving the list of dates from self.calendar that are available in a list of Quote

        Parameters
        ----------
        list_quotes : List[Quote]
            list of quotes

        Returns
        -------
        list_dates : List[timedelta]
        """
        list_dates = []
        for date in self.calendar:
            if date in list_quotes:
                list_dates.append(date)
        return list_dates


class PositionGenerator:
    """
    Allows to calculate the signals from a skew student SARIMA-GARCH model
    """

    def __init__(self, quotes_by: Dict[str, Union[List[Quote], Set[Quote]]]):
        
        self.quotes_by = quotes_by
        config = Config()

        quotes = {k: v.adj_close for k, v in self.quotes_by.items()}
        x = pd.Series(quotes)
        
        # Log differencing the quotes to get returns
        x = np.log(x) - np.log(x.shift(1))
        x = x.drop(x.index[0])
        
        # Retrieving only the required data in case more data than required have been loaded
        list_index = x.index[x.index > config.model_parameters['train_start']]
        x = x[list_index]

        # Retrieving the index of the start date in the returns
        start_index = config.get_index_of_ts(x.to_dict(), config.start_ts)

        # Retrieving data and calling functions of SARIMA_GARCH_Article
        order = sg.get_arima_orders(x[:start_index],
                                    config.model_parameters['max_p'],
                                    config.model_parameters['max_q'],
                                    config.model_parameters['min_p'],
                                    config.model_parameters['min_q'],
                                    disp=config.model_parameters['disp'])
        
        # Defining the orders of the Seasonal part of SARIMA using config if defined
        if config.model_parameters['seasonal_lag']>0:
            seasonal_order = config.model_parameters['seasonal_lag']
            seasonal_orders = list(order)
            seasonal_orders.append(seasonal_order)
            seasonal_orders = tuple(seasonal_orders)
            
        # Defining the orders of the Seasonal part of SARIMA if not predefined
        else:
            seasonal_orders = sg.get_seasonal_orders(x[:start_index],
                                                 config.frequency.value,
                                                 order,
                                                 disp=config.model_parameters['disp'])
            
        # Seasonal_lag differencing of exogenous variable
        if seasonal_orders[-1] > order[0]:
            x = x.diff(seasonal_orders[-1])
            x.dropna(inplace=True)
        
        # Fitting the SARIMA model on the train set
        sarima_model = sg.get_sarimax(x[:start_index],
                                      order,
                                      seasonal_orders,
                                      disp=config.model_parameters['disp'])
        
        # Display the results of the SARIMA if disp==True
        if config.model_parameters['disp'] is True:
            print(sarima_model.summary())
        
        # Forecasting the values of the test set with the fitted parameters of the train set
        sarima_forecast = sg.forecast_arima_sarima(sarima_model,
                                                   x,
                                                   seasonal_orders[-1],
                                                   start_index)
        
        # Applying a skew student GARCH on the residual of the SARIMA in the train set
        resid = sarima_model.resid
        skew_garch_model = sg.get_skew_garch(resid)
        
        # Display the results of the GARCH if disp==True
        if config.model_parameters['disp'] is True:
            print(skew_garch_model.summary())

        # Forecasting the Variance of the test set residuals of the SARIMA with the GARCH fitted parameters
        expected_variance = sg.forecast_garch(skew_garch_model, x, sarima_forecast, start_index)

        # If we got seasonality, we undifferentiate the serie to get returns instead of seasonality_lagged returns
        if seasonal_orders[-1] > order[0]:
            x = pd.Series(quotes)
            x = np.log(x) - np.log(x.shift(1))
            x = x.drop(x.index[0])
            x = x[list_index]
            forecast = sarima_forecast.to_list()
            past_value = x[start_index:-seasonal_orders[-1]].to_list()
            expected_return = [forecast[i] + past_value[i] for i in range(0, len(forecast))]
            expected_return = pd.Series(expected_return)
            expected_return.index = sarima_forecast.index
        else:
            expected_return = sarima_forecast

        # Compute the pd.Serie of conditional standard deviation forecasts
        expected_stdev = np.sqrt(expected_variance)

        # Get the signals : stadardised return forecasts
        signals = expected_return / expected_stdev
        
        # Assigning the signals to the attribute signals of PositionGenerator
        self.signals = signals

    def compute_positions(self) -> List[Position]:
        """
        This method will return a list of position for the backtest period

        Returns
        -------
        output : List[int]
            list of position for the trading period
        """
        output = []
        for indx in self.signals.index:
            if self.signals[indx] > Config.model_parameters['up_triger']:  # long
                output.append(Position(1, indx, Config.universe))
            elif self.signals[indx] < Config.model_parameters['low_triger']:  # short
                output.append(Position(-1, indx, Config.universe))
            else:
                output.append(Position(0, indx, Config.universe))

        return output


class Backtester:
    """
    Class used for backtesting the strategy, calling parameters in config and retrieving positiongenerator
    """

    def __init__(self, config: Config):

        self._config = config  # My config
        self._basis = self._config.basis
        self._calendar = self._config.calendar  # List of date between start_ts and end_ts
        self._universe = self._config.universe  # List of tickers
        self._timedelta = self._config.timedelta  # frequency

        if self._config.data_auto is True:
            self._quote_by_pk = self._config.get_quotes()  # Dictionary of quotes
        else:
            self._quote_by_pk = self._config.get_quotes_from_excel()

        position_generator = PositionGenerator(self._quote_by_pk)  # Compute positions

        # Calculer les poids en fn des signaux
        self._weight_by_pk = position_generator.compute_positions()  # Dictionary of weights !
        # Il faut ajouter les params en argument venant de Config
        self._level_by_ts = dict()  # Dictionary of portfolio level
        self._cash = dict()  # Cash of ptf
        self._position_value = dict()  # Position of ptf, + if long

    def run_backtest(self):
        """
        This method computes the value, cash and position of the backest portfolio at
        each step of the backtesting period and assigns a dict to self._levels_by_ts, 
        self._cash and self._position_value

        For each step, my signal is the forecast from the information of the previous step,
        meaning that my signal at t was computed at t-1.
        
        If my signal went up at t, I buy at the price of the open, then the value of my 
        position is updated with the closing price, and the cash computed from the amount
        bought at the open price.
        
        If my signal went down at t, I sell at the price of the open, then the value of my 
        position is updated with the closing price, and the cash computed from the amount
        sold at the open price.

        At t0 no trade is possible
        """
        short: bool = self._config.model_parameters['short']  # short position allowed if true
        signals = [position.value for position in self._weight_by_pk]
        
        list_ts = self._config.get_training_ts(self._quote_by_pk) # list of ts of the backtest period
        
        backtest_quotes = [self._quote_by_pk[k] for k in list_ts] # quotes of the backtest period
        opens = [v.open for v in backtest_quotes] # opens of the backtest
        closes = [v.adj_close for v in backtest_quotes]  # closes of the backtest
        value = [self._basis] # value of the portfolio at the begining
        cash = [self._basis] # cash at beginning
        position = [0]  # no position at the start 
        
        first_signal = len(closes) - len(signals)  # mismatch between the number of quotes and the
        # signals when there is seasonality, we start trading when there are signals

        if short is False:
            signals = list(map(lambda x: x if x > 0 else 0, signals))  # returns to 0 all that is -1 if short

        signals[0] = 0  # no purchase in the first period, initialization

        # Compare the signal in t to t-1 so buy in t: need 1 value before buying/selling

        for i in range(1, len(closes)):

            if i > first_signal:

                if signals[i - first_signal] == signals[i - 1 - first_signal]:
                    # if signal in t = signal in t-1 : 
                    # we do not move, just update, cash does not change
                    position.append(position[i - 1] / closes[i - 1] * closes[i])
                    cash.append(cash[i - 1])

                elif signals[i - first_signal] > signals[i - 1 - first_signal]:
                    # if I buy, signal in t > signal in t-1 : 
                    # position will change, can buy up to cash times signal in t - signal in t-1
                    # we buy at open and position updates with close
                    position.append((cash[i - 1] + position[i - 1] * (opens[i] / closes[i - 1])) * (signals[i - first_signal]) * closes[i] / opens[i])
                    cash.append(cash[i - 1] + position[i - 1] * opens[i] / closes[i - 1] - position[i] * opens[i] / closes[i])

                elif signals[i - first_signal] < signals[i - 1 - first_signal]:
                    # if I sell, signal in t < signal in t-1 :
                    # position will change, can sell down to cash times signal in t - signal in t-1
                    # we sell at open and position updates with close
                    cash.append(cash[i - 1] + (cash[i - 1] + position[i - 1] * (opens[i] / closes[i - 1])) * (signals[i - 1 - first_signal] - signals[i - first_signal]))
                    position.append((cash[i - 1] + position[i - 1] * opens[i] / closes[i - 1] - cash[i]) * closes[i] / opens[i])

                value.append(position[i] + cash[i])

            else:
                # initial value
                position.append(position[i - 1])
                cash.append(cash[i - 1])
                value.append(value[i - 1])

        # Series of what we have calculated, easier to convert to dictation
        s_value = pd.Series(value)
        s_value.index = list_ts
        s_cash = pd.Series(cash)
        s_cash.index = list_ts
        s_position = pd.Series(position)
        s_position.index = list_ts

        self._level_by_ts = s_value.to_dict()
        self._cash = s_cash.to_dict()
        self._position_value = s_position.to_dict()

    def plot_perf(self):
        """
        Method to display the value of the portfolio
        """
        if self._level_by_ts == dict():
            self.run_backtest()

        plt.plot(pd.Series(self._level_by_ts), label=self._level_by_ts.keys())
        plt.show()

    def plot_detailed_perf(self):
        """
        Method for displaying the value of the portfolio, the position and the cash
        """
        if self._level_by_ts == dict():
            self.run_backtest()

        fig, axs = plt.subplots(3, figsize=(20, 15))
        axs[0].plot(pd.Series(self._level_by_ts), label=self._level_by_ts.keys())
        axs[0].set_title('Portfolio value')
        axs[1].plot(pd.Series(self._cash), label=self._level_by_ts.keys())
        axs[1].set_title('Cash value')
        axs[2].plot(pd.Series(self._position_value), label=self._level_by_ts.keys())
        axs[2].set_title('Position value')


my_backtest = Backtester(Config())
# Choose here the desired plot
my_backtest.plot_detailed_perf()
