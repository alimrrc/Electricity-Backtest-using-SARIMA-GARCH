import warnings # used to disable warnings
import pandas as pd # use pd.Series objects
import numpy as np # is used for various computing
import matplotlib.pyplot as plt # if disp = True the autocorrelation of the quotes
from arch import arch_model # fitting GARCH model
from statsmodels.tsa.statespace.sarimax import SARIMAX # fitting ARIMA and/or SARIMA
from statsmodels.tsa.stattools import adfuller # stationnarity testing
from statsmodels.stats.diagnostic import het_arch # heteroscedasticity testing
from statsmodels.tsa.stattools import acf # is used to compute the autocorrelation to identify seasonality

warnings.filterwarnings('ignore')

def optimize_arima(order_list: list, exog: pd.Series(), disp: bool = False) -> tuple:
    """
    Calculation of AIC of an optimal ARIMA for each combination of (p,d,q) order tuples
    Return dataframe of 1 row with (p,d,q) orders and corresponding AIC

    Parameters
    ----------
    order_list :
        list with (p, d, q) tuples
    exog : pd.Series()
        the exogenous variable
    disp : bool
        to display charts and print infos

    Returns
    -------
    result_df.iloc[0, 0] : tuple
    """

    results = []

    for order in order_list:
        if disp:
            print(f'reaching order {order}...')
        try:
            model = SARIMAX(exog, order=order).fit(disp=False)
        except:
            continue
        if disp:
            print(f'reached order {order}, AIC = {model.aic}')
        results.append([order, model.aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df.iloc[0, 0]

def get_arima_orders(data: pd.Series(), max_p: int, max_q: int, min_p: int = 0, min_q: int = 0,
                     disp: bool = False) -> tuple:
    """
    Returns the optimal tuple (p,d,q) for a time series to apply an ARIMA model

    Parameters
    ----------
    data : pd.Series()
    max_p : int
        max p of arima
    max_q : int
        max q of arima
    min_p : int
        min p of arima
    min_q : int
        min q of arima
    disp : bool
        to display charts and print infos

    Returns
    -------
    best_arima : tuple
    """
    d = 0

    if disp:
        # Augmented Dickey-Fuller test
        ad_fuller_result = adfuller(data)
        # ARCH heteroscedasticity test
        arch_test_result = het_arch(data)
        print(f'ADF Statistic: {ad_fuller_result[0]}')
        print(f'p-value: {ad_fuller_result[1]}')
        print(f'ARCH Statistic: {arch_test_result[2]}')
        print(f'p-value: {arch_test_result[3]}')
        print(f'intergration factor: {d}')

    ps = range(min_p, max_p + 1, 1)
    qs = range(min_q, max_q + 1, 1)

    # Create a list with all possible combination of (p,q) parameters
    parameters = list()
    for x in ps:
        for y in qs:
            parameters.append((x, y))
    
    # We add d=0 in each parameter combination so we get (p,d,q)
    order_list = []
    for each in parameters:
        each = list(each)
        each.insert(1, d)
        each = tuple(each)
        order_list.append(each)

    best_arima = optimize_arima(order_list, exog=data, disp=disp)

    if disp:
        print(f'chosen ARIMA orders {best_arima}')

    return best_arima

def get_seasonality(x: pd.Series(), frequency: str, disp: bool = False) -> int:
    """
    Returns the optimal lag of seasonality of x from its observed waves in the autocorrelation function

    Parameters
    ----------
    x : pd.Series()
        data
    frequency : str
        frequency
    disp : bool
        to display charts and print infos

    Returns
    -------
    lag of seasonality
    """
    if disp:
        print("Finding the seasonnal lag of the exogenous variable")

    max_j = 3  # max_j = 3 : the number of homogenous waves that we have to observe
               #             until we can affirm that x has got seasonality
    acf_len = 0

    # we set the length of the autocorrelation list depending on the frequency of the data
    if frequency == "Half Hourly":
        acf_len = 48 * max_j + 1
    if frequency == "Hourly":
        acf_len = 24 * max_j + 1
    if frequency == "Daily":
        acf_len = 255 * max_j + 1
    if frequency == "Weekly":
        acf_len = 52 * max_j + 1
    if frequency == "Monthly":
        acf_len = 12 * max_j + 1

    if acf_len > 0:
        
        # Create the autocorrelation function
        my_acf = acf(x, nlags=acf_len, qstat=True)[0]
        
        # Show the autocorrelation function if disp = True
        if disp:
            plt.plot(my_acf)
            plt.show()
            
            
        is_seasonal = True
        i: int = 0
        y = my_acf[0]
        z = my_acf[1]
        local_max = i
        my_max = []

        # Loop through the list of autocorrelations to catch all the local miximums
        while i < len(my_acf) - 2:

            # as long as it goes down go to next lag
            while z <= y and i < len(my_acf) - 2:
                i = i + 1
                y = z
                z = my_acf[i + 1]

            # as long as it goes up go to next lag
            while z >= y and i < len(my_acf) - 2:
                i = i + 1
                y = z
                z = my_acf[i + 1]
                local_max = i

            # lag where we get the local max, corresponding to the coordinates of the wave maximum
            my_max.append(local_max)
        
        # If the abcisse of the local maximums of our waves aren't consistently separated then it is not seasonal
        for j in range(0, max_j - 1):
            if my_max[j] != my_max[j + 1] - my_max[0]:
                is_seasonal = False

        seasonal_lag = my_max[0]

        # If we think that the serie is seasonal, we find the highest local maximum abcisse
        if is_seasonal:
            local_max = my_acf[my_max[0]]
            for i in range(0, len(my_max)):
                lag = my_max[i]
                if local_max < my_acf[lag]:
                    local_max = my_acf[my_max[i]]
                    seasonal_lag = my_max[i]

        # if the chosen maxium of the autocorrelation is significantly above 0 then we chose its abcisse for our seasonal lag
        if is_seasonal is True and my_acf[seasonal_lag] > 0.2:  # sort on seasonality > 0.2
            return seasonal_lag
        else:
            return 0
    else:
        return 0


def get_seasonal_orders(x: pd.Series(), frequency: str, order: tuple, disp: bool = False) -> tuple:
    """
    Returns the optimal orders of the SARIMA model for x

    Parameters
    ----------
    x : pd.Series()
        data
    frequency : str
    order : tuple
        previous order
        order : p,d,q
        order[0] = p
    disp : bool
        to display charts and print infos

    Returns
    -------
    seasonal_orders : tuple
    """
    seasonal_order = get_seasonality(x, frequency, disp=disp)
    seasonal_orders = list(order)
    seasonal_orders.append(seasonal_order)
    if disp is True and seasonal_order > 0:
        print(f'Chosen SARIMA orders {seasonal_orders}')
    if disp is True and seasonal_order == 0:
        print('No seasonality found')
    return tuple(seasonal_orders)

def get_sarimax(x: pd.Series(), order: tuple, seasonal_orders: tuple, disp: bool = False):
    """
    Returns a fitted SARIMA or ARIMA model

    Parameters
    ----------
    x : pd.Series()
        data
    order : tuple
        ARIMA order
    seasonal_orders : tuple
        SARIMA orders
    disp : bool
        to display charts and print infos

    Returns
    -------
    sarimax
    """
    if seasonal_orders[-1] > order[0]:
        if disp:
            print(f'Fitting a SARIMA {order} {seasonal_orders}')
        return SARIMAX(x, order=order, seasonal_order=seasonal_orders).fit(disp=False, low_memory=True)
    else:
        if disp:
            print(f'Fitting an ARIMA {order}')
        return SARIMAX(x, order=order).fit(disp=False)


def get_skew_garch(resid: pd.Series()):
    """
    Returns SKEW Student GARCH fitted model over SARIMA or ARIMA resids

    Parameters
    ----------
    resid : pd.Series()
        residual

    Returns
    -------
    skewed_garch : ARCHModelResult
    """
    skewed_garch = arch_model(resid, mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='skewstudent')
    return skewed_garch.fit(disp=False)

def predict_arima(time_series: list[float], const_param: int, ar_params: list[float], ma_params: list[float]) -> list[float]:
    """
    Calculates the forecasts over the backtest period for ARIMA
    Important : 
        # The forecast in time t is computed thanks to the information in t-1
        # The values of the parameters do not change when filtering on the test set
        
    Parameters
    ----------
    time_series : list[float]
    const_param : bool
    ar_params : list[float]
    ma_params : list[float]

    Returns
    -------
    predictions : list[float]
    """
    # Convert the input lists in numpy arrays
    time_series = np.array(time_series)  # list
    ar_params = np.array(ar_params)  # list
    ma_params = np.array(ma_params)  # list

    # Compute the number of time series values and the number of AR and MA parameters
    n_values = len(time_series)
    n_ar_params = len(ar_params)
    n_ma_params = len(ma_params)

    # Initialize the list of predictions with the first value of the time_series
    predictions = [time_series[0]]

    # Loop through the time series values and compute the predictions
    for t in range(1, n_values):
        
        # Compute the AR component of the prediction
        ar_prediction = 0
        min_t_ar_params = min(t, n_ar_params)
        for i in range(0, min_t_ar_params):
            if t - i - 1 >= 0:
                ar_prediction += ar_params[i] * time_series[t - i - 1]

        # Compute the MA component of the prediction
        ma_prediction = 0
        min_t_ma_params = min(t, n_ma_params)
        for i in range(0, min_t_ma_params):
            if t - i - 1 >= 0:
                ma_prediction += ma_params[i] * (time_series[t - i - 1] - predictions[t - i - 1])

        # Compute the overall prediction by summing the AR and MA components
        prediction = ar_prediction + ma_prediction + const_param
        predictions.append(prediction)

    return predictions

def predict_sarima(time_series: list[float], const_param: int, ar_params: list[float], ma_params: list[float],
                   sar_params: list[float], sma_params: list[float], seasonal_order: int) -> list[float]:
    """
    Calculates the forecasts over the backtest period for SARIMA
    Important : 
        # The forecast in time t is computed thanks to the information in t-1
        # The values of the parameters do not change when filtering on the test set
        
    Parameters
    ----------
    time_series : list[float]
    const_param : bool
    ar_params : list[float]
    ma_params : list[float]
    sar_params : list[float]
    sma_params : list[float]
    seasonal_order : int

    Returns
    -------
    predictions : list[float]
    """
    # Ensure all parameter lists are numpy arrays
    ar_params = np.array(ar_params)
    ma_params = np.array(ma_params)
    sar_params = np.array(sar_params)
    sma_params = np.array(sma_params)

    # Get the number of AR and MA parameters
    num_ar_params = len(ar_params)
    num_ma_params = len(ma_params)
    num_sar_params = len(sar_params)
    num_sma_params = len(sma_params)

    # Initialize a list to store the predictions
    predictions = [time_series[0]]

    # Iterate through the time series and make a prediction for each point
    for i in range(1, len(time_series)):
        
        # Initialize the prediction to be the mean of the time series
        prediction = const_param

        # Calculate the contribution of the AR terms
        for j in range(num_ar_params):
            if i - j - 1 >= 0:
                prediction += ar_params[j] * time_series[i - j - 1]

        # Calculate the contribution of the MA terms
        for j in range(0, min(i, num_ma_params)):
            if i - j - 1 >= 0 and j + 1 <= len(predictions):
                prediction += ma_params[j] * (time_series[i - j - 1] - predictions[i - j - 1])

        # Calculate the contribution of the SAR terms
        for j in range(num_sar_params):
            if i - j * seasonal_order - 2 >= 0:
                prediction += sar_params[j] * time_series[i - j * seasonal_order - 1] - sar_params[j] * ar_params[j] * \
                              time_series[i - j * seasonal_order - 2]

        # Calculate the contribution of the SMA terms
        for j in range(0, min(i, num_sma_params)):
            if i - j * seasonal_order - 2 >= 0 and j * seasonal_order + 2 <= len(predictions):
                prediction += sma_params[j] * (
                        time_series[i - j * seasonal_order - 1] - predictions[i - j * seasonal_order - 1]) + \
                              sma_params[j] * ma_params[j] * (time_series[i - j * seasonal_order - 2] - predictions[
                                i - j * seasonal_order - 2])

        # Add the prediction to the list
        predictions.append(prediction)

    return predictions

def forecast_arima_sarima(sarima_model, time_series: pd.Series(), seasonal_lag: int, start_index: int) -> pd.Series():
    """
    Returns forecast serie of already fitted arima or sarima on the backtest set
    Important : 
        # The forecast in time t is computed thanks to the information in t-1
        # The values of the parameters do not change when filtering on the test set

    Parameters
    ----------
    sarima_model : SARIMAXResult
    time_series : pd.Series()
    seasonal_lag : int
    start_index : int

    Returns
    -------
    predict[start_index:] : pd.Series()
    """
    
    my_index = time_series.index
    time_series = time_series.to_list()
    
    # Compute the maximum lenght of each order (p,d,q)(P,D,Q)
    params_list_len = max(
        [len(sarima_model._params_ar), len(sarima_model._params_ma), len(sarima_model._params_seasonal_ar),
         len(sarima_model._params_seasonal_ma)])

    ar_s = []
    ma_s = []
    sar_s = []
    sma_s = []

    # Store the value of each fitted parameter of sarima_model in a list ar_s, ma_s, sar_s of sma_s
    for i in range(0, params_list_len):
        if i < len(sarima_model._params_ar):
            ar_s.append(sarima_model._params_ar[i])
        else:
            ar_s.append(0)
        if i < len(sarima_model._params_ma):
            ma_s.append(sarima_model._params_ma[i])
        else:
            ma_s.append(0)
        if i < len(sarima_model._params_seasonal_ar):
            sar_s.append(sarima_model._params_seasonal_ar[i])
        else:
            sar_s.append(0)
        if i < len(sarima_model._params_seasonal_ma):
            sma_s.append(sarima_model._params_seasonal_ma[i])
        else:
            sma_s.append(0)

    # if there is so seasonality, we forecast using ARIMA, else using SARIMA
    if seasonal_lag == 0:
        predict = predict_arima(time_series, 0, ar_s, ma_s)
    else:
        predict = predict_sarima(time_series, 0, ar_s, ma_s, sar_s, sma_s, seasonal_lag)

    predict = pd.Series(predict)
    predict.index = my_index

    return predict[start_index:]

def forecast_garch(garch_model, time_series: pd.Series(), sarima_forecast, start_index: int) -> pd.Series():
    """
    Returns the garch model, forecasts it under the SARIMA residues and returns the garch forecast

    Parameters
    ----------
    garch_model : ARCHModelResult
    time_series : pd.Series()
    sarima_forecast : 
    start_index : int

    Returns
    -------
    expected_variance : pd.Series()
    """
    
    # Get the fitted parameters of the GARCH model
    #omega = garch_model.params[0]
    alpha = garch_model.params[1]
    beta = garch_model.params[2]
    
    # Initialize the first variance and the list of variances
    variance = garch_model.conditional_volatility[-1]
    list_variance = [np.var(time_series[start_index:] - sarima_forecast)]

    # Compute the forecast of the conditional variance at each step
    for i in range(1, len(sarima_forecast)):
        resid_i_1 = time_series[start_index + i - 1] - sarima_forecast[i - 1]
        resid_i_1 = resid_i_1 * resid_i_1
        variance =  alpha * resid_i_1 + beta * variance
        list_variance.append(variance)

    # Convert the forecasts to pd.Series
    expected_variance: pd.Series = pd.Series(list_variance)
    expected_variance.index = time_series[start_index:].index

    return expected_variance
