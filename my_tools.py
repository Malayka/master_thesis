import blocksci
import matplotlib.pyplot as plt
import matplotlib.ticker
import collections
import pandas as pd
import numpy as np
import datetime



#Q_ANALYSIS

S_REL_FLOWS = 'relativeFlowVolumes'

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox



def testStationarityTwice(series):
    print('prob if stationary:', statsmodels.tsa.stattools.kpss(series)[1])
    print('prob if NOT stationary:', adfuller(series)[1])

def checkStationarities(df):
    for col in df.columns:
        if col != 'times':
            print(col)
            testStationarityTwice(df[col])

def grangerCasualityNp(y, x, maxlag):
    xy = pd.DataFrame({'x': x, 'y': y})
    r = statsmodels.tsa.stattools.grangercausalitytests(xy[['y', 'x']], maxlag=maxlag, verbose=False)
    result_np = np.zeros((maxlag + 1, 5))
    for lag in range(1, maxlag + 1):
        lag_r = r[lag]
        p_values = [test_results[1] for test_results in lag_r[0].values()]
        #print(list(lag_r[0].values()))
        #p_values = [np.round(np.float(test_results[1]),5) for test_results in lag_r[0].values()]
        result_np[lag][1:] = np.array(p_values).round(3)
        result_np[lag][0] = (np.array(p_values).round(3)).max()
    return result_np[1:]

def todaYamamoto(y, x, maxlags=None, verbose=0):
    xy = pd.DataFrame({'x': x, 'y': y})
    model = VAR(xy)
    if maxlags is None:
        maxlags = model.select_order().selected_orders['aic']
        if maxlags == 0:
            maxlags = 1
        #print('maxlags:', maxlags)
        if verbose >= 3:
            print('maxlags is', maxlags)
    model_fit = model.fit(maxlags=maxlags)
    lag = model_fit.k_ar
    if verbose >= 2:
        print('lag is', lag)
    
    if verbose >= 1:
        print('x resids acorr p:', acorr_ljungbox(model_fit.resid['x'], lag)[1].max())
        print('y resids acorr', acorr_ljungbox(model_fit.resid['y'], lag)[1].max())
    
    #print('lag:', lag)
    if lag == 0:
        print('lag was 0! Columns:', y.name, x.name)
        lag = 1
    x_not_cause_y = grangerCasualityNp(y, x, lag)[lag - 1][0]
    y_not_cause_x = grangerCasualityNp(x, y, lag)[lag - 1][0]
    if verbose >= 1:
        print('x does NOT cause y:', x_not_cause_y)
        print('y does NOT cause x:', y_not_cause_x)
    return x_not_cause_y, y_not_cause_x

def todaYamamotoWithCol(df, the_col, verbose=0):
    for a_col in df.columns:
        if not a_col in (the_col, 'times'):
            print('y={},   x={}'.format(a_col, the_col))
            todaYamamoto(df[a_col], df[the_col], verbose=verbose)
            print('-------------------------------')

def todaYamamotoDf(df, verbose=0):
    ty_df = df.corr().copy()
    for col in ty_df.columns:
        for ind in ty_df.index:
            if ind == col:
                ty_df[col][ind] = None
                continue
            if verbose >= 1:
                print('y={},   x={}'.format(col, ind))
            ty_df[col][ind], ty_df[ind][col] = todaYamamoto(df[col], df[ind], verbose=verbose)
            if verbose >= 1:
                print('-------------------------------')
    return ty_df

def testStationarity(timeseries, plot=True):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    if plot:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def getMovingAvgDiff(series, lag=12):
    moving_avg = pd.rolling_mean(series, lag)
    return (series - moving_avg).dropna()

def getMovingEwmaDiff(series, lag=12):
    moving_ewma = pd.ewma(series, halflife=lag)
    return (series - moving_ewma).dropna()

def getDiff(series, lag=1):
    return (series - series.shift(lag)).dropna()


def drawAcfPacf(series):
    #ACF and PACF plots:
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(series, nlags=20)
    lag_pacf = pacf(series, nlags=20, method='ols')

    f = plt.figure(figsize=(10, 4))
    axes = f.subplots(1, 2, sharex=False)
    axes[0].plot(lag_pacf)
    axes[0].axhline(y=0,linestyle='--',color='gray')
    axes[0].axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    axes[0].axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    axes[0].set_title('Partial Autocorrelation Function')

    axes[1].plot(lag_acf)
    axes[1].axhline(y=0,linestyle='--',color='gray')
    axes[1].axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    axes[1].axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    axes[1].set_title('Autocorrelation Function')
    plt.pause(0.05)

def standartizeDiffs(series):
    diffs = getDiff(series)
    standartized_diffs = (diffs - diffs.mean()) / diffs.std()
    
    std_diffs_with_first = np.empty(len(series))
    std_diffs_with_first[0] = series.values[0]
    std_diffs_with_first[1:] = standartized_diffs.values

    result = pd.Series(std_diffs_with_first).cumsum()
    result.index = series.index
    return result, diffs.mean(), diffs.std()
    
def restoreDiffs(series, mean, std):
    diffs = getDiff(series)
    restored_diffs = diffs * std + mean
    
    std_diffs_with_first = np.empty(len(series))
    std_diffs_with_first[0] = series.values[0]
    std_diffs_with_first[1:] = restored_diffs.values

    result = pd.Series(std_diffs_with_first).cumsum()
    result.index = series.index
    return result

def standartizeDiffsDf(df):
    result = pd.DataFrame()
    for col in df.columns:
        result[col], _,_ = standartizeDiffs(df[col])
    return result


def prepareToArimax(df, endog_col=S_REL_FLOWS):
    df_shifted = pd.DataFrame()
    for col in df.columns:
        if col != endog_col:
            df_shifted[col]  = df[col][:-1]
        else:
            df_shifted[col]  = df[col].shift(-1).dropna()
    return df_shifted.dropna()

def createTag(names_list):
    return '_'.join(sorted(names_list))

def arimaPredictInSampleRMSE(X, arima_order, plot=True):
    model = ARIMA(X, order=arima_order)  
    model_fit = model.fit(disp=-1) 
    model_predict = model_fit.predict(typ='levels')
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(X)
        plt.plot(model_predict, color='red')
    rmse = np.sqrt(mean_squared_error(X[-len(model_predict):], model_predict))
    print('RMSE: %.4f'% rmse)
    return rmse

def getMetricsCombinationsScores(df, metrics_list, arima_order):
    rmses = {}
    for r in range(1, len(metrics_list) + 1):
        for comb in itertools.combinations(metrics_list, r):
            columns = list(comb) + [S_REL_FLOWS]
            try:
                res = ARIMAX_result(df.dropna()[columns], S_REL_FLOWS, arima_order=arima_order, plot=False)
            except Exception as e:
                print(createTag(comb), 'is failed')
                print(e)
            else:
                rmses[createTag(comb)] = res
    return rmses

def getMetricsAndPCombinationsScores(df, metrics_list, max_p=3):
    rmses_p = {}
    for p in range(max_p + 1):
        rmses = getMetricsCombinationsScores(df, metrics_list, (p, 1, 1))
        best_comb = [k for (k,v) in rmses.items() if v <= min(rmses.values())][0]
        rmses_p[str(p)] = (best_comb, rmses[best_comb])
        print('p = {} is done'.format(p))
    return rmses_p

def ARIMAX_PredictInSampleRMSE(df, endog_col, arima_order, plot=True):
    X = pd.DataFrame()
    for col in df.columns:
        if col != endog_col:
            X = df[col]

    y = df[endog_col]
    model = sm.tsa.ARIMA(endog=y, exog=X, order=arima_order)
    model_fit = model.fit(disp=-1) 
    model_predict = model_fit.predict(exog=X, typ='levels')
    print(len(X))
    print(len(model_predict))
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(y)
        plt.plot(model_predict, color='red')
    rmse = np.sqrt(mean_squared_error(y[-len(model_predict):], model_predict))
    print('RMSE: %.4f'% rmse)
    return rmse

def arimaWalkForwardValidation(X, size, arima_order, plot=True):
    train, test = X[0:size], X[size:]
    history = [x for x in train]
    predictions = list()
    confs_1 = list()
    confs_2 = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=False)
        #yhat = model_fit.forecast()[0]
        yhat, _, conf_int = model_fit.forecast(alpha=0.1)
        predictions.append(yhat)
        confs_1.append(conf_int[0][0])
        confs_2.append(conf_int[0][1])
        obs = test[t]
        history.append(obs)
        #print('>predicted=%.3f, expected=%.3f' % (float(yhat), float(obs)))

    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.fill_between(x=range(len(test)), y1=confs_1, y2=confs_2, color='red', alpha=0.3)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.4f' % rmse)
    return rmse

def ARIMAX_WalkForwardValidation(df, endog_col, size, arima_order, plot=True):
    #X = pd.DataFrame()
    #X.index = df.index
    #for col in df.columns:
    #    if col != endog_col:
    #        X = df[col]
    X = df.copy()
    del X[endog_col]

    y = df[endog_col]
    X_train, X_test, y_train, y_test = X[0:size], X[size:], y[0:size], y[size:]
    X_history = X_train
    y_history = y_train
    predictions = list()
    conf_1 = list()
    conf_2 = list()
    for t in range(len(y_test)):
        model = sm.tsa.ARIMA(endog=y_history.values, exog=X_history.values, order=arima_order)
        try:
            model_fit = model.fit(disp=False)
            yhat, _, conf_int  = model_fit.forecast(exog=X_history[-1:].values, alpha=0.1)
        except Exception as e:
            print("step", t)
            print(e)
            yhat, conf_int = np.nan, [(np.nan, np.nan)]
        predictions.append(yhat)
        conf_1.append(conf_int[0][0])
        conf_2.append(conf_int[0][1])

        X_history = X_history.append(X_test[t:t+1])
        y_history = y_history.append(y_test[t:t+1])
        #print('>predicted=%.3f, expected=%.3f' % (float(yhat), float(obs)))

    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(y_test.values)
        plt.plot(predictions, color='red')
        plt.fill_between(x=range(len(y_test.values)), y1=conf_1, y2=conf_2, color='red', alpha=0.3)
        #plt.plot(conf_1, color='green')
        #plt.plot(conf_2, color='green')
    #rmse = np.sqrt(mean_squared_error(y_test, predictions))
    #print('Test RMSE: %.4f' % rmse)
    #return predictions, (conf_1, conf_2)
    return {'predictions': predictions, 'conf_1': conf_1, 'conf_2': conf_2, 'index': y_test.index}

def ARIMAX_result(df, endog_col, arima_order, plot=False, summary=False):
    X = df.copy()
    del X[endog_col]
    y = df[endog_col]

    model = sm.tsa.ARIMA(endog=y.values, exog=X.values, order=arima_order)
    model_fit = model.fit(disp=-1) 

    if summary:
        print(model_fit.summary())
    model_predict = model_fit.predict(exog=X.values, typ='levels')
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(y.values)
        plt.plot(model_predict, color='red')
    rmse = np.sqrt(mean_squared_error(y[-len(model_predict):], model_predict))
    #print('RMSE: %.4f'% rmse)
    return rmse

def ARIMAX_result_cheat(df, endog_col, arima_order, plot=False, summary=False, ):
    X = df.copy()
    del X[endog_col]
    y = df[endog_col]

    model = sm.tsa.ARIMA(endog=y.values, exog=X.values, order=arima_order)
    model_fit = model.fit(disp=-1) 

    if summary:
        print(model_fit.summary())
    model_predict = model_fit.predict(exog=X.values, typ='levels')
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(y.values)
        plt.plot(model_predict, color='red')
    rmse = np.sqrt(mean_squared_error(y[-len(model_predict):], model_predict))
    #print('RMSE: %.4f'% rmse)
    return rmse


def ewmaPreditcionRMSE(series, begin, halflife=12, plot=True):
    ewma_prediction = pd.ewma(series.shift(), halflife=halflife)
    ewma_errors = (series - ewma_prediction).dropna()[begin:]
    if plot:
        plt.figure(figsize=(20, 5))
        plt.plot(series[begin:])
        plt.plot(ewma_prediction[begin:], color='red')

    rmse = np.sqrt(sum((ewma_errors)**2)/len(ewma_errors))
    print('RMSE: %.4f'% rmse)
    return rmse

def runArimaWFAndEwmaPredictionsRMSE(df, endog_col, size, arima_order, plot=True, ewma_halflife=1):
    ARIMAX_WalkForwardValidation(df=df, endog_col=endog_col, size=size, arima_order=arima_order, plot=plot)
    arimaWalkForwardValidation(X=df[endog_col].values, size=size, arima_order=arima_order, plot=plot)
    ewmaPreditcionRMSE(series=df[endog_col], begin=size, halflife=ewma_halflife, plot=plot)




def getTimes(blocks, interval):
    times = []
    for i in range(0, len(blocks), interval):
        times.append(blocks[min(i + interval -1, len(blocks) - 1)].time)
    return times


# Q_NAKAMOTO
UNKNOWN = 'Unknown'
def guessMinerAdds3(blocks):
    first_block_h = blocks[0].height
    result = [UNKNOWN,] * len(blocks)
    
    coinbase_outs = [b.coinbase_tx.outputs for b in blocks]
    good_outs = [CO[0] for CO in coinbase_outs if len(CO) == 1]
    for o in good_outs:
        result[o.block.height - first_block_h] = o.address
    
    one_out_CO_adds = set([o.address for o in good_outs])
    mult_out_COs = [CO for CO in coinbase_outs if len(CO) > 1]
    
    for CO in mult_out_COs:
        out_with_main_prize = [out for out in CO if out.value > out.block.revenue * 0.6]
        if len(out_with_main_prize) > 0:
            candidate = out_with_main_prize[0]
            
            result[candidate.block.height - first_block_h] = candidate.address
    return result


def calcNakamotoCoef2(miners, interval):
    egoists = []
    result = []
    for ind in range(0, len(miners), interval):
        cur_miners = miners[ind: ind + interval]
        cur_miners_cnt = collections.Counter(cur_miners)
        must_be_beaten = float(len(cur_miners)) * 0.51
        
        # Too much Unknown
        if len(cur_miners) - cur_miners_cnt[UNKNOWN] <= must_be_beaten:
            result.append(-1)
        else:
            what_we_got = 0.
            i = 1
            for miner, freq in cur_miners_cnt.most_common():
                if miner == UNKNOWN:
                    continue
                what_we_got += float(freq)
                if what_we_got > must_be_beaten:
                    result.append(i)
                    if i == 1:
                        egoists.append((ind, cur_miners_cnt.most_common(1)))
                    break
                i += 1
    return result, egoists


# Q_ADDRESSES
def getAddresses(block):
    addresses = set()
    for in_ in block.inputs:
        addresses.add(in_.address)
    for out in block.outputs:
        addresses.add(out.address)
    return addresses

def getAddressesOfBlocks(blocks):
    addresses = set()
    for b in blocks:
        addresses |= getAddresses(b)
    return addresses

def getActiveAddressesCounts(blocks, interval):
    adds_counts = []
    for i in range(0, len(blocks), interval):
        subchain = blocks[i: i + interval]
        adds_counts.append(len(getAddressesOfBlocks(subchain)))
    return adds_counts


# Q_MAP
def getMapAddress2Cluster2Np(cluster_manager, all_addresses_dict, buffer_size=100000):
    map_address2cluster = {}
    for address_type in all_addresses_dict:
        buffers_list = []
        buffer = np.zeros(buffer_size, dtype=np.int32)
        ind = 1
        for address in all_addresses_dict[address_type]:
            cluster = cluster_manager.cluster_with_address(address)
            #if address.address_num != ind:
            #    print('Huynya!', ind, address.address_num)
            buffer[ind % buffer_size] = cluster.index
            ind += 1
            if ind % buffer_size == 0:
                buffers_list.append(buffer)
                buffer = np.zeros(buffer_size, dtype=np.int32)
        
        buffers_list.append(buffer)
        map_array = np.array(buffers_list).flatten()
        map_array[0] = ind - 1
        map_address2cluster[address_type] = map_array
        #if ind % 10000000 == 0:
        #    print("  {}m  addresses are proceed".format(ind / 1000000))
        #ind += 1
    return map_address2cluster


# Q_CLUSTERS
def getClusterIdsOfBlocksNp(blocks, map_address2cluster_dict):
    cluster_ids = set()
    active_addresses = getAddressesOfBlocks(blocks)
    for address in active_addresses:
        if address.type != blocksci.address_type.nulldata:
            cluster_idx = map_address2cluster_dict[address.type][address.address_num]
            cluster_ids.add(cluster_idx)
    return cluster_ids

def getActiveClustersCounts2Np(blocks, map_address2cluster, interval):
    clusters_counts = []
    for i in range(0, len(blocks), interval):
        subchain = blocks[i: i + interval]
        clusters_counts.append(len(getClusterIdsOfBlocksNp(subchain, map_address2cluster)))
    return clusters_counts


# Q_GINIS
def my_gini(array):
    array = np.sort(array)
    n = array.shape[0]
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))



def getClustersBalanceChangeNp(block, map_address2cluster_dict, cluster_balances_np):
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        cluster_balances_np[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        cluster_balances_np[cluster_idx] -= in_.value
    return cluster_balances_np

def getClustersBalanceChangeBlocksNp(blocks, map_address2cluster, cluster_balances_np):
    for b in blocks:
        cluster_balances_np = getClustersBalanceChangeNp(b, map_address2cluster, cluster_balances_np)
    return cluster_balances_np

def getClustersBalanceChangeNp2(block, map_address2cluster_dict, cluster_balances_np):
    fee_is_collected = True
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster_dict[miner.type][miner.address_num]
            cluster_balances_np[cluster_idx] += block.fee
        else:
            #print(UNKNOWN, 'block height: {}, value is {}'.format(block.height, block.fee))
            fee_is_collected = False
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        cluster_balances_np[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        cluster_balances_np[cluster_idx] -= in_.value
    return cluster_balances_np, fee_is_collected

def getClustersBalanceChangeBlocksNp2(blocks, map_address2cluster, cluster_balances_np):
    print('Et ya2!')
    bad_blocks = {}
    for b in blocks:
        cluster_balances_np, fee_is_collected = getClustersBalanceChangeNp2(b, map_address2cluster, cluster_balances_np)
        if not fee_is_collected:
            bad_blocks[str(b.height)] = b.fee
    return cluster_balances_np, bad_blocks
    
def getGinisOfClustersNp(blocks, map_address2cluster, interval):
    clusters_count = max(add2clustermap[1:].max() for add2clustermap in map_address2cluster.values()) + 1
    print(clusters_count)
    cluster_balances = np.zeros(clusters_count)
    nulldata_map = map_address2cluster[blocksci.address_type.nulldata]
    nulldata_cluster_ids = list(set(nulldata_map[1:nulldata_map[0]]))
    ginis = []
    for i in range(0, len(blocks), interval):
        cur_blocks = blocks[i: i + interval]
        cluster_balances = getClustersBalanceChangeBlocksNp(cur_blocks, map_address2cluster, cluster_balances)
        cluster_balances[nulldata_cluster_ids] = 0.
        ginis.append(my_gini(cluster_balances[cluster_balances.nonzero()]))
    return ginis


# Q_CLUSTER_BALANCE_CHANGE

def my_zero():
    return 0

def getClustersBalanceChangeDict(block, map_address2cluster_dict):
    balance_deltas = collections.defaultdict(my_zero)
    fee_is_collected = True
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster_dict[miner.type][miner.address_num]
            balance_deltas[cluster_idx] += block.fee
        else:
            #print(UNKNOWN, 'block height: {}, value is {}'.format(block.height, block.fee))
            fee_is_collected = False
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        balance_deltas[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        balance_deltas[cluster_idx] -= in_.value
    return balance_deltas#, fee_is_collected

class getClustersBalanceChangeDict_classEdition:
    def __init__(self, map_address2cluster):
        self.map_address2cluster = map_address2cluster
    def DO_IT(self, block):
        return getClustersBalanceChangeDict(block, self.map_address2cluster)

def sumClusterDicts(balance_change_dicts):
    result_dict = balance_change_dicts[0]
    for bcd in balance_change_dicts[1:]:
        for cl_idx in bcd:
            result_dict[cl_idx] += bcd[cl_idx]
    return result_dict

def getClustersBalanceChangeByGroupsPar(chain, max_block, map_address2cluster, interval):
    helper = getClustersBalanceChangeDict_classEdition(map_address2cluster)
    balance_change_dicts = chain.map_blocks(helper.DO_IT, end=max_block)
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getClustersBalanceChangeByGroupsNonPar(chain, max_block, map_address2cluster, interval):
    balance_change_dicts = [getClustersBalanceChangeDict(b, map_address2cluster) for b in chain[:max_block]]        
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getNonEmptyClustersCountsSoFar(chain, max_block, map_address2cluster, interval, clusters_count, parrallel=True):
    if parrallel:
        balance_change_dicts = getClustersBalanceChangeByGroupsPar(chain, max_block, map_address2cluster, interval)
    else:
        balance_change_dicts = getClustersBalanceChangeByGroupsNonPar(chain, max_block, map_address2cluster, interval)

    nonEmptyClustersCountSoFar = []
    clusters_balances_np = np.zeros(clusters_count)
    for bcd in balance_change_dicts:
        for cl_idx in bcd:
            clusters_balances_np[cl_idx] += bcd[cl_idx]
        nonEmptyClustersCountSoFar.append(len(np.nonzero(clusters_balances_np)[0]))
    return nonEmptyClustersCountSoFar

# RESEARCH
def getClustersBalanceChangeDict_woFees(block, map_address2cluster_dict):
    balance_deltas = collections.defaultdict(my_zero)
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        balance_deltas[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        balance_deltas[cluster_idx] -= in_.value
    return balance_deltas

class getClustersBalanceChangeDict_woFees_classEdition:
    def __init__(self, map_address2cluster):
        self.map_address2cluster = map_address2cluster
    def DO_IT(self, block):
        return getClustersBalanceChangeDict_woFees(block, self.map_address2cluster)

def getClustersBalanceChangeByGroupsWoFeesPar(chain, max_block, map_address2cluster, interval):
    helper = getClustersBalanceChangeDict_woFees_classEdition(map_address2cluster)
    balance_change_dicts = chain.map_blocks(helper.DO_IT, end=max_block)
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getNewbornClusters(chain, max_block, map_address2cluster, interval, born_from, born_to):
    balance_change_dicts = getClustersBalanceChangeByGroupsWoFeesPar(chain, max_block, map_address2cluster, interval)

    print("Going for oldborns!")
    oldborns_set = set()
    for ind in range(born_from):
        bcd = balance_change_dicts[ind]
        oldborns_set.update(set(bcd.keys()))

    print("Going for newborns!")
    newborns_set = set()
    for ind in range(born_from, born_to):
        bcd = balance_change_dicts[ind]
        cl_ids = set(bcd.keys())
        newborns_set.update(cl_ids - oldborns_set)
    return newborns_set

def getTxesWithNewBorns(block, map_address2cluster, newborns_set):
    def add2cl(address):
        return map_address2cluster[address.type][address.address_num]

    txes_with_newborns = []
    for tx in block.txes:
        io_cls = set([add2cl(i.address) for i in tx.inputs] + [add2cl(o.address) for o in tx.outputs])
        if len(io_cls & newborns_set) > 0:
            txes_with_newborns.append(tx.index)
    return txes_with_newborns

class getTxesWithNewBorns_classEdition:
    def __init__(self, map_address2cluster, newborns_set):
        self.map_address2cluster = map_address2cluster
        self.newborns_set = newborns_set
    def DO_IT(self, block):
        return getTxesWithNewBorns(block, self.map_address2cluster, self.newborns_set)

import itertools
def getTxesWithNewBornPar(chain, map_address2cluster, newborns_set, start=None, end=None):
    map_function = getTxesWithNewBorns_classEdition(map_address2cluster, newborns_set).DO_IT
    result_transactions_grouped = chain.map_blocks(map_function, start=start, end=end)
    return list(itertools.chain(*result_transactions_grouped))





def getClustersBalanceChangeDict_filteringClusters(block, map_address2cluster, clusters_2b_ignored):
    balance_deltas = collections.defaultdict(my_zero)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster[miner.type][miner.address_num]
            if not cluster_idx in clusters_2b_ignored:
                balance_deltas[cluster_idx] += block.fee
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster[add.type][add.address_num]
        if not cluster_idx in clusters_2b_ignored:
            balance_deltas[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster[add.type][add.address_num]
        if not cluster_idx in clusters_2b_ignored:
            balance_deltas[cluster_idx] -= in_.value
    return balance_deltas#, fee_is_collected


class getClustersBalanceChangeDict_filteringClusters_classEdition:
    def __init__(self, map_address2cluster, clusters_2b_ignored):
        self.map_address2cluster = map_address2cluster
        self.clusters_2b_ignored = clusters_2b_ignored
    def DO_IT(self, block):
        return getClustersBalanceChangeDict_filteringClusters(block, self.map_address2cluster, self.clusters_2b_ignored)

def getClustersBalanceChangeDict_filteringTxes(block, map_address2cluster, txes_2b_ignored):
    balance_deltas = collections.defaultdict(my_zero)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster[miner.type][miner.address_num]
            balance_deltas[cluster_idx] += block.fee
    for tx in block.txes:
        if tx.index in txes_2b_ignored:
            continue
        for out in tx.outputs:
            add = out.address
            cluster_idx = map_address2cluster[add.type][add.address_num]
            balance_deltas[cluster_idx] += out.value
        for in_ in tx.inputs:
            add = in_.address
            cluster_idx = map_address2cluster[add.type][add.address_num]
            balance_deltas[cluster_idx] -= in_.value
    return balance_deltas#, fee_is_collected

class getClustersBalanceChangeDict_filteringTxes_classEdition:
    def __init__(self, map_address2cluster, txes_2b_ignored):
        self.map_address2cluster = map_address2cluster
        self.txes_2b_ignored = txes_2b_ignored
    def DO_IT(self, block):
        return getClustersBalanceChangeDict_filteringTxes(block, self.map_address2cluster, self.txes_2b_ignored)

def getClustersBalanceChangeDict_filteringUnspents(block, map_address2cluster):
    balance_deltas = collections.defaultdict(my_zero)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster[miner.type][miner.address_num]
            balance_deltas[cluster_idx] += block.fee
    for out in block.outputs:
        if out.is_spent:
            add = out.address
            cluster_idx = map_address2cluster[add.type][add.address_num]
            balance_deltas[cluster_idx] += out.value
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster[add.type][add.address_num]
        balance_deltas[cluster_idx] -= in_.value
    return balance_deltas#, fee_is_collected


class getClustersBalanceChangeDict_filteringUnspents_classEdition:
    def __init__(self, map_address2cluster):
        self.map_address2cluster = map_address2cluster
    def DO_IT(self, block):
        return getClustersBalanceChangeDict_filteringUnspents(block, self.map_address2cluster)


def getClustersBalanceChangeByGroupsPar_filtering(chain, max_block, interval, map_address2cluster, clusters_2b_ignored,
    txes_2b_ignored):
    if not clusters_2b_ignored is None:
        map_function = getClustersBalanceChangeDict_filteringClusters_classEdition(map_address2cluster, clusters_2b_ignored).DO_IT
    elif not txes_2b_ignored is None:
        map_function = getClustersBalanceChangeDict_filteringTxes_classEdition(map_address2cluster, txes_2b_ignored).DO_IT
    else:
        map_function = getClustersBalanceChangeDict_filteringUnspents_classEdition(map_address2cluster).DO_IT

    balance_change_dicts = chain.map_blocks(map_function, end=max_block)
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)


def getClustersBalanceChangeByGroupsNonPar_filtering(chain, max_block, interval, map_address2cluster, clusters_2b_ignored,
    txes_2b_ignored):
    if not clusters_2b_ignored is None:
        map_function = getClustersBalanceChangeDict_filteringClusters_classEdition(map_address2cluster, clusters_2b_ignored).DO_IT
    elif not txes_2b_ignored is None:
        map_function = getClustersBalanceChangeDict_filteringTxes_classEdition(map_address2cluster, txes_2b_ignored).DO_IT
    else:
        map_function = getClustersBalanceChangeDict_filteringUnspents_classEdition(map_address2cluster).DO_IT
    balance_change_dicts = [map_function(b) for b in chain[:max_block]]        
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getNonEmptyClustersCountsSoFar_filtering(chain, max_block, map_address2cluster, interval, clusters_count, clusters_2b_ignored,
    txes_2b_ignored, parrallel=True):
    if parrallel:
        balance_change_dicts = getClustersBalanceChangeByGroupsPar_filtering(chain, max_block, interval=interval,
            map_address2cluster=map_address2cluster, clusters_2b_ignored=clusters_2b_ignored, txes_2b_ignored=txes_2b_ignored)
    else:
        balance_change_dicts = getClustersBalanceChangeByGroupsNonPar_filtering(chain, max_block, interval=interval,
            map_address2cluster=map_address2cluster, clusters_2b_ignored=clusters_2b_ignored, txes_2b_ignored=txes_2b_ignored)

    nonEmptyClustersCountSoFar = []
    clusters_balances_np = np.zeros(clusters_count)
    for bcd in balance_change_dicts:
        for cl_idx in bcd:
            clusters_balances_np[cl_idx] += bcd[cl_idx]
        nonEmptyClustersCountSoFar.append(len(np.nonzero(clusters_balances_np)[0]))
    return nonEmptyClustersCountSoFar

# Q_ADDRESS_BALANCE_CHANGE

map_addtype2num = {}
addtype_num = 0
for add_type in blocksci.address_type.types:
    map_addtype2num[add_type] = addtype_num
    map_addtype2num[addtype_num] = add_type
    addtype_num += 1

def add2uniAdd(address):
    return address.address_num * 10 + hash(address.type)

def uniAdd2add(uniAddress):
    return map_addtype2num[uniAddress % 10], uniAddress // 10

def add2cl(address, map_address2cluster):
    return map_address2cluster[address.type][address.address_num]

def uniAdd2cl(uniAddress, map_address2cluster):
    return map_address2cluster[map_addtype2num[uniAddress % 10]][uniAddress // 10]

def getAddressesBalanceChangeDict(block):
    balance_deltas = collections.defaultdict(my_zero)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            balance_deltas[add2uniAdd(miner)] += block.fee
    for out in block.outputs:
        add = out.address
        balance_deltas[add2uniAdd(add)] += out.value
    for in_ in block.inputs:
        add = in_.address
        balance_deltas[add2uniAdd(add)] -= in_.value
    return balance_deltas#, fee_is_collected

def getAddressesBalanceChangeDict_filtering(block, map_address2cluster, clusters_2b_ignored):
    balance_deltas = collections.defaultdict(my_zero)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            if not add2cl(miner, map_address2cluster) in clusters_2b_ignored:
                balance_deltas[add2uniAdd(miner)] += block.fee
    for out in block.outputs:
        add = out.address
        if not add2cl(add, map_address2cluster) in clusters_2b_ignored:
            balance_deltas[add2uniAdd(add)] += out.value
    for in_ in block.inputs:
        add = in_.address
        if not add2cl(add, map_address2cluster) in clusters_2b_ignored:
            balance_deltas[add2uniAdd(add)] -= in_.value
    return balance_deltas#, fee_is_collected


class getAddressesBalanceChangeDict_filtering_classEdition:
    def __init__(self, map_address2cluster, clusters_2b_ignored):
        self.map_address2cluster = map_address2cluster
        self.clusters_2b_ignored = clusters_2b_ignored
    def DO_IT(self, block):
        return getAddressesBalanceChangeDict_filtering(block, self.map_address2cluster, self.clusters_2b_ignored)

def getAddressesBalanceChangeByGroupsPar(chain, max_block, interval, map_address2cluster, clusters_2b_ignored):
    if not map_address2cluster is None:
        map_function = getAddressesBalanceChangeDict_filtering_classEdition(map_address2cluster, clusters_2b_ignored).DO_IT
    else:
        map_function = getAddressesBalanceChangeDict
    balance_change_dicts = chain.map_blocks(map_function, end=max_block)
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getAddressesBalanceChangeByGroupsNonPar(chain, max_block, interval, map_address2cluster, clusters_2b_ignored):
    if not map_address2cluster is None:
        balance_change_dicts = [getAddressesBalanceChangeDict_filtering(b, map_address2cluster, clusters_2b_ignored)
            for b in chain[:max_block]]  
    else:
        balance_change_dicts = [getAddressesBalanceChangeDict(b) for b in chain[:max_block]]  
    balance_change_dicts = [getAddressesBalanceChangeDict(b, map_address2cluster, clusters_2b_ignored) for b in chain[:max_block]]        
    return aggregateByGroups(balance_change_dicts, interval, agg_function=sumClusterDicts)

def getNonEmptyAddressesCountsSoFar(chain, max_block, interval, addresses_count, parrallel=True, map_address2cluster=None,
    clusters_2b_ignored=None):
    if parrallel:
        balance_change_dicts = getAddressesBalanceChangeByGroupsPar(chain, max_block, interval,
            map_address2cluster, clusters_2b_ignored)
    else:
        balance_change_dicts = getAddressesBalanceChangeByGroupsNonPar(chain, max_block, interval,
            map_address2cluster, clusters_2b_ignored)

    nonEmptyAddressesCountSoFar = []
    addresses_balances_np = np.zeros(addresses_count)
    ind = 0
    for bcd in balance_change_dicts:
        for uniAdd in bcd:
            #if uniAdd2cl(uniAdd, map_address2cluster) in clusters_2b_ignored:
            #    print("PIZDA IVANOVNA!")
            addresses_balances_np[uniAdd] += bcd[uniAdd]
        nonEmptyAddressesCountSoFar.append(len(np.nonzero(addresses_balances_np)[0]))
        ind += 1
        if ind % 100 == 0:
            print("{} groups are done".format(ind))
    return nonEmptyAddressesCountSoFar

def getNonEmptyAddressesCountsSoFarWONullData(chain, max_block, interval, addresses_count, parrallel=True):
    if parrallel:
        balance_change_dicts = getAddressesBalanceChangeByGroupsPar(chain, max_block, interval)
    else:
        balance_change_dicts = getAddressesBalanceChangeByGroupsNonPar(chain, max_block, interval)

    nonEmptyAddressesCountSoFar = []
    addresses_balances_np = np.zeros(addresses_count)
    ind = 0
    nulldata_ind = 6
    for bcd in balance_change_dicts:
        for uniAdd in bcd:
            if uniAdd % 10 != nulldata_ind:
                addresses_balances_np[uniAdd] += bcd[uniAdd]
        nonEmptyAddressesCountSoFar.append(len(np.nonzero(addresses_balances_np)[0]))
        ind += 1
        if ind % 100 == 0:
            print("{} groups are done".format(ind))
    return nonEmptyAddressesCountSoFar


def getClustersBalanceChangeBlocksPar(chain, start, end, map_address2cluster, cluster_balances_np):
    helper = getClustersBalanceChangeDict_classEdition(map_address2cluster)
    balance_change_dicts = chain.map_blocks(helper.DO_IT, start=start, end=end)
    #flow_income_pairs = [helper.DO_IT(b) for b in chain[:max_block]]
    
    for balance_change_dict_and_fee in balance_change_dicts:
        balance_change_dict = balance_change_dict_and_fee[0]
        for cl_idx in balance_change_dict:
            cluster_balances_np[cl_idx] += balance_change_dict[cl_idx]
    return cluster_balances_np

def getClustersBalanceChangeBlocksNonPar(chain, start, end, map_address2cluster, cluster_balances_np):
    helper = getClustersBalanceChangeDict_classEdition(map_address2cluster)
    #balance_change_dicts = chain.map_blocks(helper.DO_IT, start=start, end=end)
    balance_change_dicts = [helper.DO_IT(b) for b in chain[start:end]]
    
    for balance_change_dict_and_fee in balance_change_dicts:
        balance_change_dict = balance_change_dict_and_fee[0]
        for cl_idx in balance_change_dict:
            cluster_balances_np[cl_idx] += balance_change_dict[cl_idx]
    return cluster_balances_np


#Q_BALANCE_FLOW_TXCOUNT

def my_zero3():
    return np.zeros(3, dtype=int)

def updateBalanceFlowTxCount(balance_flow_txcnt, value):
    if value != 0:
        balance_flow_txcnt[0] += value
        balance_flow_txcnt[1] += abs(value)
        balance_flow_txcnt[2] += 1

def getClustersBalanceFlowTxCountChangeDict(block, map_address2cluster_dict):
    balance_flow_txcnt_deltas = collections.defaultdict(my_zero3)
    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
            cluster_idx = map_address2cluster_dict[miner.type][miner.address_num]
            updateBalanceFlowTxCount(balance_flow_txcnt_deltas[cluster_idx], block.fee)
    for out in block.outputs:
        add = out.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        updateBalanceFlowTxCount(balance_flow_txcnt_deltas[cluster_idx], out.value)
    for in_ in block.inputs:
        add = in_.address
        cluster_idx = map_address2cluster_dict[add.type][add.address_num]
        updateBalanceFlowTxCount(balance_flow_txcnt_deltas[cluster_idx], -in_.value)
    return balance_flow_txcnt_deltas

class getClustersBalanceFlowTxCountChangeDict_classEdition:
    def __init__(self, map_address2cluster):
        self.map_address2cluster = map_address2cluster
    def DO_IT(self, block):
        return getClustersBalanceFlowTxCountChangeDict(block, self.map_address2cluster)

def getClustersBalanceFlowTxCountChangeDictBlocksPar(chain, max_block, map_address2cluster):
    helper = getClustersBalanceFlowTxCountChangeDict_classEdition(map_address2cluster)
    balance_flow_txcnt_deltas = chain.map_blocks(helper.DO_IT, end=max_block)
    return sumClusterDicts(balance_flow_txcnt_deltas)





def getAddressesBalanceFlowTxCountChangeDict(block):
    balance_flow_txcnt_deltas = collections.defaultdict(my_zero3)

    if block.fee > 0:
        miner = guessMinerAdds3((block,))[0]
        if miner != UNKNOWN:
#            add_idx = 
            updateBalanceFlowTxCount(balance_flow_txcnt_deltas[add2uniindex(miner)], block.fee)
    for out in block.outputs:
        add = out.address
        updateBalanceFlowTxCount(balance_flow_txcnt_deltas[add2uniindex(add)], out.value)
    for in_ in block.inputs:
        add = in_.address
        updateBalanceFlowTxCount(balance_flow_txcnt_deltas[add2uniindex(add)], -in_.value)
    return balance_flow_txcnt_deltas


def getAddressesBalanceFlowTxCountChangeDictBlocksPar(chain, start, end):
    #end = min(end, len(chain))
    balance_flow_txcnt_deltas = chain.map_blocks(getAddressesBalanceFlowTxCountChangeDict, start=start, end=end)
    return sumClusterDicts(balance_flow_txcnt_deltas)

def getAddressesBalanceFlowTxCountChangeDictBlocksNonPar(chain, start, end):
    #end = min(end, len(chain))
    balance_flow_txcnt_deltas = [getAddressesBalanceFlowTxCountChangeDict(b) for b in chain[start:end]]
    return sumClusterDicts(balance_flow_txcnt_deltas)



#Q_REL_FLOWS
def getFlowAndIncomeVolumeForOneBlock(block, map_address2cluster):
    balance_deltas = collections.defaultdict(lambda: 0)

    for out in block.outputs:
        if out.tx.is_coinbase:
            cluster_idx = 'income'
        else:
            add = out.address
            cluster_idx = map_address2cluster[add.type][add.address_num]
        balance_deltas[cluster_idx] += out.value
    for in_ in block.inputs:
        #if in_.tx.is_coinbase:
        #    continue
        add = in_.address
        cluster_idx = map_address2cluster[add.type][add.address_num]
        balance_deltas[cluster_idx] -= in_.value
    
    balance_deltas['fee'] = block.fee
    income = balance_deltas['income']
    flow = sum([abs(delta) for delta in balance_deltas.values()]) - income

    return flow, income

class getFlowAndIncomeVolumeForOneBlock_classEdition:
    def __init__(self, map_address2cluster):
        self.map_address2cluster = map_address2cluster
    def DO_IT(self, block):
        return getFlowAndIncomeVolumeForOneBlock(block, self.map_address2cluster)

def getFlowAndIncomeVolume(chain, map_address2cluster, interval, max_block=None):
    if max_block is None:
        max_block = len(chain)
    helper = getFlowAndIncomeVolumeForOneBlock_classEdition(map_address2cluster)
    #flow_income_pairs = chain.map_blocks(helper.DO_IT, end=max_block)
    flow_income_pairs = [helper.DO_IT(b) for b in chain[:max_block]]
    
    flows = []
    incomes = []
    for i in range(0, max_block, interval):
        each_block_flows_incomes = list(zip(*flow_income_pairs[i: i + interval]))
        flows.append(sum(each_block_flows_incomes[0]))
        incomes.append(sum(each_block_flows_incomes[1]))
    return flows, incomes

def getFlowAndIncomeVolumePar(chain, map_address2cluster, interval, max_block=None):
    if max_block is None:
        max_block = len(chain)
    helper = getFlowAndIncomeVolumeForOneBlock_classEdition(map_address2cluster)
    flow_income_pairs = chain.map_blocks(helper.DO_IT, end=max_block)
    
    flows = []
    incomes = []
    for i in range(0, max_block, interval):
        each_block_flows_incomes = list(zip(*flow_income_pairs[i: i + interval]))
        flows.append(sum(each_block_flows_incomes[0]))
        incomes.append(sum(each_block_flows_incomes[1]))
    return flows, incomes

def getRelativeFlowVolumes(flows, incomes):
    relativeFlowVolumes = []
    total_income = 0
    for i in range(len(flows)):
        total_income += incomes[i]
        relativeFlowVolumes.append(flows[i] / total_income)
    return relativeFlowVolumes


# Q_FEE_REVENUE
def getFees(blocks, interval):
    each_block_fees = [b.fee for b in blocks]
    grouped_fees = []
    for i in range(0, len(blocks), interval):
        grouped_fees.append(sum(each_block_fees[i: i + interval]))
    return grouped_fees

def getFeesPar(chain, max_block, interval):
    each_block_fees = chain.map_blocks(lambda b: b.fee, end=max_block)
    grouped_fees = []
    for i in range(0, max_block, interval):
        grouped_fees.append(sum(each_block_fees[i: i + interval]))
    return grouped_fees

def getRevenues(blocks, interval):
    each_block_revenues = [b.revenue for b in blocks]
    grouped_revenues = []
    for i in range(0, len(blocks), interval):
        grouped_revenues.append(sum(each_block_revenues[i: i + interval]))
    return grouped_revenues

def getRevenuesPar(chain, max_block, interval):
    each_block_revenues = chain.map_blocks(lambda b: b.revenue, end=max_block)
    grouped_revenues = []
    for i in range(0, max_block, interval):
        grouped_revenues.append(sum(each_block_revenues[i: i + interval]))
    return grouped_revenues


#Q_UNSPENTS
def calculateUnspent(block):
    unspent = 0
    for o in block.outputs:
        if o.is_spent == False:
            unspent += o.value
    return unspent

def getUnspentsByGroups(blocks, interval):
    each_block_unspents = [calculateUnspent(b) for b in blocks]
    return aggregateByGroups(each_block_unspents, interval)

def getUnspentsByGroupsPar(chain, max_block, interval):
    each_block_unspents = chain.map_blocks(calculateUnspent, end=max_block)
    return aggregateByGroups(each_block_unspents, interval)


#Q_DIFF
def getDifficulties(chain, max_block, interval):
    bits = chain[:max_block].bits
    return aggregateByGroups(bits, interval)



# Q_USEFUL
def checkFlowSum2(blocks):
    flowSum = 0
    for b in blocks:
        for out in b.outputs:
            if out.tx.is_coinbase:
                continue
            flowSum += out.value
        for in_ in b.inputs:
            if in_.tx.is_coinbase:
                continue
            flowSum -= in_.value
        flowSum += b.fee
    return flowSum

def meanDfColumns(pd_df, col_win_dict):
    result_df_dict = {}
    max_window = max(col_win_dict.values())
    first_pos = max_window // 2
    result_len = pd_df.shape[0] - max_window + 1
    for col in col_win_dict:
        window = col_win_dict[col]
        if window > 1:
            col_after_rolling = pd.rolling_mean(pd_df[col], window=window, center=True)
        else:
            col_after_rolling = pd_df[col]
        result_df_dict[col] = col_after_rolling[first_pos: first_pos + result_len]
    return pd.DataFrame(result_df_dict)


def aggregateByGroups(values, interval, agg_function=sum):
    aggregated = []
    for i in range(0, len(values), interval):
        aggregated.append(agg_function(values[i: i + interval]))
    return aggregated

def runningMean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)



# Q_SAVE_READ
import json
def saveJson(data, path):
    with open(path + '.json', 'w') as fout:
        fout.write(json.dumps(data))
def readJson(path):
    with open(path + '.json', 'r') as fin:
        data = json.loads(fin.read())
    return data
        

import h5py
def saveMapNpInHdfs(the_map, filename):
    with h5py.File(filename + '.hdf5', 'w') as fout:
        fout.create_dataset(name='nonstandard', data = the_map[blocksci.address_type.nonstandard])
        fout.create_dataset(name='pubkey', data = the_map[blocksci.address_type.pubkey])
        fout.create_dataset(name='pubkeyhash', data = the_map[blocksci.address_type.pubkeyhash])
        fout.create_dataset(name='multisig_pubkey', data = the_map[blocksci.address_type.multisig_pubkey])
        fout.create_dataset(name='scripthash', data = the_map[blocksci.address_type.scripthash])
        fout.create_dataset(name='multisig', data = the_map[blocksci.address_type.multisig])
        fout.create_dataset(name='nulldata', data = the_map[blocksci.address_type.nulldata])
        fout.create_dataset(name='witness_pubkeyhash', data = the_map[blocksci.address_type.witness_pubkeyhash])
        fout.create_dataset(name='witness_scripthash', data = the_map[blocksci.address_type.witness_scripthash])
        
def readMapNpFromHdfs(filename):
    the_map = {}
    with h5py.File(filename + '.hdf5', 'r') as fin:
        the_map[blocksci.address_type.nonstandard] = fin['nonstandard'][()]
        the_map[blocksci.address_type.pubkey] = fin['pubkey'][()]
        the_map[blocksci.address_type.pubkeyhash] = fin['pubkeyhash'][()]
        the_map[blocksci.address_type.multisig_pubkey] = fin['multisig_pubkey'][()]
        the_map[blocksci.address_type.scripthash] = fin['scripthash'][()]
        the_map[blocksci.address_type.multisig] = fin['multisig'][()]
        the_map[blocksci.address_type.nulldata] = fin['nulldata'][()]
        the_map[blocksci.address_type.witness_pubkeyhash] = fin['witness_pubkeyhash'][()]
        the_map[blocksci.address_type.witness_scripthash] = fin['witness_scripthash'][()]
    return the_map

import pandas as pd
def saveCSV(pd_df, filename):
    pd_df.to_csv(filename + '.csv', index=False)
    
def readCSV(filename):
    return pd.read_csv(filename + '.csv')



# Q_DATA_VERSIONS
import os
from os import path
import re

class DataVersions:
    def __init__(self,
                 common_path_prefix,
                 dataname,
                 default_save_function=saveJson,
                 default_read_function=readJson):
        if not common_path_prefix is None:
            self.common_path_prefix = common_path_prefix + '_' + dataname
        self.dataname = dataname
        # versions
        self.v = {}
        self.default_save_function = default_save_function
        self.default_read_function = default_read_function
        
    def __getitem__(self, key):
        return self.v[key]

    def add(self, key, data, overwrite=False):
        if key in self.v:
            if overwrite:
                print("There WAS already {} version of data, now overwitten".format(key))
                self.v[key] = data
            else:
                print("There is already {} version of data, stop".format(key))
        else:
            self.v[key] = data
    
    def save(self, key, suffix=None, save_function=None, check=True, read_function=None):
        if suffix is None:
            suffix = key
        path_to_file = '{}{}{}'.format(self.common_path_prefix,
                                       '_' if suffix != '' else '',
                                       suffix)
        if save_function is None:
            save_function = self.default_save_function
        if os.path.isfile(path_to_file):
            print("There is already \"{}\" file, stop".format(path_to_file))
        else:
            print("Saving \"{}\" version in \"{}\" file".format(key, path_to_file))
            data_to_save = self.v[key]
            save_function(data_to_save, path_to_file)
            if check:
                if read_function is None:
                    read_function = self.default_read_function
                if data_to_save != read_function(path_to_file):
                    print('  Save is fucked up! Saved and read is not the same')
            
    def read(self, key, suffix=None, read_function=None):
        if suffix is None:
            suffix = key
        path_to_file = '{}{}{}'.format(self.common_path_prefix,
                                       '_' if suffix != '' else '',
                                       suffix)
        if read_function is None:
            read_function = self.default_read_function
        read_data = read_function(path_to_file)
        self.add(key, read_data)
    
    def readAll(self, dry_run=False):
        splitted_path = path.split(self.common_path_prefix)
        folder, file_prefix = path.join(*splitted_path[:-1]), splitted_path[-1]
        print('Reading folder \'{}\' with prefix \'{}\'...'.format(folder, file_prefix))
        RE_TEMPLATE = file_prefix + '(.*)\..*'
        for f in os.listdir(folder):
            m = re.search(RE_TEMPLATE, f)
            if not m is None:
                key = m.group(1)
                if key != '':
                    # removing leading '_' char
                    key = key[1:]
                print('  Got \'{}\', key is \'{}\''.format(f, key))
                if not dry_run:
                    self.read(key)
        print('Finish')



# Q_COIN_DATA_MGR
S_MINERS = 'miners'
S_NCS = 'NacamotoCoefs'
S_ADDS_CNTS = 'activeAddressesCounts'
S_MAP_A2C = 'map_address2cluster'
S_CLS_CNTS = 'activeClustersCounts'
S_GINIS = 'ginisOfClustersWealth'
S_FLOWS_INCOMES = 'flowAndIncomeVolumes'
S_FEES = 'fees'
S_REVENUES = 'revenues'
S_UNSPENTS = 'unspents'
S_NONEMPTY_CLS = 'nonEmptyClustersCounts'
S_NONEMPTY_ADDS = 'nonEmptyAddressesCounts'
S_DIFFS = 'blockDifficulties'
S_SYNC_PRICES = 'synchronizedPrices'
S_PRICES = 'prices'

FOR_CHARTS = {
    S_NCS: ('оц. коэффициента Накамото', 'майнеры'),
    S_CLS_CNTS: ('лог. числа активных пользователей за период', ''),
    S_GINIS: ('log(1 - g), g - коэф. Джини распределения валюты', ''),
    S_FEES: ('лог. комиссии (в cатоши) за период', ''),
    S_UNSPENTS: ('лог. объёма (в cатоши) непотраченных выходов периода', ''),
    S_NONEMPTY_CLS: ('лог. суммарного числа пользователей', ''),
    S_REL_FLOWS: (' лог. доли оборота валюты за период', '')
}



import time
def measure_time(f):
    def wrapper(*args, **kwargs):
        start_time_allah_velik = time.time()
        result = f(*args, **kwargs)
        print('Work time {}s'.format(round(time.time() - start_time_allah_velik, 2)))
        return result
    return wrapper

class CoinDataMgr:
    def __init__(self, blocksci_path, path_to_clusters, folder_with_calculated, max_block=None, group_size=1000):
        self.chain = blocksci.Blockchain(blocksci_path)
        self.blocks = self.chain[:max_block]
        print("Got {} blocks".format(len(self.blocks)))
        
        #self.cl_mgr = blocksci.cluster.ClusterManager(path_to_clusters, self.chain)
        #self.cl_mgr.create_clustering(location=path_to_clusters, chain=self.chain)
        self.cl_mgr = blocksci.cluster.ClusterManager(path_to_clusters, self.chain)
        print("Got {} clusters".format(len(self.cl_mgr.clusters())))
        
        self.group_size = group_size
        self.times = getTimes(self.blocks, self.group_size)
        
        self.files_prefix = folder_with_calculated + '/int{}b'.format(self.group_size)
        
        # data
        self.d = {}
        self.d[S_MINERS] = DataVersions(None, S_MINERS)
        self.d[S_NCS] = DataVersions(self.files_prefix, S_NCS)
        self.d[S_ADDS_CNTS] = DataVersions(self.files_prefix, S_ADDS_CNTS)
        self.d[S_MAP_A2C] = DataVersions(self.files_prefix, S_MAP_A2C,
                                                default_save_function=saveMapNpInHdfs,
                                                default_read_function=readMapNpFromHdfs)
        self.d[S_CLS_CNTS] = DataVersions(self.files_prefix, S_CLS_CNTS)
        self.d[S_GINIS] = DataVersions(self.files_prefix, S_GINIS)
        self.d[S_FLOWS_INCOMES] = DataVersions(self.files_prefix, S_FLOWS_INCOMES)
        self.d[S_REL_FLOWS] = DataVersions(self.files_prefix, S_REL_FLOWS)
        self.d[S_FEES] = DataVersions(self.files_prefix, S_FEES)
        self.d[S_REVENUES] = DataVersions(self.files_prefix, S_REVENUES)
        self.d[S_UNSPENTS] = DataVersions(self.files_prefix, S_UNSPENTS)
        self.d[S_NONEMPTY_CLS] = DataVersions(self.files_prefix, S_NONEMPTY_CLS)
        self.d[S_NONEMPTY_ADDS] = DataVersions(self.files_prefix, S_NONEMPTY_ADDS)
        self.d[S_DIFFS] = DataVersions(self.files_prefix, S_DIFFS)
        self.d[S_SYNC_PRICES] = DataVersions(self.files_prefix, S_SYNC_PRICES)
        self.allMetrics = DataVersions(self.files_prefix, 'allMetrics', default_save_function=saveCSV,
                                                                        default_read_function=readCSV)

        self.prices = DataVersions(self.files_prefix, 'prices', default_save_function=saveCSV,
                                                                default_read_function=readCSV)
        
    def __getitem__(self, key):
        return self.d[key]

    @measure_time
    def guessMiners(self):
        self.d[S_MINERS].add('', guessMinerAdds3(self.blocks))
        
    @measure_time
    def getNCs(self):
        NCs, self.egos = calcNakamotoCoef2(self.d[S_MINERS][''], self.group_size)
        self.d[S_NCS].add('', NCs)
        
    @measure_time
    def getActiveAddressesCounts(self):
        self.d[S_ADDS_CNTS].add('', getActiveAddressesCounts(self.blocks, self.group_size))
        
    @measure_time
    def getMapAddress2Cluster(self, key='np'):
        all_addresses_dict = {}
        for add_type in blocksci.address_type.types:
            all_addresses_dict[add_type] = self.chain.addresses(add_type)
        if key == 'np':
            self.d[S_MAP_A2C].add(key, getMapAddress2Cluster2Np(self.cl_mgr, all_addresses_dict))
    
    @measure_time
    def getActiveClustersCounts(self, key='usingNpMap'):
        if key == 'usingNpMap':
            data = getActiveClustersCounts2Np(self.blocks, self.d[S_MAP_A2C].v['np'], self.group_size)
        self.d[S_CLS_CNTS].add(key, data)
        
    @measure_time
    def getGinis(self, key='usingNpMap'):
        if key == 'usingNpMap':
            data = getGinisOfClustersNp(self.blocks, self.d[S_MAP_A2C].v['np'], self.group_size)
        self.d[S_GINIS].add(key, data)

    @measure_time
    def getFlowAndIncomeVolume(self, key='par'):
        if key == 'par':
            flows, incomes = getFlowAndIncomeVolumePar(self.chain, self.d[S_MAP_A2C].v['np'],
                self.group_size, max_block=len(self.blocks))
        elif key == 'nonPar':
            flows, incomes = getFlowAndIncomeVolume(self.chain, self.d[S_MAP_A2C].v['np'],
                self.group_size, max_block=len(self.blocks))
        data = {'flows' : flows, 'incomes': incomes}
        self.d[S_FLOWS_INCOMES].add(key, data)
        
    @measure_time
    def getRelativeFlowVolumes(self, key='par'):
        if key in ('par', 'nonPar'):
            flows = self.d[S_FLOWS_INCOMES][key]['flows']
            incomes = self.d[S_FLOWS_INCOMES][key]['incomes']
            data = getRelativeFlowVolumes(flows, incomes)
        self.d[S_REL_FLOWS].add(key, data)

    @measure_time
    def getFees(self, key='par'):
        if key == 'par':
            data = getFeesPar(self.chain, len(self.blocks), self.group_size)
        elif key == 'nonPar':
            data = getFees(self.blocks, self.group_size)
        self.d[S_FEES].add(key, data)

    @measure_time
    def getRevenues(self, key='par'):
        if key == 'par':
            data = getRevenuesPar(self.chain, len(self.blocks), self.group_size)
        elif key == 'nonPar':
            data = getRevenues(self.blocks, self.group_size)
        self.d[S_REVENUES].add(key, data)

    @measure_time
    def getUnspents(self, key='par', last_weeks_to_nan=12):
        if key == 'par':
            data = getUnspentsByGroupsPar(self.chain, len(self.blocks), self.group_size)
        elif key == 'nonPar':
            data = getUnspentsByGroups(self.blocks, self.group_size)

        last_time_look_for = self.blocks[-1].time - datetime.timedelta(weeks=last_weeks_to_nan)
        for height in range(len(self.blocks) - 1, -1, -1):
            if self.blocks[height].time < last_time_look_for:
                last_block = height // self.group_size
                break
        for height in range(last_block + 1, len(data)):
            data[height] = None

        self.d[S_UNSPENTS].add(key, data)

    @measure_time
    def getNonEmptyClustersCounts(self, key='par', filtering=False, clusters_2b_ignored=None, txes_2b_ignored=None):
        if key == 'par':
            parrallel = True
        elif key == 'nonPar':
            parrallel = False
        else:
            raise None


        if filtering:
            key += '_filtering'
            if not clusters_2b_ignored is None:
                key += 'Clusters'
            elif not clusters_2b_ignored is None:
                key += 'Txes'
            else: 
                key += 'Unspents'
            data = getNonEmptyClustersCountsSoFar_filtering(self.chain, len(self.blocks), self.d[S_MAP_A2C].v['np'], self.group_size,
            clusters_count=len(self.cl_mgr.clusters()), clusters_2b_ignored=clusters_2b_ignored, txes_2b_ignored=txes_2b_ignored,
            parrallel=parrallel)
        else:
            data = getNonEmptyClustersCountsSoFar(self.chain, len(self.blocks), self.d[S_MAP_A2C].v['np'], self.group_size,
                clusters_count=len(self.cl_mgr.clusters()), parrallel=parrallel)


        self.d[S_NONEMPTY_CLS].add(key, data)

    @measure_time
    def getNonEmptyAddressesCounts(self, key='par', clusters_2b_ignored=None):
        if key == 'par':
            parrallel = True
        elif key == 'nonPar':
            parrallel = False
        else:
            raise None

        if not clusters_2b_ignored is None:
            the_map = self.d[S_MAP_A2C]['np']
            key += '_filtering'
        else:
            the_map = None

        addresses_count = self.chain.address_count(blocksci.address_type.pubkey) * 10 + 9
        data = getNonEmptyAddressesCountsSoFar(self.chain, len(self.blocks), self.group_size, addresses_count=addresses_count,
            parrallel=parrallel, map_address2cluster=the_map, clusters_2b_ignored=clusters_2b_ignored)
        self.d[S_NONEMPTY_ADDS].add(key, data)



    @measure_time
    def getDifficulties(self, key=''):
        data = getDifficulties(self.chain, len(self.blocks), self.group_size)
        self.d[S_DIFFS].add(key, data)
        
    def showDataAndVersions(self):
        for dataname in self.d:
            print('\'{}\' versions:'.format(dataname))
            for version in self.d[dataname].v:
                print('  \'{}\''.format(version))

    def gatherAllMetrics(self, metric_version_dict):
        dict_for_pd_df = {'times': self.times}
        tag = ''
        for metric in metric_version_dict:
            version = metric_version_dict[metric]
            dict_for_pd_df[metric] = self.d[metric].v[version]
            tag += metric + '=' + version + '_'
        tag = tag[:-1]
        print('Gathered under tag \'{}\''.format(tag))

        self.allMetrics.add(tag, pd.DataFrame(dict_for_pd_df))
        return tag

    def rollingMeanAllMetrics(self, allMetrics_tag, common_win=None, col_win_dict=None):
        if not common_win is None:
            tag_suffix = '_rolledMeanComWnd={}'.format(common_win)
            result_col_win_dict = {}
        if not col_win_dict is None:
            tag_suffix = '_rolledMean_wnds:' + '_'.join(['{}={}'.format(k, v) for k, v in col_win_dict.items()])
            result_col_win_dict = col_win_dict

        to_be_rolled = self.allMetrics[allMetrics_tag]
        for col in to_be_rolled.columns:
            if col == 'times':
                result_col_win_dict[col] = 1
                continue
            if col_win_dict is None:
                result_col_win_dict[col] = common_win
            elif not col in col_win_dict:
                result_col_win_dict[col] = 1

        self.allMetrics.add(allMetrics_tag + tag_suffix, meanDfColumns(to_be_rolled, result_col_win_dict))
        return allMetrics_tag + tag_suffix

       
    def drawGraphTwinx(self, metric_version_dict=None, allMetrics_tag=None, figsize=None, begin=None, end=None, prices_key=None,
        vlines=[], prices_log=True, final=True):
        dict_to_draw = {}
        if not metric_version_dict is None:
            for metric in metric_version_dict:
                version = metric_version_dict[metric]
                metric_and_version = metric + ('_' if version != '' else '') + version
                dict_to_draw[metric_and_version] = self.d[metric].v[version][begin:end]
            times = self.times[begin:end]
        else:
            allMetrics_df = self.allMetrics.v[allMetrics_tag]
            for metric in allMetrics_df.columns:
                if metric != 'times':
                    dict_to_draw[metric] = allMetrics_df[metric][begin:end]
            times = allMetrics_df['times'][begin:end]

        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)


        subplots_count =  len(dict_to_draw) + (0 if prices_key is None else 1)    
        if figsize is None:
            figsize = (20, 7)
        #plt.locator_params(axis='x', nbins=10)
        f, host = plt.subplots()
        f.set_size_inches(figsize)
        #axes = f.subplots(subplots_count, 1, sharex=True, squeeze=False)
        axes = [host.twinx() for i in range(subplots_count)]
        colors = ['b','g','c','y','m','r', np.array((165,42,42))/256]
        if prices_key is None:
            colors = colors[3:]
        #f.locator_params(axis='x', nbins=10)
        tkw = dict(size=4, width=1.5)


        ind = 0
        for metric in dict_to_draw:

            is_host = False
            if not (prices_key is None and ind == subplots_count - 1):
                axis = axes[ind]

                axis.spines["right"].set_position(("axes", 1. + 0.04*ind))
                make_patch_spines_invisible(axis)
                axis.spines["right"].set_visible(True)
            else:
                axis = host
                is_host = True

            if final:
                label = FOR_CHARTS[metric][0]
            else:
                label = metric

            pl, = axis.plot(times, dict_to_draw[metric], color=colors[ind], label=label)

            #axis.set_title(FOR_CHARTS[metric][0])
            #axis.set_ylabel(FOR_CHARTS[metric][1])

            for x in vlines:
                if begin is None or begin < x:
                    if end is None or x < end:
                        axis.axvline(x=self.times[x], color='red')
            ind += 1

            axis.set_ylabel(label)
            axis.yaxis.label.set_color(pl.get_color())
            axis.tick_params(axis='y', colors=pl.get_color(), **tkw)

        if not prices_key is None:
            #axes[-1, 0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))

            prices_df_00 = self.prices[prices_key]
            prices_df_0 = prices_df_00[pd.to_datetime(prices_df_00['times']) >= self.times[begin:][0]]
            prices_df =   prices_df_0 [pd.to_datetime(prices_df_0 ['times']) <= self.times[:end][-1]]

            if prices_log:
                prices_to_draw = np.log(prices_df['prices'])
            else:
                prices_to_draw = prices_df['prices']

            pl, = host.plot(pd.to_datetime(prices_df['times']), prices_to_draw, color='k', label=label)
            if final:
                print('kutak!')
                label = 'лог. рыночной цены (в USD)'
                host.set_ylabel(label)
                host.yaxis.label.set_color(pl.get_color())
                host.tick_params(axis='y', colors=pl.get_color(), **tkw)
            else:
                axes[-1].set_title('prices from {}'.format(prices_key))



            #axes[-1].spines["right"].set_position(("axes", 1. + 0.2*i))
            #make_patch_spines_invisible(axes[-1])
            #axes[-1].spines["right"].set_visible(True)

            # brown color np.array((165,42,42))/256.
            for x in vlines:
                if begin is None or begin < x:
                    if end is None or x < end:
                        host.axvline(x=self.times[x], color='red')

        f.legend()
        #axes[4].plot(pd.to_datetime(prices_half_b_df['snapped_at'])[b:e], prices_half_b_df['price'][b:e])
        #axes[4].set_title("Price (USD)")
        return f


    def drawGraph(self, metric_version_dict=None, allMetrics_tag=None, figsize=None, begin=None, end=None, prices_key=None,
        vlines=[], prices_log=True, final=True):
        dict_to_draw = {}
        if not metric_version_dict is None:
            for metric in metric_version_dict:
                version = metric_version_dict[metric]
                metric_and_version = metric + ('_' if version != '' else '') + version
                dict_to_draw[metric_and_version] = self.d[metric].v[version][begin:end]
            times = self.times[begin:end]
        else:
            allMetrics_df = self.allMetrics.v[allMetrics_tag]
            for metric in allMetrics_df.columns:
                if metric != 'times':
                    dict_to_draw[metric] = allMetrics_df[metric][begin:end]
            times = allMetrics_df['times'][begin:end]

        subplots_count =  len(dict_to_draw) + (0 if prices_key is None else 1)    
        if figsize is None:
            figsize = (20, 6 * subplots_count)
        #plt.locator_params(axis='x', nbins=10)
        f = plt.figure(figsize=figsize)
        axes = f.subplots(subplots_count, 1, sharex=True, squeeze=False)
        #f.locator_params(axis='x', nbins=10)
        ind = 0
        for metric in dict_to_draw:
            axes[ind, 0].plot(times, dict_to_draw[metric])
            if final:
                axes[ind, 0].set_title(FOR_CHARTS[metric][0])
                axes[ind, 0].set_ylabel(FOR_CHARTS[metric][1])
            else:
                axes[ind, 0].set_title(metric)
            for x in vlines:
                if begin is None or begin < x:
                    if end is None or x < end:
                        axes[ind, 0].axvline(x=self.times[x], color='red')
            ind += 1
        if not prices_key is None:
            #axes[-1, 0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(10))

            prices_df_00 = self.prices[prices_key]
            prices_df_0 = prices_df_00[pd.to_datetime(prices_df_00['times']) >= self.times[begin:][0]]
            prices_df =   prices_df_0 [pd.to_datetime(prices_df_0 ['times']) <= self.times[:end][-1]]

            if prices_log:
                prices_to_draw = np.log(prices_df['prices'])
            else:
                prices_to_draw = prices_df['prices']
            axes[-1, 0].plot(pd.to_datetime(prices_df['times']), prices_to_draw)
            if final:
                print('kutak!')
                axes[-1, 0].set_title('лог. рыночной цены (в USD)')
                axes[-1, 0].set_ylabel('')
            else:
                axes[-1, 0].set_title('prices from {}'.format(prices_key))
            for x in vlines:
                if begin is None or begin < x:
                    if end is None or x < end:
                        axes[-1, 0].axvline(x=self.times[x], color='red')


        #axes[4].plot(pd.to_datetime(prices_half_b_df['snapped_at'])[b:e], prices_half_b_df['price'][b:e])
        #axes[4].set_title("Price (USD)")
        return f