from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from operator import add



class rolling_regression():
    """description of class"""
    def __init__(self, data):
        self.data = data


    def rolling_horizon(data, wsize=4 , startInd=0):
        error2 = []
        for i in range(startInd,len(data)):
            rlg = LinearRegression()
            trainx = data[i:i + wsize].drop('change_in_spot', axis = 1)
            trainy = data[i:i + wsize].change_in_spot
            rlg.fit(trainx,trainy)
            testx = data[i + wsize:i + wsize + 1].drop('change_in_spot', axis = 1)
            testy = data[i + wsize:i + wsize + 1].change_in_spot
            prd = rlg.predict(testx)
            error2.append(((testy - prd) ** 2))
            if i + wsize + 1 == len(data):
                break
        assert len(data) == wsize + len(error2) + startInd
        assert len(error2) == len(data) - (startInd + wsize)
        return error2



    def rolling_regression(data, fold=5, windowList=[4,6,12]):
        tss = TimeSeriesSplit(fold)
        ts_Split = tss.split(data)
        Ind = [[k,j] for k,j in ts_Split]
        lrg = LinearRegression()
        rmsedict = {}
        for j in windowList:
            rmseli = []
            for i in Ind:
                train = data.iloc[i[0]]
                test = data.iloc[i[1]]
                train = train.append(test)
                crmse, rli, r2 = rolling_horizon(train,wsize = j , startInd = 0)
                rmseli.append(crmse)
            rmsedict[j] = np.mean(rmseli)
        rolling_num = min(rmsedict, key = rmsedict.get)
        crsm = rmsedict[k]
        return rolling_num, crsm



