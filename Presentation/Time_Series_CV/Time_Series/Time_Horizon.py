import pandas as pd
import numpy as np


class Time_Horizon(object):
    """description of class"""
    def __init__(self, a):
        self.x = a


    def rolling_horizon(self,mdl, data, wsize=4 , startInd=0, regress = True, fixed = True):
        error2 = []
        prdList = []
        coefMatrix = []
        for i in range(startInd,len(data)):
            rlg = mdl
            if fixed:
                trainx = data[i:i + wsize].drop('change_in_spot', axis = 1)
                trainy = data[i:i + wsize].change_in_spot
            else:
                trainx = data[startInd:i + wsize].drop('change_in_spot', axis = 1)
                trainy = data[startInd:i + wsize].change_in_spot
            rlg.fit(trainx,trainy)
            testx = data[i + wsize:i + wsize + 1].drop('change_in_spot', axis = 1).copy()
            testy = data[i + wsize:i + wsize + 1].change_in_spot.values.copy()
            prd = rlg.predict(testx)
        
            if regress:
                error2.append(((testy[0] - prd[0]) ** 2))
                prdList.append(prd[0])
            else:
                testy[0] = percentile_transform_three_bin(data['change_in_spot'],testy[0])
                prdList.append(percentile_transform_three_bin(trainy,prd[0]))
                prd = percentile_transform_three_bin(trainy,prd[0])
                if testy[0] - prd == 0:
                    error2.append(0)
                else:
                    error2.append(1)
            if i + wsize + 1 == len(data):
                break
        try:
            assert len(data) == wsize + len(error2) + startInd
            assert len(error2) == len(data) - (startInd + wsize)
        except:
            print('confirmation error, different length')
        return error2, prdList


    def rolling_cv (self, data, mdl,windowList, regress = True, fixed = True):
        rmse = {}
        wsize = 0
        for w in windowList:
            error2, prd = rolling_horizon(mdl = mdl, data = data, wsize = w, startInd = 0,regress = regress, fixed = fixed)
            rmse[w] = np.sqrt(mean(np.cumsum(error2)))
        wsize = min(rmse, key = rmse.get)
        se, prdList = rolling_horizon(mdl = mdl, data = data, wsize= wsize, startInd = 0 , regress = regress, fixed = fixed )
        rmse = np.sqrt(mean(np.cumsum(se)))
        return se, rmse, wsize, prdList

    def add(self,a,b):
        return a+b


