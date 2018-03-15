
import pandas as pd
import numpy as np
from sklearn.externals.joblib import Parallel, delayed

class rolling_Horizon(object):
    """description of class"""

    def __init__(self, mdl, data, wsize=4 , startInd=0, regress = True, fixed = True):
        self.error2 = []
        self.prdList = []
        self.wsize = wsize
        self.startInd = startInd
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
                self.error2.append(((testy[0] - prd[0]) ** 2))
                self.prdList.append(prd[0])
            else:
                testy[0] = percentile_transform_three_bin(data['change_in_spot'],testy[0])
                self.prdList.append(percentile_transform_three_bin(trainy,prd[0]))
                prd = percentile_transform_three_bin(trainy,prd[0])
                if testy[0] - prd == 0:
                    self.error2.append(0)
                else:
                    self.error2.append(1)
            if i + wsize + 1 == len(data):
                break
        assert len(data) == self.wsize + len(self.error2) + self.startInd
        assert len(self.error2) == len(data) - (self.startInd + self.wsize)


 


class rolling_cv(object):

    from Time_Series.CrossValidation import rolling_Horizon


    def __init__(self, data, mdl,windowList, regress = True, fixed = True):
        rmse = {}
        wsize = 0
        for w in windowList:
             rh = rolling_Horizon(mdl = mdl, data = data, wsize = w, startInd = 0,regress = regress, fixed = fixed)
             error2 = rh.error2
             prd = rh.prdList
             rmse[w] = np.sqrt(np.mean(np.cumsum(error2)))
        wsize = min(rmse, key = rmse.get)
        rh = rolling_Horizon(mdl = mdl, data = data, wsize= wsize, startInd = 0 , regress = regress, fixed = fixed )
        se = rh.error2
        prdList = rh.prdList
        rmse = np.sqrt(np.mean(np.cumsum(se)))
        self.rmse = rmse
        self.bestWindow = wsize
        self.performance = rmse
        self.prdList = prdList
        self.error2 = se
        self.windowList = windowList
        self.isRegress = regress
        self.isFixed = fixed



class grid_tune_parameter (object):
    from Time_Series.CrossValidation import rolling_cv
    
    def __init__(self, mdl, data, window, paramList , paramName, regress = True, fixed = True):
        tuneSelection = pd.DataFrame(columns = ['param','window','rmse'])
        sse = {}
        prdList = {}
        for i in paramList:
            sizeselect = {}
            setattr(mdl,paramName,i)
            rc = rolling_cv(data, mdl,window, regress,fixed = fixed)
            se = rc.error2
            rmse = rc.rmse 
            wsize = rc.bestWindow 
            prdList[i] = rc.prdList
            tuneSelection = tuneSelection.append({'param':i,'window':wsize,'rmse':rmse},ignore_index=True)
            sse[i] = se
        self.tuned = tuneSelection.iloc[tuneSelection.rmse.idxmin()]
        self.para = self.tuned.param
        self.wsize = self.tuned.window
        self.error2 = sse[self.tuned.param]
        self.prdList = prdList[self.tuned.param]

class sequential_grid_tune (object):
    from Time_Series.CrossValidation import grid_tune_parameter

    def __init__(self, data, mdl, window , paramList, paramName , startPara = 0 , regress = True, fixed = True):

        windowSelect = grid_tune_parameter(mdl,data,window,[startPara],paramName,regress, fixed = fixed)
        size = [int(windowSelect.wsize)]
        paraSelect = grid_tune_parameter(mdl,data,size,paramList,paramName, regress, fixed = fixed)
        self.tuned = paraSelect.tuned
        self.para = paraSelect.para
        self.wsize = paraSelect.wsize
        self.error2 = paraSelect.error2
        self.prdList = paraSelect.prdList


class paralell_processing (object):
    from Time_Series.CrossValidation import sequential_grid_tune
    from sklearn.externals.joblib import Parallel, delayed
    def __init__(self, mdl, data ,windowList, paramList, paraName,colName ,regress = True, fixed = True):
        errorList = {}
        wisize = {}
        prdList= {}
        report = Parallel(n_jobs = -1, verbose = 50, backend = 'threading')(delayed(self.paralell_support)(mdl,data[i],regress,windowList,paramList, paraName, fixed) for i in colName)
        for i in colName:
            errorList[i] = report[colName.index(i)]['el']
            wisize[i] =report[colName.index(i)]['tune']
            prdList[i] = report[colName.index(i)]['prd']
        self.errorList = errorList
        self.wisize = wisize
        self.prdList = prdList


        
    def paralell_support (self,mdl, data , regress , windowList, paramList, paraName, fixed):
        
        tune_res = pd.DataFrame()
        el = []
        mdl = mdl
        sq = sequential_grid_tune(data,mdl, window = windowList, paramList = paramList, paramName = paraName, startPara = 0, regress = regress, fixed = fixed)
        se = sq.error2
        tuned = sq.tuned
        para = sq.para
        wsize = sq.wsize
        prdList = sq.prdList
        tune_res = tune_res.append({'Window_size': wsize, 'Currency': data.keys(), 'para': para},ignore_index= True)
        el= se
        return {'tune': tune_res, 'el':el, 'prd':prdList}





