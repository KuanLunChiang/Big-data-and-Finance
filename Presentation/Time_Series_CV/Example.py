from Time_Series import CrossValidation as tcv
from Time_Series import Report as trpt
import pandas as pd
import numpy as np

####################### create variables ################################################## 

def variable_cons (cpi, eurodep, fxspot, indprod, moneysupply, monStr = 'AUD'):
    data = pd.DataFrame(columns = ['change_in_spot','int_diff','inf_diff','ip_diff','ms_diff', 'ppp', 'moneyrtl'])
    data.change_in_spot = np.log(fxspot[monStr]) - np.log(fxspot[monStr].shift(1))
    data.int_diff = eurodep[monStr] - eurodep.USD
    data.inf_diff = (np.log(cpi[monStr]) - np.log(cpi[monStr].shift(1)).apply(lambda x: 0 if pd.isnull(x) else x) )- (np.log(cpi.USD) - np.log(cpi.USD.shift(1)).apply(lambda x: 0 if pd.isnull(x) else x))
    data.ip_diff = np.log(indprod[monStr]) - np.log(indprod.USD)
    data.ms_diff = np.log(moneysupply[monStr]) - np.log(moneysupply.USD)
    data.moneyrtl = (np.log(moneysupply.USD) - np.log(moneysupply[monStr])) - (np.log(indprod.USD) - np.log(indprod[monStr]) - fxspot[monStr])
    data.ppp = np.log(cpi.USD) -np.log(cpi[monStr]) - fxspot[monStr]
    data = data.dropna()
    return data


cpi = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\CPI.csv')
eurodep = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\EuroDep.csv')
fxspot = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\FX_Spot.csv')
indprod = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\IndProduction.txt', sep = '\t')
moneysupply = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\MoneySupply.csv')
impx = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\df1.csv')


cpi.Date = cpi.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
eurodep.Date = eurodep.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
fxspot.Date = fxspot.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
indprod.Date = indprod.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
moneysupply.Date = moneysupply.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))

_colName = fxspot.drop('Date',axis = 1).columns.tolist()
_coefMatrix = []
_dataDict = {}
_ppp = {}
_moneyrtl = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        _dataDict[i] = (variable_cons(cpi, eurodep, fxspot, indprod, moneysupply, monStr = i))
        _ppp[i] = _dataDict[i][['change_in_spot','ppp']].copy()
        _moneyrtl[i] = _dataDict[i][['change_in_spot','moneyrtl']].copy()
        _dataDict[i] = _dataDict[i].drop('ppp',axis = 1).drop('moneyrtl', axis = 1)

_responseVar = 'change_in_spot' 
###################### LASSO ####################################
from sklearn.linear_model import Lasso

mdl = Lasso(normalize = True)
widnowList = [1,2,3,4,5]
parameter = [1,2,3,4,5]
tuned_mdl = tcv.paralell_processing(mdl,_dataDict, _responseVar, widnowList, parameter,'alpha',_colName)
bench_mdl = {}
for i in _colName:
    bch = tcv.benchMark()
    bench_mdl[i] = bch.historical_mean(data = _dataDict[i].change_in_spot, wsize = int(tuned_mdl.wisize[i].Window_size))

sse_report = {}
for i in _colName:
    sse_report[i] = trpt.cum_sse_report(tuned_mdl.errorList[i], bench_mdl[i]).reportDF

trpt.plot_differential_report(_colName,sse_report,'SSEDif',3,3,"SSE Report")
trpt.plot_differential_report(_colName,sse_report,'oosrsquare',3,3,"OOSR Report")


######################## Random Forest ##########################################
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators = 10)
widnowList = [1]
parameter = [1]
tuned_mdl = tcv.paralell_processing(mdl,_dataDict, widnowList, [2,3,4],'max_features',_colName,True,True,False,-4,50)

bench_mdl = {}
for i in _colName:
    bch = tcv.benchMark()
    bench_mdl[i] = bch.historical_mean(data = _dataDict[i].change_in_spot, wsize = int(tuned_mdl.wisize[i].Window_size))

sse_report = {}
for i in _colName:
    sse_report[i] = trpt.cum_sse_report(tuned_mdl.errorList[i], bench_mdl[i]).reportDF

trpt.plot_differential_report(_colName,sse_report,'SSEDif',3,3,"SSE Report")
trpt.plot_differential_report(_colName,sse_report,'oosrsquare',3,3,"OOSR Report")

########################## SVM ###########################################
from sklearn.svm import SVR
mdl = SVR(cache_size = 1000)
widnowList = np.arange(1,20)
parameter = np.arange(1,50)
tuned_mdl = tcv.paralell_processing(mdl,_dataDict, widnowList, [2,3,4],'max_features',_colName,True,True,False,-4,50)

bench_mdl = {}
for i in _colName:
    bch = tcv.benchMark()
    bench_mdl[i] = bch.historical_mean(data = _dataDict[i].change_in_spot, wsize = int(tuned_mdl.wisize[i].Window_size))

sse_report = {}
for i in _colName:
    sse_report[i] = trpt.cum_sse_report(tuned_mdl.errorList[i], bench_mdl[i]).reportDF

trpt.plot_differential_report(_colName,sse_report,'SSEDif',3,3,"SSE Report")
trpt.plot_differential_report(_colName,sse_report,'oosrsquare',3,3,"OOSR Report")



########################## SVM Classification #######################################
from sklearn.svm import SVR
mdl = SVR(cache_size = 5000)
widnowList = np.arange(1,20)
parameter = np.arange(1,50)
tuned_mdl = tcv.paralell_processing(mdl,_dataDict, widnowList, [2,3,4],'max_features',_colName,False,True,False,-4,50)

bench_mdl = {}
for i in _colName:
    bch = tcv.benchMark()
    bench_mdl[i] = bch.historical_mean(data = _dataDict[i].change_in_spot, wsize = int(tuned_mdl.wisize[i].Window_size))

sse_report = {}
for i in _colName:
    sse_report[i] = trpt.cum_sse_report(tuned_mdl.errorList[i], bench_mdl[i]).reportDF

trpt.plot_differential_report(_colName,sse_report,'SSEDif',3,3,"SSE Report")
trpt.plot_differential_report(_colName,sse_report,'oosrsquare',3,3,"OOSR Report")
