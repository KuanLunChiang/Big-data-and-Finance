import numpy as np
import pandas as pd
from operator import add
from sklearn.externals.joblib import Parallel, delayed
import timeit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import load_workbook



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

colName_ = fxspot.drop('Date',axis = 1).columns.tolist()
coefMatrix = []


############################ Function #######################################################################################
def new_excel (path):
    emt = pd.DataFrame()
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    emt.to_excel(writer,'new sheet')
    writer.save()
    writer.close()
    return print('new sheet created')

def excel_writer (path, name, data):

    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine = 'openpyxl')
    writer.book = book
    data.to_excel(writer, sheet_name = name)
    writer.save()
    writer.close()
    return print('Done')

def percentile_transform_three_bin (data, testd, lb = 0.33, ub = 0.66):
    res = 0
    tempdata = data.copy()
    lower = tempdata.quantile(lb)
    upper = tempdata.quantile (ub)
    if testd < lower:
        res = 1
    elif testd > upper:
        res = 3
    else:
        res = 2
    return res

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

def rolling_horizon(mdl, data, wsize=4 , startInd=0, regress = True, fixed = True):
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

def rolling_cv (data, mdl,windowList = [4,6,12], regress = True, fixed = True):
    rmse = {}
    wsize = 0
    for w in windowList:
        error2, prd = rolling_horizon(mdl = mdl, data = data, wsize = w, startInd = 0,regress = regress, fixed = fixed)
        rmse[w] = np.sqrt(mean(np.cumsum(error2)))
    wsize = min(rmse, key = rmse.get)
    se, prdList = rolling_horizon(mdl = mdl, data = data, wsize= wsize, startInd = 0 , regress = regress, fixed = fixed )
    rmse = np.sqrt(mean(np.cumsum(se)))
    return se, rmse, wsize, prdList

def historical_mean (data, wsize):
    prd = data.rolling(wsize).mean().dropna()
    prd.drop(prd.tail(1).index, inplace = True)
    prd.index = prd.index+1
    testy = data[wsize:,]
    error = prd.subtract(testy)
    se = error.apply(lambda x: pow(x,2)).tolist()
    try:
        assert len(prd) == len(testy)
    except:
        print('different length')
    return se

def cum_sse_report (regError, meanError):
    resDf = pd.DataFrame({'RegError': regError, 'MeanError': meanError})
    resDf['cum_reg_error'] = resDf.RegError.cumsum()
    resDf['cum_mean_error'] = resDf.MeanError.cumsum()
    resDf['SSEDif'] =  resDf.cum_mean_error - resDf.cum_reg_error
    resDf['oosrsquare'] = 1- (resDf.cum_reg_error / resDf.cum_mean_error)
    return resDf

def plot_differential_report (colName, data, para ,row = 3, col = 3, supT = 'Sum of Square Error Differential'):
    pltind = row*100+col*10+1
    plt.figure(figsize=(15,15))
    for i in colName:
        plt.subplot(pltind)
        data[i][para].plot()
        plt.title(i)
        plt.suptitle(supT, fontsize = 15, y = 0.92)
        pltind +=1
    return plt.show()

def grid_tune_parameter (mdl, data, window = [50,60,80,90, 100,120], paramList = [0.1,0.0001,5,7,8], paramName = 'alpha', regress = True, fixed = True):
    tuneSelection = pd.DataFrame(columns = ['param','window','rmse'])
    sse = {}
    for i in paramList:
        sizeselect = {}
        setattr(mdl,paramName,i)
        se, rmse, wsize, prdList = rolling_cv(data, mdl,window, regress,fixed = fixed)
        tuneSelection = tuneSelection.append({'param':i,'window':wsize,'rmse':rmse},ignore_index=True)
        sse[i] = se
    tuned = tuneSelection.iloc[tuneSelection.rmse.idxmin()]
    para = tuned.param
    wsize = tuned.window
    se = sse[tuned.param]

    return se, rmse, wsize, para, prdList

def sequential_grid_tune (data, mdl, window , paramList, paramName , startPara = 0 , regress = True, fixed = True):

    se, rmse, wsize, para, prdList = grid_tune_parameter(mdl,data,window,[startPara],paramName,regress, fixed = fixed)
    se, rmse, wsize, para, prdList = grid_tune_parameter(mdl,data,[int(wsize)],paramList,paramName, regress, fixed = fixed)

    return se, rmse, wsize, para, prdList

def classification_benchmark (data):
    upper = data.quantile(0.66)
    lower = data.quantile(0.33)
    datay = data.copy()
    res = data.shift(1).dropna().apply(lambda x: 1 if x < lower else (3 if x > upper else 2))
    datay = datay.apply(lambda x: 1 if x < lower else (3 if x > upper else 2))
    datay = datay[1:]
    se = datay.subtract(res).apply(lambda x: 0 if x == 0 else 1)
    se = list(se)
    return se


def results_report (mdl, data , isTuned , isClassif, func, arg ):
    winSize_currency = pd.DataFrame(columns= ['Window_size', 'Currency' ,'RMSE','para'])
    error2_list = {}
    prd_List = {}
    for i in fxspot.columns.tolist():
        if i !='Date':
            mdl = mdl
            if isTuned:
                se, rmse, wsize, para, prdList = func(data[i],*arg)
                winSize_currency = winSize_currency.append({'Window_size': wsize, 'Currency': i, 'RMSE': rmse, 'para': para},ignore_index= True)
            else:
                se, rmse, wsize, prdList = func(data[i],*arg)
                winSize_currency = winSize_currency.append({'Window_size': wsize, 'Currency': i, 'RMSE': rmse, 'para': None},ignore_index= True)
            error2_list[i] = se
            prd_List[i] = prdList

    mean_se = {}
    if isClassif:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = winSize_currency.loc[winSize_currency.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = classification_benchmark(dataDict[i].change_in_spot[int(ws)-1:])
    else:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = winSize_currency.loc[winSize_currency.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = historical_mean(dataDict[i].change_in_spot,int(ws))

    SSED = {}
    for i in fxspot.columns.tolist():
        if i !='Date':
            regError = error2_list[i]
            meanError = mean_se[i]
            SSED[i] = cum_sse_report (regError, meanError)
    return SSED, winSize_currency, prd_List

###################################### LASSO Parallel Computing ###############################################################3
def parallel_LASSO (name, regress , windowList, paramList, fixed):
    i = name
    LASSO_currency = pd.DataFrame()
    el = []
    lasso = Lasso(normalize = True)
    se, rmse, wsize, para, prdList = sequential_grid_tune( dataDict[i],lasso, window = windowList, paramList = paramList, paramName = 'alpha', startPara = 0, regress = regress, fixed = fixed)
    LASSO_currency = LASSO_currency.append({'Window_size': wsize, 'Currency': i, 'Alpha': para},ignore_index= True)
    el= se
    return {'lasso': LASSO_currency, 'el':el, 'prd':prdList}

def multi_proccess_LASSO(windowList, paramList, regress = True, fiexed = True):
    errorList_lasso = {}
    lasso_wisize = {}
    prdList= {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    lasso_report = Parallel(n_jobs = -4, verbose = 50, backend = 'threading')(delayed(parallel_LASSO)(i,regress,windowList,paramList, fiexed) for i in col)
    for i in colName:
        errorList_lasso[i] = lasso_report[colName.index(i)]['el']
        lasso_wisize[i] =lasso_report[colName.index(i)]['lasso']
        prdList[i] = lasso_report[colName.index(i)]['prd']
    return errorList_lasso, lasso_wisize, prdList

def multi_proccess_LASSO_report (errorList_lasso, lasso_wisize, isClassif = False):
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    lasso_tuned = pd.DataFrame()
    for i in range(len(colName)):
        lasso_tuned = lasso_tuned.append(lasso_wisize[colName[i]])
    lasso_tuned= lasso_tuned.reset_index().drop('index',axis = 1)

    mean_se = {}
    if isClassif:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = lasso_tuned.loc[lasso_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = classification_benchmark(dataDict[i].change_in_spot[int(ws)-1:])
    else:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = lasso_tuned.loc[lasso_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = historical_mean(dataDict[i].change_in_spot, int(ws))

    SSED_lasso = {}
    for i in fxspot.columns.tolist():
        if i !='Date':
            regError = errorList_lasso[i]
            meanError = mean_se[i]
            SSED_lasso[i] = cum_sse_report (regError, meanError)
    return SSED_lasso, lasso_tuned

def parallel_svm (name, regress , windowList, paramList, fixed = True):
    i = name
    svm_currency = pd.DataFrame()
    el = []
    svm = SVR()
    se, rmse, wsize, para, prdList = sequential_grid_tune( dataDict[i],svm, window = windowList, paramList = paramList, paramName = 'C', startPara = 0.01, regress = regress)
    svm_currency = svm_currency.append({'Window_size': wsize, 'Currency': i, 'C': para},ignore_index= True)
    el= se
    return {'svm': svm_currency, 'el':el, 'prd': prdList}

def multi_proccess_svm(windowList, paramList, regress = True, fixed = True):
    errorList_svm = {}
    svm_wisize = {}
    prd_list = {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    svm_report = Parallel(n_jobs = -4, verbose = 50, backend = 'threading')(delayed(parallel_svm)(i,regress,windowList,paramList, fixed) for i in col)
    for i in colName:
        errorList_svm[i] = svm_report[colName.index(i)]['el']
        svm_wisize[i] =svm_report[colName.index(i)]['svm']
        prd_list[i]= svm_report[colName.index(i)]['prd']
    return errorList_svm, svm_wisize, prd_list

def multi_proccess_svm_report (errorList_svm, svm_wisize, isClassif = False):
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    svm_tuned = pd.DataFrame()
    for i in range(len(colName)):
        svm_tuned = svm_tuned.append(svm_wisize[colName[i]])
    svm_tuned= svm_tuned.reset_index().drop('index',axis = 1)

    mean_se = {}
    if isClassif:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = svm_tuned.loc[svm_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = classification_benchmark(dataDict[i].change_in_spot[int(ws)-1:])
    else:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = svm_tuned.loc[svm_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = historical_mean(dataDict[i].change_in_spot, int(ws))

    SSED_svm = {}
    for i in fxspot.columns.tolist():
        if i !='Date':
            regError = errorList_svm[i]
            meanError = mean_se[i]
            SSED_svm[i] = cum_sse_report (regError, meanError)
    return SSED_svm, svm_tuned

def parallel_elastic (name, regress , windowList, paramList, fixed, data):
    i = name
    elastic_currency = pd.DataFrame()
    el = []
    elastic = ElasticNet(normalize = True)
    se, rmse, wsize, para, prdList = sequential_grid_tune( data[i],elastic, window = windowList, paramList = paramList, paramName = 'alpha', startPara = 0.01, regress = regress, fixed = fixed)
    elastic_currency = elastic_currency.append({'Window_size': wsize, 'Currency': i, 'Alpha': para},ignore_index= True)
    el= se
    return {'elastic': elastic_currency, 'el':el, 'prd':prdList}

def multi_proccess_elastic(windowList, paramList, data ,regress = True, fiexed = True):
    errorList_elastic = {}
    elastic_wisize = {}
    prdList= {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    elastic_report = Parallel(n_jobs = -4, verbose = 50, backend = 'threading')(delayed(parallel_elastic)(i,regress,windowList,paramList, fiexed, data) for i in col)
    for i in colName:
        errorList_elastic[i] = elastic_report[colName.index(i)]['el']
        elastic_wisize[i] =elastic_report[colName.index(i)]['elastic']
        prdList[i] = elastic_report[colName.index(i)]['prd']
    return errorList_elastic, elastic_wisize, prdList

def multi_proccess_elastic_report (errorList_elastic, elastic_wisize, isClassif = False):
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    elastic_tuned = pd.DataFrame()
    for i in range(len(colName)):
        elastic_tuned = elastic_tuned.append(elastic_wisize[colName[i]])
    elastic_tuned= elastic_tuned.reset_index().drop('index',axis = 1)

    mean_se = {}
    if isClassif:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = elastic_tuned.loc[elastic_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = classification_benchmark(dataDict[i].change_in_spot[int(ws)-1:])
    else:
        for i in fxspot.columns.tolist():
            if i !='Date':
                ws = elastic_tuned.loc[elastic_tuned.Currency ==i].Window_size.tolist()[0]
                mean_se[i] = historical_mean(dataDict[i].change_in_spot, int(ws))

    SSED_elastic = {}
    for i in fxspot.columns.tolist():
        if i !='Date':
            regError = errorList_elastic[i]
            meanError = mean_se[i]
            SSED_elastic[i] = cum_sse_report (regError, meanError)
    return SSED_elastic, elastic_tuned




#################### Variable Construction ################################

dataDict = {}
ppp = {}
moneyrtl = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        dataDict[i] = (variable_cons(cpi, eurodep, fxspot, indprod, moneysupply, monStr = i))
        ppp[i] = dataDict[i][['change_in_spot','ppp']].copy()
        moneyrtl[i] = dataDict[i][['change_in_spot','moneyrtl']].copy()
        dataDict[i] = dataDict[i].drop('ppp',axis = 1).drop('moneyrtl', axis = 1)


########################## Regression ############################################

mdl = LinearRegression()
reg_SSED, reg_winSize_currency, reg_prd = results_report(mdl = mdl, data = dataDict,isTuned = False, isClassif = False, func = rolling_cv, arg = (mdl, np.arange(65,90)))
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = reg_SSED, para = 'SSEDif', row = 3, col = 3)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = reg_SSED, para = 'oosrsquare', row = 3, col = 3, supT = 'Out of Sample R-Squared')


############################## LASSO #########################################################
from sklearn.linear_model import Lasso
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\lasso.xlsx'

errorList_lasso, lasso_wisize, lasso_prd = multi_proccess_LASSO(windowList = np.arange(40,100,1), paramList = np.arange(0,0.0003,0.0001), regress = True)
SSED_lasso, lasso_tuned = multi_proccess_LASSO_report(errorList_lasso, lasso_wisize,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_lasso, para = 'SSEDif', row = 3, col = 3, supT = 'LASSO SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_lasso, para = 'oosrsquare', row = 3, col = 3, supT = 'LASSO OOS R-Squared')


new_excel(storepath)
[excel_writer(storepath,('SSED_'+i),SSED_lasso[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
[excel_writer(storepath,('SSED_'+i),SSED_lasso[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
excel_writer(storepath, ('Window_size'), lasso_tuned)

################################## Random Forest #####################################################
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\rf.xlsx'

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_SSED, rf_winSize_currency, rf_prd  = results_report( mdl = rf,data = dataDict,isTuned =True,isClassif = False ,func = sequential_grid_tune,arg = (rf, np.arange(79,80,1),np.arange(3,4), 'max_depth', 2, True))
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = rf_SSED, para = 'SSEDif', row = 3, col = 3, supT = 'Random Forest SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = rf_SSED, para = 'oosrsquare', row = 3, col = 3, supT = 'Random Forest OOS R-Squared')
[excel_writer(storepath,('SSED_'+i),rf_SSED[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
excel_writer(storepath, ('Window_size'), rf_winSize_currency)



########################################## LASSO Classification ####################################################################
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\lassoc.xlsx'

errorList_lassoc, lassoc_wisize, lassoc_prd = multi_proccess_LASSO(windowList = np.arange(60,61), paramList = np.arange(0.0001,0.0002,0.0001),regress = False)

SSED_lassoc, lassoc_tuned = multi_proccess_LASSO_report(errorList_lassoc, lassoc_wisize, isClassif = True)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_lassoc, para = 'SSEDif', row = 3, col = 3, supT = 'LASSO Classification SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_lassoc, para = 'oosrsquare', row = 3, col = 3, supT = 'LASSO Classification OOS R-Squared')

from collections import Counter
for i in colName_:
    print(i,Counter(lassoc_prd[i]))
lassoc_prd

################################## Random Forest Classification ###############################################################3
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\rfc.xlsx'

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 50)
rfc_SSED, rfc_winSize_currency = results_report(mdl = rf,data = dataDict,isTuned = True, isClassif = True, func = sequential_grid_tune,arg = (rf, np.arange(10,11,1),np.arange(2,4), 'max_depth', 2, False))
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = rfc_SSED, para = 'SSEDif', row = 3, col = 3, supT = 'Random Forest SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = rfc_SSED, para = 'oosrsquare', row = 3, col = 3, supT = 'Random Forest OOS R-Squared')
[excel_writer(storepath,('SSED_'+i),rfc_SSED[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
excel_writer(storepath, ('Window_size'), rfc_winSize_currency)

################################## SVM #############################################
from sklearn.svm import SVR
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\svm.xlsx'

svm = SVR()
errorList_svm, svm_wisize, svm_prd = multi_proccess_svm(windowList = np.arange(10,11,1), paramList = np.arange(0.01,0.02,0.01), regress = True)
SSED_svm, svm_tuned = multi_proccess_svm_report(errorList_svm, svm_wisize,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'SSEDif', row = 3, col = 3, supT = 'SVM SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'oosrsquare', row = 3, col = 3, supT = 'SVM OOS R-Squared')


################################### SVM Classification ######################################################
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\svmc.xlsx'
svmc = SVR()

errorList_svmc, svmc_wisize, svmc_prd = multi_proccess_svm(windowList = np.arange(54,55,1), paramList = np.arange(0.01,0.02,0.01), regress = False, fixed = False)
SSED_svmc, svmc_tuned = multi_proccess_svm_report(errorList_svmc, svmc_wisize,isClassif = True)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'SSEDif', row = 3, col = 3, supT = 'SVM SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'oosrsquare', row = 3, col = 3, supT = 'SVM OOS R-Squared')



########################################## Elastic Net ######################################################
from sklearn.linear_model import ElasticNet
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\elasticppp.csv'

errorList_elastic, elastic_wisize, elastic_prd = multi_proccess_elastic(windowList = np.arange(60,80,1), paramList = np.arange(0.001,0.01,0.001), regress = True)
SSED_elastic, elastic_tuned = multi_proccess_elastic_report(errorList_elastic, elastic_wisize,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elastic, para = 'SSEDif', row = 3, col = 3, supT = 'elastic SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elastic, para = 'oosrsquare', row = 3, col = 3, supT = 'elastic OOS R-Squared')


errorList_elasticpp, elastic_wisizepp, elastic_prdpp = multi_proccess_elastic(windowList = np.arange(60,80,1), paramList = np.arange(0.001,0.01,0.001),data = ppp ,regress = True)
SSED_elasticpp, elastic_tunedpp = multi_proccess_elastic_report(errorList_elasticpp, elastic_wisizepp,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elasticpp, para = 'SSEDif', row = 3, col = 3, supT = 'PPP Elastic SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elasticpp, para = 'oosrsquare', row = 3, col = 3, supT = 'PPP Elastic OOS R-Squared')


errorList_elasticrtl, elastic_wisizertl, elastic_prdrtl = multi_proccess_elastic(windowList = np.arange(60,80,1), paramList = np.arange(0.001,0.01,0.001),data = moneyrtl ,regress = True)
SSED_elasticrtl, elastic_tunedrtl = multi_proccess_elastic_report(errorList_elasticrtl, elastic_wisizertl,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elasticrtl, para = 'SSEDif', row = 3, col = 3, supT = 'RTL Elastic SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elasticrtl, para = 'oosrsquare', row = 3, col = 3, supT = 'RTL Elastic OOS R-Squared')

errorList_elastic, elastic_wisize, elastic_prd = multi_proccess_elastic(windowList = np.arange(60,80,1), paramList = np.arange(0.00001,0.0001,0.00001), regress = False, fiexed = False, data = dataDict)
SSED_elastic, elastic_tuned = multi_proccess_elastic_report(errorList_elastic, elastic_wisize,isClassif = True)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elastic, para = 'SSEDif', row = 3, col = 3, supT = 'Elastic Net Classification SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_elastic, para = 'oosrsquare', row = 3, col = 3, supT = 'Elastic Net Classification OOS R-Squared')


newDt = dataDict['JPY'].copy()
newDt['imx'] = impx['JPY']
mdl = Lasso(normalize = True)
se, rmse, wsize, para, prdList = sequential_grid_tune(newDt.dropna().drop('imx', axis =3),mdl,np.arange(50,85),np.arange(0.00001,0.0001,0.00001),'alpha')
me = historical_mean(newDt.dropna().change_in_spot,int(wsize))
resDf = cum_sse_report(se,me)
resDf.SSEDif.plot()


newDt = dataDict['AUD'].copy()
newDt['imx'] = impx['AUD']
mdl = ElasticNet(normalize = True)
se, rmse, wsize, para, prdList = sequential_grid_tune(newDt.dropna(),mdl,np.arange(50,85),np.arange(0.00001,0.0001,0.00001),'alpha')
me = historical_mean(newDt.dropna().change_in_spot,int(wsize))
resDf = cum_sse_report(se,me)
plt.title('AUD')
plt.suptitle('Elastic Net (Terms of Trade)')
resDf.SSEDif.plot()

newDt = dataDict['SEK'].copy()
newDt['imx'] = impx['SEK']
mdl = ElasticNet(normalize = True)
se, rmse, wsize, para, prdList = sequential_grid_tune(newDt.dropna(),mdl,np.arange(50,85),np.arange(0.00001,0.0001,0.00001),'alpha')
me = historical_mean(newDt.dropna().change_in_spot,int(wsize))
resDf = cum_sse_report(se,me)
plt.title('SEK')
plt.suptitle('Elastic Net (Terms of Trade)')
resDf.SSEDif.plot()



plt.subplot(pltind)
data[i][para].plot()
plt.title(i)
plt.suptitle(supT, fontsize = 15, y = 0.92)



elastic_tuned.to_csv(storepath)

############################################### Replicate Results #################################################################
rf = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\Result\randomforest.csv')
las = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\Result\lasso.csv')
svm = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\Result\svm.csv')
els = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\Result\elastic.csv')


from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet

lso = Lasso(normalize = True)
rfmdl = RandomForestRegressor(n_estimators = 50)

def replicate_result(mdl, dt, data, paraName):
    errorDict = {}
    prdDict= {}
    winSize = {}
    for i in colName_:
        setattr(mdl,paraName,dt.loc[dt['Currency']== i].para.values[0] )
        error2, prdList = rolling_horizon(mdl,data[i],dt.loc[dt['Currency']== i].Window_size.values[0])
        errorDict[i] = error2
        prdDict[i] = prdList
        winSize[i] = dt.loc[dt['Currency']== i].Window_size.values[0]
    return errorDict, prdDict, winSize

def in_sample_error (mdl, data):
    error2 = {}
    for i in colName_:
        x = data[i].drop('change_in_spot',axis = 1)
        y = data[i].change_in_spot
        mdl.fit(x,y)
        prd = mdl.predict(x)
        error2[i] = [(y.tolist()[j] - prd[j])**2 for j in range(len(prd))]
    return error2

def SSED_rpt (fstError, se, inErr):
    ooEr = {}
    inEr = {}
    for i in colName_:
        regError = fstError[i]
        meanError = se[i]
        inError = inErr[i][len(inErr[i])-len(fstError[i]):]
        ooEr[i] = cum_sse_report (regError, meanError)
        inEr[i] = cum_sse_report(inError,meanError)
    return ooEr, inEr

def plot_report (outSam, inSam, para ,row = 3, col = 3, supT = 'Sum of Square Error Differential'):
    pltind = row*100+col*10+1
    plt.figure(figsize=(15,15))
    for i in colName_:
        ptd = pd.DataFrame(columns = ['In Sample','Out of Sample'])
        ptd['In Sample'] = inSam[i][para]
        ptd['Out of Sample'] = outSam[i][para]
        plt.subplot(pltind)
        ptd['In Sample'].plot(style = ['--'], legend = True)
        ptd['Out of Sample'].plot(legend = True)
        plt.title(i)
        plt.suptitle(supT, fontsize = 15, y = 0.92)
        pltind +=1
    return plt.show()


lasso_in_error = in_sample_error(lso, dataDict)
lasso_errorDict, lasso_prdDict, lasso_wind = replicate_result(lso,las,dataDict,'alpha')
lasso_se = {}
for i in colName_:
    lasso_se[i] = historical_mean(dataDict[i].change_in_spot,lasso_wind[i])

lassso_out_ssed, lasso_in_ssed = SSED_rpt(lasso_errorDict,lasso_se,lasso_in_error)
plot_report(lassso_out_ssed,lasso_in_ssed,'SSEDif',3,3,'LASSO: Cumulative SSE')
plot_report(lassso_out_ssed,lasso_in_ssed,'oosrsquare',3,3,'LASSO: Out of Sample R2')


rf_in_error = in_sample_error(rfmdl,dataDict)
rf_errorDict, rf_prdDict, rf_wind = replicate_result(rfmdl,rf,dataDict,'max_depth')
rf_se= {}
for i in colName_:
    rf_se[i] = historical_mean(dataDict[i].change_in_spot,rf_wind[i])
rf_out_ssed, rf_in_ssed = SSED_rpt(rf_errorDict,rf_se,rf_in_error)
plot_report(rf_out_ssed,rf_in_ssed,'SSEDif',3,3,'Random Forest: Cumulative SSE')
plot_report(rf_out_ssed,rf_in_ssed,'oosrsquare',3,3,'Random Forest: Out of Sample R2')

las_oor2 = {}
rf_oo2 = {}
svm_oo2 ={}
els_oo2 = {}


for i in colName_:
    las_oor2[i] = lassso_out_ssed[i].tail(1)['oosrsquare'].values[0]
    rf_oo2[i] = rf_out_ssed[i].tail(1)['oosrsquare'].values[0]
    svm_oo2[i] = svm_out_ssed[i].tail(1)['oosrsquare'].values[0]
    els_oo2[i] = els_out_ssed[i].tail(1)['oosrsquare'].values[0]

las_oor2['mdl'] = 'LASSO'
rf_oo2 ['mdl'] = 'Random Forest'
svm_oo2['mdl'] = 'SVM'
els_oo2 ['mdl'] = 'Elastic Net'

storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\report.csv'
rr2_report = pd.DataFrame()
rr2_report = rr2_report.append(las_oor2, ignore_index = True)
rr2_report = rr2_report.append(rf_oo2,ignore_index = True)
rr2_report = rr2_report.append(svm_oo2,ignore_index = True)
rr2_report = rr2_report.append(els_oo2,ignore_index = True)

rr2_report.to_csv(storepath)

svmMdl = SVR()
ndataDict = {}
for i in colName_:
    tempy = dataDict[i].change_in_spot
    ndataDict[i] = dataDict[i].apply(lambda x: (x-np.mean(x))/np.std(x), axis = 1)
    ndataDict[i].change_in_spot = tempy

svm_in_error = in_sample_error(svmMdl,ndataDict)
svm_errorDict, svm_prdDict, svm_wind = replicate_result(svmMdl,svm,ndataDict,'C')
svm_se= {}
for i in colName_:
    svm_se[i] = historical_mean(ndataDict[i].change_in_spot,svm_wind[i])
svm_out_ssed, svm_in_ssed = SSED_rpt(svm_errorDict,svm_se,svm_in_error)
plot_report(svm_out_ssed,svm_in_ssed,'SSEDif',3,3,'SVM: Cumulative SSE')
plot_report(svm_out_ssed,svm_in_ssed,'oosrsquare',3,3,'SVM: Out of Sample R2')

elsmdl = ElasticNet(normalize = True)
els_in_error = in_sample_error(elsmdl,dataDict)
els_errorDict, els_prdDict, els_wind = replicate_result(elsmdl,els,dataDict,'alpha')
els_se= {}
for i in colName_:
    els_se[i] = historical_mean(dataDict[i].change_in_spot,els_wind[i])
els_out_ssed, els_in_ssed = SSED_rpt(els_errorDict,els_se,els_in_error)
plot_report(els_out_ssed,els_in_ssed,'SSEDif',3,3,'Elastic Net (PPP): Cumulative SSE')
plot_report(els_out_ssed,els_in_ssed,'oosrsquare',3,3,'Elastic Net (PPP): Out of Sample R2')




elsmdl = ElasticNet(normalize = True)
els_in_error = in_sample_error(elsmdl,ppp)
els_out_ssed, els_in_ssed = SSED_rpt(els_errorDict,els_se,els_in_error)
plot_report(SSED_elasticpp,els_in_ssed,'SSEDif',3,3,'Elastic Net (PPP): Cumulative SSE')
plot_report(SSED_elasticpp,els_in_ssed,'oosrsquare',3,3,'Elastic Net (PPP): Out of Sample R2')


elsmdl = ElasticNet(normalize = True)
els_in_error = in_sample_error(elsmdl,moneyrtl)
els_out_ssed, els_in_ssed = SSED_rpt(els_errorDict,els_se,els_in_error)
plot_report(SSED_elasticrtl,els_in_ssed,'SSEDif',3,3,'Elastic Net (Monetary Fundamentals): Cumulative SSE')
plot_report(SSED_elasticrtl,els_in_ssed,'oosrsquare',3,3,'Elastic Net (Monetary Fundamentals): Out of Sample R2')






################################## Confusion Matrix ##########################################################
from sklearn.metrics import confusion_matrix

def confusionM (realData, prdData):
    realData = realData[len(realData) - len(prdData):len(realData)].change_in_spot
    return confusion_matrix(realData,prdData)

lassoc_conM = {}
for i in colName_:
    lassoc_conM[i] = confusionM(dataDict[i],lassoc_prd[i])
    print(i)
    print(lassoc_conM[i])



