
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
from numpy import mean

cpi = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\CPI.csv')
eurodep = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\EuroDep.csv')
fxspot = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\FX_Spot.csv')
indprod = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\IndProduction.txt', sep = '\t')
moneysupply = pd.read_csv(r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\MoneySupply.csv')

cpi.Date = cpi.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
eurodep.Date = eurodep.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
fxspot.Date = fxspot.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
indprod.Date = indprod.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
moneysupply.Date = moneysupply.Date.apply(lambda x: pd.to_datetime(x,format='%Y%m'))
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
    lower = data.quantile(lb)
    upper = data.quantile (ub)
    if testd < lower:
        res = 1
    elif testd > upper:
        res = 3
    else:
        res = 2
    return res

def variable_cons (cpi, eurodep, fxspot, indprod, moneysupply, monStr = 'AUD'):
    data = pd.DataFrame(columns = ['change_in_spot','int_diff','inf_diff','ip_diff','ms_diff'])
    data.change_in_spot = np.log(fxspot[monStr]) - np.log(fxspot[monStr].shift(1))
    data.int_diff = eurodep[monStr] - eurodep.USD
    data.inf_diff = (np.log(cpi[monStr]) - np.log(cpi[monStr].shift(1)).apply(lambda x: 0 if pd.isnull(x) else x) )- (np.log(cpi.USD) - np.log(cpi.USD.shift(1)).apply(lambda x: 0 if pd.isnull(x) else x))
    data.ip_diff = np.log(indprod[monStr]) - np.log(indprod.USD)
    data.ms_diff = np.log(moneysupply[monStr]) - np.log(moneysupply.USD)
    data = data.dropna()
    return data

def rolling_horizon(mdl, data, wsize=4 , startInd=0, regress = True):
    error2 = []
    prdList = []
    for i in range(startInd,len(data)):
        rlg = mdl
        trainx = data[i:i + wsize].drop('change_in_spot', axis = 1)
        trainy = data[i:i + wsize].change_in_spot
        rlg.fit(trainx,trainy)
        testx = data[i + wsize:i + wsize + 1].drop('change_in_spot', axis = 1)
        testy = data[i + wsize:i + wsize + 1].change_in_spot.values
        prd = rlg.predict(testx)
        prdList.append(prd[0])
        if regress:
            error2.append(((testy[0] - prd[0]) ** 2))
        else:
            testy[0] = percentile_transform_three_bin(data['change_in_spot'],testy)
            error2.append(int([1 if testy[0] == prd[0] else 0][0]))
        if i + wsize + 1 == len(data):
            break
    try:
        assert len(data) == wsize + len(error2) + startInd
        assert len(error2) == len(data) - (startInd + wsize)
    except:
        print('confirmation error, different length')
    return error2, prdList

def rolling_cv (data, mdl,windowList = [4,6,12], regress = True):
    rmse = {}
    wsize = 0
    for w in windowList:
        error2, prd = rolling_horizon(mdl = mdl, data = data, wsize = w, startInd = 0,regress = regress)
        rmse[w] = np.sqrt(mean(np.cumsum(error2)))
    wsize = min(rmse, key = rmse.get)
    se, prdList = rolling_horizon(mdl = mdl, data = data, wsize= wsize, startInd = 0 , regress = regress )
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

def grid_tune_parameter (mdl, data, window = [50,60,80,90, 100,120], paramList = [0.1,0.0001,5,7,8], paramName = 'alpha', regress = True):
    tuneSelection = pd.DataFrame(columns = ['param','window','rmse'])
    sse = {}
    for i in paramList:
        sizeselect = {}
        setattr(mdl,paramName,i)
        se, rmse, wsize, prdList = rolling_cv(data, mdl,window, regress)
        tuneSelection = tuneSelection.append({'param':i,'window':wsize,'rmse':rmse},ignore_index=True)
        sse[i] = se
    tuned = tuneSelection.iloc[tuneSelection.rmse.idxmin()]
    para = tuned.param
    wsize = tuned.window
    se = sse[tuned.param]

    return se, rmse, wsize, para, prdList

def sequential_grid_tune (data, mdl, window , paramList, paramName , startPara = 0 , regress = True):

    se, rmse, wsize, para, prdList = grid_tune_parameter(mdl,data,window,[startPara],paramName,regress)
    se, rmse, wsize, para, prdList = grid_tune_parameter(mdl,data,[int(wsize)],paramList,paramName, regress)

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
def parallel_LASSO (name, regress , windowList, paramList):
    i = name
    LASSO_currency = pd.DataFrame()
    el = []
    lasso = Lasso(normalize = True)
    se, rmse, wsize, para, prdList = sequential_grid_tune( dataDict[i],lasso, window = windowList, paramList = paramList, paramName = 'alpha', startPara = 0.01, regress = regress)
    LASSO_currency = LASSO_currency.append({'Window_size': wsize, 'Currency': i, 'Alpha': para},ignore_index= True)
    el= se
    return {'lasso': LASSO_currency, 'el':el, 'prd':prdList}

def multi_proccess_LASSO(windowList, paramList, regress = True):
    errorList_lasso = {}
    lasso_wisize = {}
    prdList= {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    lasso_report = Parallel(n_jobs = -4, verbose = 50, backend = 'threading')(delayed(parallel_LASSO)(i,regress,windowList,paramList) for i in col)
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

def parallel_svm (name, regress , windowList, paramList):
    i = name
    svm_currency = pd.DataFrame()
    el = []
    svm = SVR()
    se, rmse, wsize, para, prdList = sequential_grid_tune( dataDict[i],svm, window = windowList, paramList = paramList, paramName = 'C', startPara = 0.01, regress = regress)
    svm_currency = svm_currency.append({'Window_size': wsize, 'Currency': i, 'C': para},ignore_index= True)
    el= se
    return {'svm': svm_currency, 'el':el, 'prd': prdList}

def multi_proccess_svm(windowList, paramList, regress = True):
    errorList_svm = {}
    svm_wisize = {}
    prd_list = {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    svm_report = Parallel(n_jobs = -4, verbose = 50, backend = 'threading')(delayed(parallel_svm)(i,regress,windowList,paramList) for i in col)
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


#################### Variable Construction ################################

dataDict = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        dataDict[i] = (variable_cons(cpi, eurodep, fxspot, indprod, moneysupply, monStr = i))

##############################################################################
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\svmc.xlsx'
svmc = SVR()
from sklearn.svm import SVR
storepath = r'C:\Users\USER\Documents\Imperial College London\Spring Module\Big Data and Finane\Assignment\A2\Presentation\Presentation\Data\svm.xlsx'

svm = SVR()
errorList_svm, svm_wisize, svm_prd = multi_proccess_svm(windowList = np.arange(10,11,1), paramList = np.arange(0.01,0.02,0.01), regress = True)
SSED_svm, svm_tuned = multi_proccess_svm_report(errorList_svm, svm_wisize,isClassif = False)
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'SSEDif', row = 3, col = 3, supT = 'SVM SSE Differential')
plot_differential_report(colName = fxspot.drop("Date", axis = 1).columns.tolist(), data = SSED_svm, para = 'oosrsquare', row = 3, col = 3, supT = 'SVM OOS R-Squared')

new_excel(storepath)
[excel_writer(storepath,('SSED_'+i),SSED_svm[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
[excel_writer(storepath,('prd_'+i),SSED_prd[i])for i in fxspot.drop('Date',axis =1).columns.tolist()]
excel_writer(storepath, ('Window_size'), svm_tuned)
