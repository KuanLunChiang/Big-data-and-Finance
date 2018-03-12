def parallel_LASSO (name):
    i = name
    LASSO_currency = pd.DataFrame()
    el = []
    mdl = Lasso(normalize = True)
    se, rmse, wsize, para = grid_tune_parameter (mdl, dataDict[i], window = np.arange(65,85), paramList = np.arange(0.001,0.1,0.001))
    LASSO_currency = LASSO_currency.append({'Window_size': wsize, 'Currency': i, 'Alpha': para},ignore_index= True)
    el= se
    return {'lasso': LASSO_currency, 'el':el}



def multi_proccess_LASSO():
    errorList_lasso = {}
    lasso_wisize = {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    col = fxspot.drop("Date", axis = 1).columns.tolist()
    lasso_report = Parallel(n_jobs = 4, verbose = 50, backend = 'threading')(delayed(parallel_LASSO)(i) for i in col)
    for i in colName:
        errorList_lasso[i] = lasso_report[colName.index(i)]['el']
        lasso_wisize[i] =lasso_report[colName.index(i)]['lasso']
    return errorList_lasso, lasso_wisize



errorList_lasso, lasso_wisize = multi_proccess_LASSO()
colName = fxspot.drop('Date',axis =1).columns.tolist()
[lasso_wisize[i]['Alpha'] for i in colName]

lasso_tuned = pd.DataFrame()
for i in range(len(colName)):
    lasso_tuned = lasso_tuned.append(lasso_wisize[colName[i]])
lasso_tuned= lasso_tuned.reset_index().drop('index',axis = 1)




def parallel_RF (name):
    i = name
    RF_currency = pd.DataFrame()
    el = []
    mdl = RandomForestRegressor(random_state = 123)
    se, rmse, wsize, para = grid_tune_parameter (mdl, dataDict[i], window = np.arange(3,100), paramList = np.arange(1,4,1), paramName = 'max_features')
    RF_currency = RF_currency.append({'Window_size': wsize, 'Currency': i, 'max_features': para},ignore_index= True)
    el= se
    return {'rf': RF_currency, 'el':el}



def multi_proccess_RF():
    errorList_RF = {}
    RF_wisize = {}
    colName = fxspot.drop('Date',axis =1).columns.tolist()
    RF_report = Parallel(n_jobs = 1, verbose = 50, backend = "threading")(delayed(parallel_RF)(i) for i in colName)
    for i in colName:
        errorList_RF[i] = RF_report[colName.index(i)]['el']
        RF_wisize[i] =RF_report[colName.index(i)]['rf']
    return errorList_RF, RF_wisize

errorList_RF, RF_wisize = multi_proccess_RF()

for i in fxspot.drop('Date',axis =1).columns.tolist():
    se, rmse, wsize, para = grid_tune_parameter (mdl, dataDict[i], window = np.arange(60,61), paramList = np.arange(3,4,1), paramName = 'max_features')
