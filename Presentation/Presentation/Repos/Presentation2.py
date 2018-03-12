

winSize_currency = pd.DataFrame(columns= ['Window_size', 'Currency' ,'RMSE'])
error2_list = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        mdl = LinearRegression()
        se, rmse, wsize = rolling_cv (mdl, dataDict[i], windowList = np.arange(10,11))
        winSize_currency = winSize_currency.append({'Window_size': wsize, 'Currency': i, 'RMSE': rmse},ignore_index= True)
        error2_list[i] = se


mean_se = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        ws = winSize_currency.loc[winSize_currency.Currency ==i].Window_size.tolist()[0]
        mean_se[i] = historical_mean(dataDict[i].change_in_spot,ws)


SSED = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        regError = error2_list[i]
        meanError = mean_se[i]
        SSED[i] = cum_sse_report (regError, meanError)



colName = fxspot.drop('Date',axis =1).columns.tolist()
lasso_tuned = pd.DataFrame()
for i in range(len(colName)):
    lasso_tuned = lasso_tuned.append(lasso_wisize[colName[i]])
lasso_tuned= lasso_tuned.reset_index().drop('index',axis = 1)

mean_se = {}
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