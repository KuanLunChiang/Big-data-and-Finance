from rolling_regression import rolling_horizon

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

#################### Variable Construction ################################
def variable_cons (cpi, eurodep, fxspot, indprod, moneysupply, monStr = 'AUD'):
    data = pd.DataFrame(columns = ['change_in_spot','int_diff','inf_diff','ip_diff','ms_diff'])
    data.change_in_spot = np.log(fxspot[monStr]) - np.log(fxspot[monStr].shift(1))
    data.int_diff = eurodep[monStr] - eurodep.USD
    data.inf_diff = (np.log(cpi[monStr]) - np.log(cpi[monStr].shift(1)).apply(lambda x: 0 if pd.isnull(x) else x) )- (np.log(cpi.USD) - np.log(cpi.USD.shift(1)).apply(lambda x: 0 if pd.isnull(x) else x))
    data.ip_diff = np.log(indprod[monStr]) - np.log(indprod.USD)
    data.ms_diff = np.log(moneysupply[monStr]) - np.log(moneysupply.USD)
    data = data.dropna()
    return data

dataDict = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        dataDict[i] = (variable_cons(cpi, eurodep, fxspot, indprod, moneysupply, monStr = i))


rolling_horizon(dataDict['SEK'])