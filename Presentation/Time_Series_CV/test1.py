import unittest

import pandas as pd
from Time_Series.CrossValidation import *
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
import pandas as pd

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

colName_ = fxspot.drop('Date',axis = 1).columns.tolist()
coefMatrix = []
dataDict = {}
ppp = {}
moneyrtl = {}
for i in fxspot.columns.tolist():
    if i !='Date':
        dataDict[i] = (variable_cons(cpi, eurodep, fxspot, indprod, moneysupply, monStr = i))
        ppp[i] = dataDict[i][['change_in_spot','ppp']].copy()
        moneyrtl[i] = dataDict[i][['change_in_spot','moneyrtl']].copy()
        dataDict[i] = dataDict[i].drop('ppp',axis = 1).drop('moneyrtl', axis = 1)


class Test_test1(unittest.TestCase):


    def test_rolling_horizon(self):
        mdl = LinearRegression()
        data = dataDict['CAD']
        rh  = rolling_Horizon(mdl = mdl, data = data)
        self.assertEqual(len(data),rh.wsize + len(rh.error2) + rh.startInd)


    def test_rolling_crossvalidation(self):
        winList = [1,2,5,6]
        mdl = LinearRegression()
        data = dataDict['SEK']
        rc = rolling_cv(data = data, mdl = mdl, windowList = winList)
        self.assertIn(rc.bestWindow,winList)

    def test_grid_tune (self):
        winList = [1,2,5,6]
        paraName = 'alpha'
        paraList = [1,2,3,4,5]
        mdl = Lasso(normalize = True)
        data = dataDict['SEK']
        gp = grid_tune_parameter(mdl,data,winList,paraList,paraName)
        self.assertIn(gp.wsize,winList)
        self.assertIn(gp.para, paraList)

    def test_sequential_grid_tune (self):
        winList = [1,2,5,6]
        paraName = 'alpha'
        paraList = [1,2,3,4,5]
        mdl = Lasso(normalize = True)
        data = dataDict['SEK']
        gp = sequential_grid_tune(data, mdl,winList,paraList,paraName)
        self.assertIn(gp.wsize,winList)
        self.assertIn(gp.para, paraList)

    def test_multi_proccessing (self):
        winList = [1,2,5,6]
        colName = ['CAD','SEK']
        paraName = 'alpha'
        paraList = [1,2,3,4,5]
        mdl = Lasso(normalize = True)
        mp = paralell_processing(mdl,dataDict, winList, paraList, 'alpha',colName,True,True,False)
        print(mp.report_tuned)
        for i in colName:
            self.assertIn(mp.wisize[i]['Window_size'].tolist()[0],winList)
            self.assertIn(mp.wisize[i]['para'].tolist()[0], paraList)
            self.assertIn(mp.wisize[i]['Currency'].tolist()[0],colName)

    def test_bench_mark (self):
        bch = benchMark()
        hs = bch.historical_mean(dataDict['CAD'].change_in_spot,2)
        self.assertEqual(len(hs), len(bch.error2))
        self.assertListEqual(hs,bch.error2)
        cbch = benchMark()
        se = cbch.classification_benchmark(dataDict['CAD'].change_in_spot)
        self.assertListEqual(se,cbch.error2)
        self.assertEqual(len(se),len(cbch.prd))





if __name__ == '__main__':
    unittest.main()




