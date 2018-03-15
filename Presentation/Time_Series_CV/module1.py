from Time_Series.CrossValidation import rolling_cv
from sklearn.linear_model import Lasso
from Time_Series.CrossValidation import *


def paralell_support (mdl, data , regress , windowList, paramList, paraName, fixed = True):
      
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

winList = [1,2,5,6]
paraName = 'alpha'
paraList = [1,2,3,4,5]
mdl = Lasso(normalize = True)



mp = paralell_support(mdl,dataDict['CAD'],True ,winList, paraList, 'alpha')



sequential_grid_tune(dataDict['CAD'],mdl, window = winList, paramList = paraList, paramName = paraName, startPara = 0)

winList = [1,2,5,6]
paraName = 'alpha'
paraList = [1,2,3,4,5]
mdl = Lasso(normalize = True)
data = dataDict['SEK']
gp = sequential_grid_tune(data, mdl,winList,paraList,paraName)

gp.error2

colName_ = fxspot.drop('Date',axis = 1).columns.tolist()
winList = [1,2,5,6]
paraName = 'alpha'
paraList = [1,2,3,4,5]
mdl = Lasso(normalize = True)
mp = paralell_processing(mdl,dataDict, winList, paraList, 'alpha',colName_)
for i in mp.wisize:
    self.assertIn(i['Window_size'],winList)
    self.assertIn(i['para'], paraList)