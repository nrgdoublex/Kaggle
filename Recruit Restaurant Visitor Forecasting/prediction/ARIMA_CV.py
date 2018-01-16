import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# figure out test range
test = pd.read_csv('../data/sample_submission.csv')
test = pd.DataFrame(test.id.str.rsplit('_',1).tolist(), columns = ['id','visit_date'])
test_date = pd.to_datetime(test['visit_date'].unique())
pred_len = max(test_date) - min(test_date)

# preprocess data
air_visit = pd.read_csv('../data/air_visit_data.csv')
air_visit.visit_date = pd.to_datetime(air_visit.visit_date)
all_ids = air_visit['air_store_id'].unique()
air_visit['visitors'] = pd.to_numeric(air_visit['visitors'],downcast='float')

# parameters
p = [2,3,4,5]

# do validation for a number of random ids
num_ids = 10
np.random.shuffle(all_ids)
sample_ids = all_ids[:num_ids]
for air_id in sample_ids:
        
    # extract only data we want
    data = air_visit[air_visit['air_store_id'] == air_id].sort_index(axis=0,level=1)
    data = data.drop('air_store_id',axis=1)
    
    training = data[data['visit_date'] <= (max(air_visit.visit_date) - pred_len)]
    testing = data[data['visit_date'] > (max(air_visit.visit_date) - pred_len)]
    training.set_index(keys='visit_date',inplace=True)
    testing.set_index(keys='visit_date',inplace=True)

    min_error, min_p, mix_q = float('infinity'), -1, -1
    for pi in p:
        # train for each set of parameters
        prediction = []
        training1 = training.copy(deep=True)
        testing1 = testing.copy(deep=True)
        try:
            for t in testing1.index:
                model = ARIMA(training1,order=(pi,0,1))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                prediction.append(yhat)
                training1 = training1.append(testing1.loc[t,:])
                
                
            error = mean_squared_error(testing1['visitors'].values, prediction)
            if error < min_error:
                min_p = pi
                min_error = error
        except np.linalg.linalg.LinAlgError:
            continue
        except ValueError:
            continue 
    print('ID: %s, best p: %d, best MSE: %.3f' % (air_id, min_p, min_error))