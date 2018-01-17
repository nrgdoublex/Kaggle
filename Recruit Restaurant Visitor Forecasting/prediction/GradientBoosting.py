import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import ensemble


# training data
# read data
air_visit = pd.read_csv('../data/air_visit_data.csv')
date_info = pd.read_csv('../data/date_info.csv')
store_info = pd.read_csv('../data/air_store_info.csv')
reserve = pd.read_csv('../data/air_reserve.csv')

# figure out features
# features are: id, genre, area, year, month, reserve visitor per day, weekend, holiday
air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date']).map(lambda x: x.date())

# weekend and holiday
date_info['weekend'] = date_info['day_of_week'].map(lambda x: int(x == 'Friday' or x == 'Saturday'))
date_info.drop('day_of_week',axis=1, inplace=True)
date_info['calendar_date'] = pd.to_datetime(date_info['calendar_date']).map(lambda x: x.date())
training = air_visit.merge(date_info,how='left',left_on='visit_date',right_on='calendar_date')
training.drop('calendar_date',axis=1,inplace=True)

# genre and area
store_info['area'] = store_info['air_area_name'].map(lambda x: str(x).rsplit(' ',1)[0])
lbl = preprocessing.LabelEncoder()
store_info['area'] = lbl.fit_transform(store_info['area'])
store_info['air_genre_name'] = lbl.fit_transform(store_info['air_genre_name'])
store_info.drop(['air_area_name'],axis=1,inplace=True)
training = training.merge(store_info,how='left',on='air_store_id')


# total reserve visitors per day
reserve['visit_datetime'] = pd.to_datetime(reserve['visit_datetime'])
reserve['visit_date'] = reserve['visit_datetime'].map(lambda x: x.date())
reserve.drop(['reserve_datetime','visit_datetime'],axis=1,inplace=True)
reserve = reserve.groupby(['visit_date','air_store_id'],as_index=False).sum()
reserve.rename(columns={'reserve_visitors':'total_reserve_visitors'},inplace=True)
training = training.merge(reserve,how='left',on=['visit_date','air_store_id'])

# year and month
training['year'] = training['visit_date'].map(lambda x: x.year)
training['month'] = training['visit_date'].map(lambda x: x.month)
training.fillna(0,inplace=True,axis=1)
training['air_store_id2'] = lbl.fit_transform(training['air_store_id'])


# test data
test = pd.read_csv('../data/sample_submission.csv')
test['air_store_id'] = test['id'].map(lambda x: x.rsplit('_',1)[0])
test['visit_date'] = pd.to_datetime(test['id'].map(lambda x: x.rsplit('_',1)[1])).map(lambda x: x.date())
test = test.merge(date_info,how='left',left_on='visit_date',right_on='calendar_date')
test.drop('calendar_date',axis=1,inplace=True)
test = test.merge(store_info,how='left',on='air_store_id')
test = test.merge(reserve,how='left',on=['visit_date','air_store_id'])
test['year'] = test['visit_date'].map(lambda x: x.year)
test['month'] = test['visit_date'].map(lambda x: x.month)
test.fillna(0,inplace=True,axis=1)
#test.drop(['id','visitors'],axis=1,inplace=True)
test['air_store_id2'] = lbl.transform(test['air_store_id'])

# finalize input
cols = sorted([col for col in training.columns.values if col not in ['visit_date','visitors','id','air_store_id']])
inputX = training[cols]
inputY = np.log1p(training['visitors'].values)
outputX = test[cols]

# train the model
model = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3, n_estimators=200, subsample=0.8, max_depth =10)
model.fit(inputX, inputY)
test['visitors'] = np.expm1(model.predict(outputX))
print test.head()
test[['id', 'visitors']].to_csv('../data/submission.csv', index=False)