import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime as dt
import warnings # for using warning and here use for hidding them 
warnings.filterwarnings('ignore')
import os # for operating system operation like path , copy , move etc 
print(os.listdir("F:\cab_rental\python")) # here display what files are in the os.listdir path          
from sklearn.model_selection import train_test_split
import xgboost as xgb

abcdda = pd.read_csv("train_cab.csv")
train=pd.read_csv("train_cab.csv")
test = pd.read_csv("test.csv")

#for noisy value 
train = train.drop(train.index[1327])
train = train.drop(train.index[1123])
train = train.drop(train.index[1015])
train = train.drop(train.index[1072])
train.dtypes

train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
train['fare_amount']=pd.to_numeric(train['fare_amount'])
train.isnull().sum()
test.isnull().sum()
train.describe()

train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -300) & (train['pickup_longitude'] < 300)]
train = train.loc[(train['pickup_latitude'] > -300) & (train['pickup_latitude'] < 300)]
train = train.loc[(train['dropoff_longitude'] > -300) & (train['dropoff_longitude'] < 300)]
train = train.loc[(train['dropoff_latitude'] > -300) & (train['dropoff_latitude'] < 300)]
#train = train.loc[train[columns_to_select] < ]
# Let's assume taxa's can be mini-busses as well, so we select up to 8 passengers.
train = train.loc[train['passenger_count'] <= 8]
train.describe()

train['fare_amount'] = train['fare_amount'].replace(0,np.NaN)
train['pickup_longitude'] = train['pickup_longitude'].replace(0,np.NaN)
train['pickup_latitude'] = train['pickup_latitude'].replace(0,np.NaN)
train['dropoff_longitude'] = train['dropoff_longitude'].replace(0,np.NaN)
train['dropoff_latitude'] = train['dropoff_latitude'].replace(0,np.NaN)


lis = []
for i in range(0, train.shape[1]):
    #print(i)
    if(train.iloc[:,i].dtypes == 'object'):
        train.iloc[:,i] = pd.Categorical(train.iloc[:,i])
        #print(marketing_train[[i]])
        train.iloc[:,i] = train.iloc[:,i].cat.codes
        train.iloc[:,i] = train.iloc[:,i].astype('object')

        lis.append(train.columns[i])

for i in range(0, train.shape[1]):
    train.iloc[:,i] = train.iloc[:,i].replace(-1, np.nan)

train = train.dropna(how = 'any', axis = 'rows')


combine = [test, train]
for dataset in combine:
    # Distance is expected to have an impact on the fare
    dataset['longitude_distance'] = dataset['pickup_longitude'] - dataset['dropoff_longitude']
    dataset['latitude_distance'] = dataset['pickup_latitude'] - dataset['dropoff_latitude']
    
    # Straight distance
    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 + dataset['latitude_distance'] ** 2) ** .5
    dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    
    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    R = 6371e3 # Metres
    phi1 = np.radians(dataset['pickup_latitude'])
    phi2 = np.radians(dataset['dropoff_latitude'])
    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    a = np.sin(phi_chg / 2) ** .5 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2) ** .5
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    dataset['haversine'] = d
    
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset['bearing'] = np.arctan2(y, x)
    
    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival
    # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
    

train = train.loc[train['haversine'] != 0]
train = train.dropna()

train.head()

train.isnull().sum()
test.isnull().sum()

median = test['haversine'].median()
test['haversine'] = test['haversine'].fillna(median)


plt.figure(figsize=(10,5))
sns.distplot(train['fare_amount'])
plt.title('Fare distribution')

corr_mat = train.corr()
corr_mat.style.background_gradient(cmap='coolwarm')

colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

train_copy= train.copy(deep=True)
train_copy_fare = train['fare_amount'].copy(deep=True)
test_copy= test.copy(deep=True)


x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)
x_train = train.drop('fare_amount', axis=1)

ab = x_train.copy(deep=True)
ab = ab.drop('pickup_datetime',axis=1)
cd = y_train
ef = x_test.copy(deep=True)
ef = ef.drop('pickup_datetime',axis=1)
# Set up the models.

from sklearn.metrics import mean_squared_error
from math import sqrt

# Linear Regression Model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(ab, cd)
regr_pred = regr.predict(ef)
rms1 = sqrt(mean_squared_error(y_test, regr_pred))
print("RMS error in Random Forest: ",rms1)


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(ab, cd)
rfr_pred = rfr.predict(ef)
rms = sqrt(mean_squared_error(y_test, rfr_pred))
print("RMS error in Random Forest: ",rms)

x_pred = test

# Let's run XGBoost and predict those fares!
#x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

from xgboost import XGBRegressor as xgb
import xgboost as xgb
model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)

# print the error matrix 
#from sklearn.metrics import confusion_matrix 
#from sklearn.metrics import accuracy_score 
#from sklearn.metrics import classification_report 
#results = confusion_matrix(y_test, xgb_pred) 
#print 'Confusion Matrix :'
#print(results) 
#print ('Accuracy Score :',accuracy_score(y_test, xgb_pred))
#print 'Report : '
#print classification_report(y_test, xgb_pred) 
