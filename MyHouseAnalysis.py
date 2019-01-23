# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:55:01 2018

@author: bhunt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Lets Start by preprocessing the data. First look at the data

#%%
print(train.head(5))

print(test.head(5))


#%%

#Save train and test Ids and then remove them from the data frames because they are not predictive
trainID = train['Id']
testID = test['Id']

test.drop('Id', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)

#save Sales price in a separate array

Y_train = train['SalePrice']



#%%
#train.drop('SalePrice', axis=1, inplace=True)

#print(test.apply(lambda x: x.count(), axis=0))
#print(train.apply(lambda x: x.count(), axis=0))



#%%


all_data = pd.concat((train, test)).reset_index(drop=True)

#plt.hist(Y_train, bins=50)

# distribution doesn't look normal, try a log transform

plt.hist(np.log(Y_train+1), bins=50)

# looks more normal now

Y_train2=np.log(Y_train+1)

#%%

all_data.drop(['SalePrice'], axis=1, inplace=True)

#%%
print(all_data.apply(lambda x: x.count(), axis=0))
arrayofmissing=all_data.apply(lambda x: x.count(), axis=0)
# Now we can see which variables are missing data and I can add it back in

#%%

all_data["Alley"] = all_data["Alley"].fillna("None")

# All basement Nans come from there being no basement

#%%

for col in ["BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2", "Fence", "FireplaceQu", 
            "GarageCond", "GarageFinish", "GarageQual", "GarageType"]:
    all_data[col]=all_data[col].fillna("None")

#%%
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


#%%

for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath",
            "GarageArea", "GarageCars", "GarageYrBlt"]:
    all_data[col]=all_data[col].fillna(0)
    
#%%

for col in ["Electrical", "Exterior1st", "Exterior2nd", "KitchenQual"]:
    all_data[col]=all_data[col].fillna(all_data[col].mode()[0])

#%%
for col in ["Functional"]:
    all_data[col]=all_data[col].fillna("Typ")
    
#%%
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    
#%%
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])

#%%
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#%%

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

#%%
 
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])

#%%

all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(0)

#%%

all_data["Utilities"] = all_data.drop(['Utilities'], axis=1)

#%%

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#%%

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for i in cols:
    labels = LabelEncoder()
    labels.fit(list(all_data[i].values)) 
    all_data[i] = labels.transform(list(all_data[i].values))

#%%

# Generally House Price correlates strongly with home total square footage
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

#%%

from scipy.stats import norm, skew #for some statistics
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

#%%

skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

#%%
all_data = pd.get_dummies(all_data)
print(all_data.shape)

#%%
ntrain = Y_train.size
train = all_data[:ntrain]
test = all_data[ntrain:]

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 0)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20, 220, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
# Fit the random search model
rf_random.fit(train, Y_train2)

rf_random.best_params_

#%%

from sklearn.model_selection import KFold, cross_val_score, train_test_split

n_folds=5

def rmsle_cvrandforest(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, Y_train2, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#%%

    
print(rf_random.best_params_)

RFModel = RandomForestRegressor(n_estimators = 400, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', max_depth = 200, bootstrap = True)
RFModel.fit(train, Y_train2)

#%%

rmsle_cvrandforest(RFModel)

#%%

predictions = RFModel.predict(test)

predictions = np.exp(predictions)-1

#%%

output = np.array([testID, predictions])
output = np.transpose(output)




#%%

outdata=pd.DataFrame(data=output, columns=["ID", "SalesPrice"])

export_csv = outdata.to_csv(r"C:\Documents\KaggleData\HousePrice\pred.csv", header=True)
