import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
def mypredict(train, test, new_test, t):
    df = pd.concat([train, new_test], join="inner",ignore_index = True)
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df['week'] = df['Date'].apply(lambda x: x.week)
    df["holiday"] = df['IsHoliday'].apply(lambda is_holiday:1 if is_holiday else 0)

    X = df[['Store','Dept','week','holiday','year']].values
    y = df[['Weekly_Sales']].values
    reg = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=4, min_samples_split=3, min_samples_leaf=1).fit(X, y)

    test["holiday"] = test['IsHoliday'].apply(lambda is_holiday:1 if is_holiday else 0)
    test['month'] = pd.DatetimeIndex(test['Date']).month
    test['year'] = pd.DatetimeIndex(test['Date']).year
    test['week'] = test['Date'].apply(lambda x: x.week)

    X_test = test[['Store','Dept','week','holiday','year']].values
    y_pred = reg.predict(X_test)
    test['Weekly_Pred'] = y_pred
    
    return df, test