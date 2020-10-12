import sys
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import time 

t0 = time.time()



#read files into df
df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

#caputure pid values
pid = df2[['PID']].to_numpy()

#drop unrelated columns
df1 = df1.drop(['PID','Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'], axis=1)
df2 = df2.drop(['PID','Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude'], axis=1)

#winsorization of data
winsor = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
for col in winsor:
    df1[col] = df1[col].clip(lower=0, upper=df1[col].quantile(0.95))
    df2[col] = df2[col].clip(lower=0, upper=df2[col].quantile(0.95))

#get rid of nan values by equating gragre year to house year
df1.Garage_Yr_Blt.fillna(df1.Year_Built, inplace=True)
df2.Garage_Yr_Blt.fillna(df2.Year_Built, inplace=True)


#get y for training data then drop in df1
y_train = np.log(df1[['Sale_Price']].values)
df1 = df1.drop(['Sale_Price'], axis=1)

#onehot encode columns in both train and test and make X_train and X_test
cols=[]
columns = df1.columns.to_list()
categorical_feature_mask = df1.dtypes==object
categorical_cols = df1.columns[categorical_feature_mask].tolist()
for i,c in enumerate(columns):
    if c in categorical_cols:
        cols.append(i)

enc = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto', handle_unknown='ignore'), cols)],  remainder='passthrough')
enc.fit(df1)
X_train = enc.fit_transform(df1).toarray()
X_test = enc.transform(df2).toarray()

#train ridge
clf = Ridge(alpha=1)
clf.fit(X_train, y_train)
y_test = np.exp(clf.predict(X_test))

#write ridge to file
f = open("mysubmission1.txt", "w")
f.write("PID,  Sale_Price")
f.write("\n")
num = len(y_test)
for i,p in enumerate(y_test):
    send = str(pid[i][0])+", "+str(p[0])
    f.write(send)
    if i != num - 1:
        f.write("\n")
f.close()

#train RF
regr = RandomForestRegressor()
regr.fit(X_train, y_train)
y_test = np.exp(regr.predict(X_test))

#write ridge to file
f = open("mysubmission2.txt", "w")
f.write("PID,  Sale_Price")
f.write("\n")
num = len(y_test)
for i,p in enumerate(y_test):
    send = str(pid[i][0])+", "+str(p)
    f.write(send)
    if i != num - 1:
        f.write("\n")
f.close()

t1 = time.time()
print(t1-t0)