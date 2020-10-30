import numpy as np
import pandas as pd

from mymain import mypredict



# save weighed mean absolute error WMAE

esitmators = [*range(20, 31, 1)]
for i in esitmators:
    train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
    test = pd.read_csv('test.csv', parse_dates=['Date'])
    n_folds = 10
    next_fold = None
    wae = []
    # time-series CV
    for t in range(1, n_folds+1):
        # *** THIS IS YOUR PREDICTION FUNCTION ***
        train, test = mypredict(train, test, next_fold, t,i)

        # Load fold file
        # You should add this to your training data in the next call to mypredict()
        fold_file = 'fold_{t}.csv'.format(t=t)
        next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

        # extract predictions matching up to the current fold
        scoring_df = next_fold.merge(test, on=['Date', 'Store', 'Dept'], how='left')

        # extract weights and convert to numpy arrays for wae calculation
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
        actuals = scoring_df['Weekly_Sales'].to_numpy()
        preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

        wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())


    average = sum(wae)/len(wae)    
    print(average, i)
