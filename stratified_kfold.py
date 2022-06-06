import random
import collections
import pandas as pd

def getstratifiedkfold(k, df):
    folds = {}
    categories = list(set(df.cat))
    for cat in categories:
        df_cat = df[df.cat==cat]
        start = 0
        diff = int(len(df_cat)/k)
        for fold in range(k):
            if fold not in folds:
                folds[fold] = df_cat[start:start+diff]
            else:
                folds[fold] = pd.concat([folds[fold], df_cat[start:start+diff]])
            start = start+diff
        if k-1 not in folds:
            folds[k-1] = df_cat[start:]
        else:
            folds[k-1] = pd.concat([folds[fold], df_cat[start:]])

    return folds
    
    
def create_train_test_splits(k, df):
    folds = getstratifiedkfold(k, df)
    kfolddata = {}
    for i in range(k):
        test_fold = folds[i]
        test_fold = test_fold.sample(frac=1).reset_index(drop=True)

        train_fold = []
        for j in range(k):
            if i!=j: 
                train_fold.append(folds[j])

        train_fold = pd.concat(train_fold)
        train_fold = train_fold.sample(frac=1).reset_index(drop=True)
        kfolddata[i] = {'train': train_fold, 'test': test_fold}
    return kfolddata