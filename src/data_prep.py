#%%def get_ieee_fraud_data():      import pandas as pd    import ai_config    #  KAGGLE_PROJECT = 'ieee-fraud-detection'        # 18 seconds    train = pd.read_csv('../data/train_transaction.csv', index_col=ai_config.IDX_COL).merge(pd.read_csv('../data/train_identity.csv', index_col=ai_config.IDX_COL), how='left', left_index=True, right_index=True)    train.columns = train.columns.str.replace("-", "_")        # 17 seconds    test  = pd.read_csv('../data/test_transaction.csv' , index_col=ai_config.IDX_COL).merge(pd.read_csv( '../data/test_identity.csv', index_col=ai_config.IDX_COL), how='left', left_index=True, right_index=True)    test.columns = test.columns.str.replace("-", "_")        test.insert(0, 'isFraud', value=([-1] * test.shape[0]))    return pd.concat([train, test])# OBJECT to NUMdef obj_to_num(df, verbose=True):    import pandas as pd        for c in df.select_dtypes(include=['object']).columns:        if verbose:            print('\n----------\ncol:' + c)            print('\nsample...')            print(df[c].sample(3))        try:            df['num_' + c] = pd.to_numeric(df[c])            df = df.drop([c], axis=1)            if verbose:                print('SUCCESS: converted ', c, ' to ', df['num_' + c].dtype)        except ValueError as ve:            if verbose:                print(ve)                    end_mem = df.memory_usage().sum() / 1024**3    if verbose:         print('Mem. usage decreased to {:5.2f} GB ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))            return df# REDUCE DATA SIZESdef minify(df, verbose=True, use_feather=True):    # TODO: multi-process this function by column    import numpy as np    import ai_config    start_mem = df.memory_usage().sum() / 1024**3            for col in df.columns:        col_type = df[col].dtypes                if col_type in ai_config.NUMERICS:            c_min = df[col].min()            c_max = df[col].max()                        if str(col_type)[:3] == 'int':                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:                    df[col] = df[col].astype(np.int8)                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:                    df[col] = df[col].astype(np.int16)                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:                    df[col] = df[col].astype(np.int32)                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:                    df[col] = df[col].astype(np.int64)              else:                if not use_feather and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:                    df[col] = df[col].astype(np.float16)                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:                    df[col] = df[col].astype(np.float32)                else:                    df[col] = df[col].astype(np.float64)                            end_mem = df.memory_usage().sum() / 1024**3    if verbose: print('Mem. usage decreased to {:5.2f} GB ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))    return df# COL TYPESdef col_types(df):    import pandas as pd        cols = []            for c in df.columns:        #if not np.issubdtype(df[c].dtype, np.number):                        col_info = {}        col_info['name'] = c        col_info['n_unique'] = df[c].nunique()        col_info['dtype'] = df[c].dtype        cols.append(col_info)    return pd.DataFrame.from_dict(cols)          import pandas as pd    cols = []            for c in df.columns:        #if not np.issubdtype(df[c].dtype, np.number):                        col_info = {}        col_info['name'] = c        col_info['n_unique'] = df[c].nunique()        col_info['dtype'] = df[c].dtype        cols.append(col_info)    return pd.DataFrame.from_dict(cols)     # ONE-HOT ENCODEdef ohe_col(df, col):    df = pd.concat([df, pd.get_dummies(df[col], drop_first=True)], axis=1)    df = df.drop(columns=[col], axis=1)    return df# LABEL ENCODEdef le_col(df, col):    from sklearn import preprocessing    df['le_' + col] = preprocessing.LabelEncoder().fit_transform(df[col])     df = df.drop([col], axis=1)    return df# CREATE & SAVE FOLDSdef fold_and_save(df, path, folds=5):    from sklearn.model_selection import KFold    kf = KFold(n_splits=folds, shuffle=True, random_state=0)    i = 0    for train_index, test_index in kf.split(df):        print(i, "TRAIN:", train_index, "TEST:", test_index)                train_data = X.iloc[train_index]        train_data.reset_index().to_feather(path + 'train_' + str(i) + '.ftr')                test_data  = X.iloc[test_index]        test_data.reset_index().to_feather(path = 'test_' + str(i) + '.ftr')                i += 1# GET FOLDdef get_fold(i, path):    import pandas as pd    import ai_config    train = pd.read_feather(path + '/train_' + str(i) + '.ftr').set_index(ai_config.IDX_COL)    test =  pd.read_feather(path + '/test_'  + str(i) + '.ftr').set_index(ai_config.IDX_COL)    return test, test