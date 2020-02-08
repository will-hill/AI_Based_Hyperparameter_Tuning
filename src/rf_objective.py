# %%
def rf_ojbective(trial):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc    
    import pandas as pd
    
    # Random Forest Params
    rf_params = {
        'n_estimators': trial.suggest_int('n_estimators', 5, 2000),
        'max_depth': trial.suggest_int('max_depth', 1, 1000)
    }        
    """
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),        
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 2000),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'min_weight_fraction_leaf': trial.suggest_int('min_weight_fraction_leaf', 0.0, 0.99),
        'max_features': trial.suggest_uniform('max_features', 0.01, 0.99),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 0, 999),
        'min_impurity_decrease': trial.suggest_int('min_impurity_decrease', 0.0, 0.99),
        'min_impurity_split': trial.suggest_uniform('min_impurity_split', 0.0, 0.99),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'oob_score': trial.suggest_categorical('oob_score', [False, True]),
        'warm_start': trial.suggest_categorical('warm_start', [False, True]),
        'class_weight': trial.suggest_categorical('class_weight', [None, "balanced", "balanced_subsample"]),
        'ccp_alpha': trial.suggest_uniform('ccp_alpha', 0.0, 0.99),
        'max_samples': trial.suggest_uniform('max_samples', 0.5, 1.0)
    }"""

    results = []
    # Induction
    for i in range(0,6):
        
        train = pd.read_feather('../data/train_'+str(i)+'.ftr').set_index('TransactionID')
        X_train = train.drop(['isFraud', 'ProductCD'], axis=1)
        y_train = train['isFraud']
        print(y_train.sample(3))
        del train
        
        test  = pd.read_feather('../data/test_' +str(i)+'.ftr').set_index('TransactionID')
        X_test = test.drop(['isFraud', 'ProductCD'], axis=1)
        y_test = test['isFraud']
        print(y_test.sample(3))
        del test        
        
        print('rf')
        clf = RandomForestClassifier(random_state=0, **rf_params)
        print('rf fit')
        clf.fit(X_train, y_train)
        
        print('rf score')
        # auc, roc_auc_score, average_precision_score
        fpr, tpr, thresholds = roc_curve(y_test, [y_hat[1] for y_hat in clf.predict_proba(X_test)], pos_label=1)
        result = 1 - auc(fpr, tpr)
        results.append(result)
        print('result ')
    return results.mean()

#%%
import optuna
study = optuna.create_study()
study.optimize(rf_ojbective, n_trials=1)