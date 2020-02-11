#%%
from catboost import CatBoostClassifier

import data_prep, ai_config
import importlib
importlib.reload(data_prep)

#%%

test, train = data_prep.get_fold(0, '../folds')

model = CatBoostClassifier(iterations=10,
                           depth=10,
                           learning_rate=0.9,
                           custom_metric=['AUC'],
                           eval_metric='AUC',
                           verbose=True)

X_train = train.drop([ai_config.Y_COL], axis=1)
y_train = train[[ai_config.Y_COL]]

model.fit(X_train, y_train)

X_test = test.drop([ai_config.Y_COL], axis=1)
y_test = test[[ai_config.Y_COL]]

scores = model.score(X_test, y_test)

display(scores)

#%%
print(y.unique())
#%%
model.fit(X, y)
#%%
scores = model.score(test.drop([ai_config.Y_COL], axis=1), test.([ai_config.Y_COL]))
#%%
def create_optuna_study(study_name):
    import optuna
    import ai_config
    study = optuna.create_study(
        study_name=study_name, 
        storage=ai_config.DB_CONN, 
        pruner=optuna.MedianPruner(n_startup_trials=100, n_warmup_steps=50, interval_steps=10)
        )   
    return study     
#%%
def go(study_name, objective_fn, training_csv, testing_csv):
    print('starting...')
    
    print('create optuna study', study_name)
    study = create_optuna_study()
    
    
    print('...done')

#%%
go()