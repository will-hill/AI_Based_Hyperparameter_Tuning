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