import os 
import sys 

import numpy as np 
import pandas as pd 
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates the performance of different models.
    
    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    X_test (np.ndarray): Testing features.
    y_test (np.ndarray): Testing labels.
    models (dict): Dictionary of models to evaluate.
    
    Returns:
    dict: A dictionary containing model names and their respective scores.
    """
    try:
        model_report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]
            
            # Hyperparameter tuning using GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=param,
                cv=3
            )
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_) 
            model.fit(X_train, y_train)
            
            # Train only model
            # model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
                
            model_report[list(models.keys())[i]] = test_model_score
            
        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Saves an object to a file using pandas.
    
    Parameters:
    obj (any): The object to save.
    file_path (str): The path where the object will be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)