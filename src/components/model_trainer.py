import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from exception import CustomException
from logger import logging
from utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,training_array, test_array):
        try:
            logging.info("splitting train and test input data")
            X_train, y_train , X_test, y_test = (
                training_array[:,:-1],
                training_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report:dict = evaluate_model(X_train = X_train,y_train = y_train ,y_test = y_test, X_test = X_test, models = models)
            # best model score
            best_model_score = max(sorted(model_report.values()))

            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("no best model found")
            logging.info("best model found")

            save_object(self.model_trainer_config.train_model_file_path,obj = best_model)

            predictions = best_model.predict(X_test)
            r2_Score = r2_score(y_test,predictions)
            return r2_Score
        except Exception as e:
            raise CustomException(e,sys)
