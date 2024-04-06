from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,load_object

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig



    def RandomForestModel(self):
        try:

            preprocessor_path = os.path.join('artifacts', "preprocessor.pkl")
            pipe = load_object(preprocessor_path)

            pipeline = make_pipeline(pipe,RandomForestClassifier())
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            rf = self.RandomForestModel()

            rf.fit(X_train,y_train)

            y_pred_train = rf.predict(X_train)

            y_pred_test = rf.predict(X_test)


            train_acc = f"Train accuracy score of model is {round(accuracy_score(y_train, y_pred_train), 4) * 100}%"

            test_acc = f"Test accuracy score of model is {round(accuracy_score(y_test, y_pred_test),4)*100}%"


            logging.info(f"{train_acc}{test_acc}")

            logging.info(f"Saved model object.")

            save_object(

                file_path=self.model_trainer_config.model_file_path,
                obj=rf

            )


            return (
                train_acc,
                test_acc
            )

        except Exception as e:
            raise CustomException(e, sys)