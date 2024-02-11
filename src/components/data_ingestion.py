import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from sklearn.model_selection import train_test_split
from exception import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config  = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("train,test split has been initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header =  True)
            logging.info("ingestion of data is compeleted")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiatedatatransformation(train_data,test_data)