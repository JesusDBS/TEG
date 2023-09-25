import os
import sys
import pickle
import joblib
import numpy as np
from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, report_done

class DetectionTrainingModelPipeline:
    """This class trains LSTM classification model for detection.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)

    def __get_filename_path(self) -> str:
        path = self.configs['SAVE_DATA_PATH']['path'].copy()
        filename = self.configs['SAVE_DATA_PATH']['filename']
        path.append(filename)

        return os.sep.join(path)

    @report_done
    def load_dataset(self):
        """Loads the dataset. Returns a tuple
        """
        filename = self.__get_filename_path()
        
        with open(filename, 'rb') as file:
            self.input_features, self.output_label = pickle.load(file)

    @report_done
    def lstm_data_transform(self):
        """ Changes data to the format for LSTM training
        """
        X, y = list(), list()
        
        for i in range(self.input_features.shape[0]):
            end_ix = i + self.configs['TIMESTEPS']
            if end_ix >= self.input_features.shape[0]:
                break
            
            seq_X = self.input_features[i:end_ix]
            seq_y = self.output_label[end_ix]
            X.append(seq_X)
            y.append(seq_y)

        x_array = np.array(X)
        y_array = np.array(y)
    
        self.data = (x_array, y_array)

    @report_done
    def scale_input_data(self):
        """Scales input data for training
        """
        scaler = MinMaxScaler(feature_range=tuple(self.configs['SCALER']['input']['feature_range']))
        scaler = scaler.fit(self.input_features)
        filename = self.__get_filename_path()
        filename = filename.replace('.pkl', '.inputScaler.gz')
        breakpoint()
        joblib.dump(scaler, filename)
        self.input_features = scaler.transform(self.input_features)
        

    def run(self):
        self.load_dataset()
        self.scale_input_data()

    def __call__(self) -> Any:
        self.run()

