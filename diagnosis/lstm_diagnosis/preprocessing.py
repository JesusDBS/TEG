import os
import sys
import glob
import ast
import pickle
import random
import pandas as pd
import numpy as np
from typing import Any
from rackio_AI import RackioAI
from tqdm import tqdm

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, convert_list2tuple, report_done
from feature_extraction import FeatureExtraction


class DiagnosisRegressionPreprocessingPipeline:
    """This class preprocess the data for training LSTM regression model for diagnosis.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)
        self.feature_extraction = FeatureExtraction(**self.configs['FEATURE_EXTRACTION_CONFIGS'])

    @report_done
    def load_data(self) -> list:
        """Loads the data in pkl format
        """
        path = os.path.join(*self.configs['DATA_PATH']['path'])
        filename = self.configs['DATA_PATH']['filename']
        self.data = list()
        files = list()
        if filename:

            files.extend([os.path.join(path, filename)])

        else:

            path = os.path.join(*self.configs['DATA_PATH']['path']) + os.path.sep + '*.pkl'
            files = glob.glob(path)

        for file in files:
            print(file)
            self.data.extend(RackioAI.load(file))
            
        random.shuffle(self.data)
        return self.data
    
    def __extract_features(self, window: pd.DataFrame) -> list:
        """Ferature extraction of a window of data
        """
        features_per_variable = list()
        variables = list(map(
            convert_list2tuple, self.configs['INPUTS_VARIABLES']
            ))
        for input_variable in variables:
            input_variable = self.__add_white_noise(
                np.array(window[input_variable])
            )
            features_per_variable.append(
                self.feature_extraction(input_variable)
                )
            
        return features_per_variable
    
    @staticmethod
    def __add_white_noise(data: np.array, mean: float = 0, std: float = 0.001) -> np.array:
        noise = np.random.normal(mean, std, data.shape)
        return data + noise
    
    def __get_end_point(self, file: dict) -> int:
        end_time = file['genkey']['Global keywords']['INTEGRATION']['ENDTIME']

        if end_time['UNIT'].lower()=='m':
            _end_point = int(end_time['VALUE'] * 60 / self.configs['SAMPLE_TIME']) #Seg.

        elif end_time['UNIT'].lower()=='s':
            _end_point = int(end_time['VALUE'] / self.configs['SAMPLE_TIME'])
        
        return _end_point
    
    def _label_data(self, file: dict):
        """Extracts and labels data's windows and its features for leak data.
        """
        time_window = self.configs['TIME_WINDOW']
        end_point = self.__get_end_point(file)
        start_point = 20
        df = file['tpl'].iloc[start_point:end_point]
        variable_to_predict = self.configs['OUTPUT_TO_USE']
       
        for point in range(0, df.shape[0] + 1):
            if point + time_window <= end_point - start_point:
                window = df.iloc[point:point + time_window]
    
                self.input_features.append(
                    np.concatenate(
                        self.__extract_features(window), axis=1)
                )
                output_label = window[
                    convert_list2tuple(
                        self.configs['OUTPUT_VARIABLES'][variable_to_predict]
                        )
                    ]
                del window

                #Set output label as the last value of the window
                self.output_label.append(
                    output_label.iat[-1]
                )
    
    def _reshape_features(self):
        self.output_label = np.array(self.output_label).reshape(len(self.output_label), 1)
        self.input_features = np.concatenate(self.input_features, axis=0)

    @report_done
    def process_data(self):
        """Labels the data.
        """
        self.output_label = list()
        self.input_features = list()

        for file in tqdm(self.data):
            if len(file['tpl']) < 360:
                    continue
            
            self._label_data(file)

        self._reshape_features()

    @report_done
    def save_data(self):
        path = self.configs['SAVE_DATA_PATH']['path']
        filename = self.configs['SAVE_DATA_PATH']['filename']
        path.append(filename)

        filename = os.sep.join(path)

        with open(filename, "wb") as file:
            pickle.dump((
                self.input_features,
                self.output_label
            ), file)
            
    def run(self):
        self.load_data()
        self.process_data()
        self.save_data()

    def __call__(self) -> Any:
        self.run()