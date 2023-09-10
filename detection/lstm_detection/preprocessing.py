import os
import sys
import glob
import ast
import pandas as pd
import numpy as np
from typing import Any
from rackio_AI import RackioAI

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, convert_list2tuple, report_done
from feature_extraction import FeatureExtraction


class DetectionPreprocessPipeline:
    """This class preprocess the data for training LSTM classification model for detection.
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

            self.data.extend(RackioAI.load(file))
        
        return self.data
    
    @staticmethod
    def __check_if_leak(components: list) -> bool:
        """Checks if the simulation has leaks
        """
        for component in components:
            parameters = component["PARAMETERS"]
            if 'leak' in parameters['LABEL'].lower():

                setpoints = parameters['SETPOINT']

                if isinstance(setpoints, str):

                    if isinstance(ast.literal_eval(setpoints), (list, tuple)):

                        return True

                if isinstance(setpoints, (list, tuple)):

                    return True

        return False
    
    @staticmethod
    def __get_leak_info(components:list):
        """
        Gets the leak info from genkey's Network Components key.
        """
        for component in components:

            parameters = component['PARAMETERS']
            if 'leak' in parameters['LABEL'].lower():

                break

        setpoints = parameters['SETPOINT']
        if isinstance(setpoints, str):

            setpoints = list(ast.literal_eval(setpoints))

        values = list(parameters['TIME']['VALUES'])
        pos = [x for x in range(len(setpoints)) if setpoints[x]==1]

        return (values, setpoints, pos)
    
    def __extract_features(self, window: pd.DataFrame) -> list:
        """Ferature extraction of a window of data
        """
        features_per_variable = list()
        variables = list(map(
            convert_list2tuple, self.configs['INPUTS_VARIABLES']
            ))
        for input_variable in variables:
            features_per_variable.append(
                self.feature_extraction(np.array(window[input_variable]))
                )
            
        return features_per_variable
    
    def __process_no_leak_data(self, file: dict):
        """Extract data's windows and its features for no leak data.
        """
        pass
    
    @staticmethod
    def __calculate_leak_flow_values_ratio(leak_flow_values_window: pd.Series) -> int:
        """Calculates leak flow ratio in order to determinate what window is leaking or not.
        """
        leak_flow_values_ratio = int((leak_flow_values_window != 0).sum()\
                                             / len(leak_flow_values_window) * 100)
        return leak_flow_values_ratio
    
    def __process_leak_data(self, file: dict):
        """Extracts and labels data's windows and its features for leak data. 
        """
        leak_components = file['genkey']['Network Component'][1::]
        leak_values, _ , leak_pos = self.__get_leak_info(components=leak_components)
        leak_point = leak_values[leak_pos[0]] * \
            60 / self.configs['SAMPLE_TIME']
        start_point = int(leak_point - \
                          self.configs['TIME_WINDOW']/self.configs['TIME_WINDOW_FRACTION'])
        end_point = int(leak_point + 60)

        df = file['tpl'][start_point:end_point]

        for point, _ in enumerate(df):
            if point + self.configs['TIME_WINDOW'] <= end_point:
                window = df[point:point + self.configs['TIME_WINDOW']]
                leak_flow_values_window = window[
                    ('GTLEAK_LEAK_LEAK', 'Leakage_total_mass_flow_rate', 'KG/S')
                    ]
                leak_flow_values_ratio = self.__calculate_leak_flow_values_ratio(
                    leak_flow_values_window
                )
                del leak_flow_values_window

                self.input_features.append(
                    np.concatenate(
                        self.__extract_features(window), axis=1)
                )
                del window

                if leak_flow_values_ratio >= self.configs['LEAK_RATIO']:
                    self.output_label.append(True)
                
                else:
                    self.output_label.append(False)

    def __reshape_features(self):
        self.output_label = np.array(self.output_label).reshape(len(self.output_label), 1)
        self.input_features = np.concatenate(self.input_features, axis=0)
    
    @report_done
    def process_data(self):
        """Labels the data as leak and no leak
        """
        self.output_label = list()
        self.input_features = list()
        for num_case, file in enumerate(self.data[0:10]):
            print(f"......Processing {num_case + 1} of {len(self.data)}.......")

            components = file['genkey']['Network Component'][1::]
            if self.__check_if_leak(components):
                self.__process_leak_data(file)

            else:
                self.__process_no_leak_data(file)

        self.__reshape_features()
    
    def run(self):
        self.load_data()
        self.process_data()
        
    def __call__(self) -> Any:
        self.run()