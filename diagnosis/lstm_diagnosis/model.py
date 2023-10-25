import os
import sys
import pickle
import joblib
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Any, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from rackio_AI import RackioAI
from tqdm import tqdm

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, report_done, convert_list2tuple
from feature_extraction import FeatureExtraction

class DiagnosisTrainingModelPipeline:
    """This class trains LSTM regression model for diagnosis.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)

    def __get_filename_path(self, key: str = 'SAVE_DATA_PATH') -> str:
        path = self.configs[key]['path'].copy()
        filename = self.configs[key]['filename']
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

        self.input_features = np.array(X)
        self.output_label = np.array(y)

    @report_done
    def scale_input_data(self):
        """Scales input data for training
        """
        scaler = MinMaxScaler(feature_range=tuple(self.configs['SCALER']['input']['feature_range']))
        scaler = scaler.fit(self.input_features)
        filename = self.__get_filename_path()
        filename = filename.replace('.pkl', '.inputScaler.gz')
        
        joblib.dump(scaler, filename)
        self.input_features = scaler.transform(self.input_features)

    @report_done
    def scale_output_data(self):
        """Scales output data for training
        """
        scaler = MinMaxScaler(feature_range=tuple(self.configs['SCALER']['output']['output_range']))
        scaler = scaler.fit(self.output_label)
        filename = self.__get_filename_path()
        filename = filename.replace('.pkl', '.outputScaler.gz')
        
        joblib.dump(scaler, filename)
        self.output_label = scaler.transform(self.output_label)

    @report_done
    def split_data(self):
        """Splits the data into train and test set
        """
        shuffle = self.configs['TRAINING']['dataset']['shuffle']
        test_size = self.configs['TRAINING']['dataset']['test_size']
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.input_features, 
                self.output_label,
                test_size = test_size, 
                shuffle = shuffle)
        
    def __create_model(self, units: list, activations: list):
        self.model = tf.keras.models.Sequential()
        output_units = units.pop(-1)
        output_activation = activations.pop(-1)
        return_sequences = True

        for layer in range(len(units)):

            if layer == len(units) - 1:
                return_sequences = False

            self.model.add(
                tf.keras.layers.LSTM(
                    units[layer],
                    activation = activations[layer],
                    return_sequences = return_sequences
                )
            )

        # output layer
        self.model.add(
            tf.keras.layers.Dense(
                output_units,
                activation = output_activation,
            )
        )
        
    @staticmethod
    def __define_callbacks(**kwargs):
        return tf.keras.callbacks.EarlyStopping(
            **kwargs
        )
    
    @staticmethod
    def __define_optimizer(**kwargs):
        return tf.keras.optimizers.Adam(
            **kwargs
        )
    
    def __compile(self, optimizer, **kwargs):
        self.model.compile(
            optimizer=optimizer,
            loss=kwargs['loss'],
            metrics = [kwargs['metrics']]
        )

    def __train(self, callbacks, x_train, y_train, x_test, y_test, epochs:int=10):
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[callbacks]
        )

    @report_done
    def train_model(self):
        """Builds and trains and LSTM model
        """
        self.__create_model(
            units = self.configs['TRAINING']['model']['units'],
            activations= self.configs['TRAINING']['model']['activations']
        )
        callbacks = self.__define_callbacks(
            **self.configs['TRAINING']['callbacks']
            )
        optimizer = self.__define_optimizer(
            **self.configs['TRAINING']['optimizer']
        )
        self.__compile(
            optimizer=optimizer,
            **self.configs['TRAINING']['compile']
        )
        self.__train(
            callbacks = callbacks,
            x_train = self.X_train,
            y_train = self.y_train,
            x_test = self.X_test,
            y_test = self.y_test,
            epochs=self.configs['TRAINING']['fit']['epochs']
        )
        filename = self.__get_filename_path(
            key='SAVE_MODEL_PATH'
        )
        self.__save_model(filename)
    
    def __save_model(self, filename: str):
        self.model.save(filename)
        
    def run(self):
        self.load_dataset()
        self.scale_input_data()
        self.scale_output_data()
        self.lstm_data_transform()
        self.split_data()
        self.train_model()

    def __call__(self) -> Any:
        self.run()
        return self.history
    
def plot_learning_curve(dict_val: dict,
                        values: List[str],
                        x_label: str,
                        y_label: str,
                        legend_list: List[str],
                        title: str,
                        legend_loc: str = 'upper left'
                        ) -> Any:
    """Plots the learning curves of the model.
    """
    for val in dict_val:
        if val in values:
            plt.plot(dict_val[val])

    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.legend(legend_list, loc=legend_loc)
    plt.show()

class DiagnosisTestingRegressionModel:
    """This class tests the Diagnosis Regression Model.
    """
    def __init__(self, configs: str) -> None:
        self.configs = parse_config(configs)
        self.feature_extraction = FeatureExtraction(**self.configs['FEATURE_EXTRACTION_CONFIGS'])
        self.__load_data()
        self.__load_model()
        self.__load_scalers()

    def __get_filename_path(self, key: str = 'SAVE_DATA_PATH') -> str:
        path = self.configs[key]['path'].copy()
        filename = self.configs[key]['filename']
        path.append(filename)

        return os.sep.join(path)

    def __load_model(self):
        filename = self.__get_filename_path(
            key='SAVE_MODEL_PATH'
        )
        self.model = load_model(filename)

    def __load_data(self) -> list:
        """Loads the data in pkl format
        """
        path = os.path.join(*self.configs['TESTING']['test_data']['path'])
        filename = self.configs['TESTING']['test_data']['filename']
        self.data = list()
        files = list()
        if filename:

            files.extend([os.path.join(path, filename)])

        else:

            path = os.path.join(*self.configs['TESTING']['test_data']['path']) + os.path.sep + '*.pkl'
            files = glob.glob(path)

        for file in files:
            print(file)
            self.data.extend(RackioAI.load(file))
            
        return self.data
    
    def __load_scalers(self):
        path = os.path.join(*self.configs['TESTING']['scalers']['path'])
        filename = self.configs['TESTING']['scalers']['filename']

        self.input_scaler = joblib.load(
            os.path.join(path, f"{filename}.inputScaler.gz")
        )
        self.output_scaler = joblib.load(
            os.path.join(path, f"{filename}.outputScaler.gz")
        )

    @staticmethod
    def __get_components(file: dict) -> list:
        return file['genkey']['Network Component'][1::]
    
    @staticmethod
    def __get_leak_info(components: list) -> tuple:
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

    def _make_predictions(self, file: dict):
        time_window = self.configs['TIME_WINDOW']
        leak_components = self.__get_components(file)
        leak_values, _ , leak_pos = self.__get_leak_info(components=leak_components)
        leak_point = leak_values[leak_pos[0]] * \
            60 / self.configs['SAMPLE_TIME']
        start_point = int(leak_point - \
                          time_window/self.configs['TIME_WINDOW_FRACTION'])
        end_point = int(leak_point + 60)

        df = file['tpl'].iloc[start_point:end_point]
        variable_to_predict = self.configs['OUTPUT_TO_USE']
        self.X = list()

        for point in range(0, df.shape[0] + 1):
            if point + time_window <= end_point - start_point:
                window = df.iloc[point:point + time_window]

                self.X.append(
                    np.concatenate(
                        self.__extract_features(window), axis=1)
                )

                if len(self.X) == 10:
                    
                    self.y.append(
                    window[
                        convert_list2tuple(
                            self.configs['OUTPUT_VARIABLES'][variable_to_predict]
                            )
                    ].iat[-1]
                    )
                    del window

                    self.X = np.concatenate(self.X, axis=0)
                    self.__scale_input_features()
                    self.__reshape_features()

                    self.y_pred.append(
                        self.__predict(
                            self.X
                        )
                    )
                    self.X = list()

    def __reshape_features(self):
        self.X = self.X.reshape(1, self.X.shape[0], self.X.shape[1])

    def __reshape_outputs(self):
        self.y = np.array(self.y).reshape(len(self.y), 1)
        self.y_pred = np.array(self.y_pred).reshape(len(self.y), 1)

    def __scale_input_features(self):
        self.X = self.input_scaler.transform(self.X)

    def __scale_output_inverse(self, y_pred: np.array) -> np.array:
        return self.output_scaler.inverse_transform(y_pred)

    def __predict(self, X: np.array) -> float:
        y_pred = self.model.predict(X)
        y_pred = self.__scale_output_inverse(y_pred)

        return y_pred[0][0]
    
    def _plot_results(self):
        """Plots the model's result vs original data.
        """
        configs = self.configs['TESTING']['plot']
        plt.plot(self.y, label=configs['y_label'])
        plt.plot(self.y_pred, label=configs['y_pred_label'])
        plt.xlabel(configs['xlabel'])
        plt.ylabel(configs['ylabel'])
        plt.title(configs['title'])
        plt.legend()
        plt.show()

    def run(self):
        """Runs the testing.
        """
        self.y = list()
        self.y_pred = list()
        _from = self.configs['TESTING']['cases']['from']
        _to = self.configs['TESTING']['cases']['to']

        for file in tqdm(self.data[_from:_to]):
            if len(file['tpl']) < 360:
                    continue
            
            self._make_predictions(file)
        
        self.__reshape_outputs()
        self._plot_results()

    def __call__(self) -> Any:
        self.run()


