import os
import sys
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# This code line is to avoid import relative error
sys.path.append("..")

from utils import parse_config, report_done

class DetectionTrainingModelPipeline:
    """This class trains LSTM classification model for detection.
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

            if layer == len(units):
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
        self.lstm_data_transform()
        self.split_data()
        self.train_model()

    def __call__(self) -> Any:
        self.run()
        return self.history

