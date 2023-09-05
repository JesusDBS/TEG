import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pywt, pickle


class Stats:
    r"""
    Documentation here
    """

    def __init__(self, features:list=['kurt', 'skew', 'std', 'crest_factor']):

        self.features = features

    def __call__(self, data:np.array):
        r"""
        Documentation here
        """
        stats_features = list()

        for stat_feature in self.features:

            fn = getattr(self, stat_feature)
            stats_features.append(fn(data))

        result = np.array(stats_features)
        result = result.reshape(1, len(result))

        return result

    def kurt(self, data:np.array):
        r"""
        Documentation here
        """
        return kurtosis(data, axis=0)
    
    def mean(self, data:np.array):
        r"""
        Documentation here
        """
        return np.mean(data, axis=0)
    
    def std(self, data:np.array):
        r"""
        Documentation here
        """
        return np.std(data, axis=0)
    
    def skew(self, data:np.array):
        r"""
        Documentation here
        """
        return skew(data, axis=0)
    
    def rms(self, data:np.array):
        r"""
        Root mean squared
        """
        return (np.sum(data ** 2, axis=0) / len(data)) ** 0.5
    
    def peak2valley(self, data:np.array):
        r"""
        Documentation here
        """
        return (np.max(data, axis=0) - np.min(data, axis=0)) / 2
    
    def peak(self, data:np.array):
        r"""
        Documentation here
        """
        return np.max(data - data[0], axis=0)
    
    def crest_factor(self, data:np.array):
        r"""
        Documentation her
        """
        peak = self.peak(data)
        rms = self.rms(data)
        return peak / rms
    
    def serialize(self):
        r"""
        Documentation here
        """
        return {
            "stats": True,
            "stats_features": self.features
        }

    
class Wavelet:
    r"""
    Documentation here
    """

    def __init__(self, wavelet_type:str='db2', lvl:int=3, stats:Stats=None):

        self.wavelet_type = wavelet_type
        self.wavelet_lvl = lvl
        self.last_coeffs = self.wavelet_lvl
        self.stats = stats

    def __call__(self, data:np.array):
        r"""
        Documentation here
        """
        _coeffs = list()
        coeffs = pywt.wavedec(data, self.wavelet_type, level=self.wavelet_lvl, axis=0)

        if self.stats:

            _coeffs.extend([self.stats(np.diff(coeff)) for coeff in coeffs[0:self.last_coeffs]])
            result = np.concatenate(_coeffs, axis=1)[0]

        else:

            _coeffs.extend([coeff for coeff in coeffs[0:self.last_coeffs]])
            result = np.concatenate(_coeffs, axis=0)

        result = result.reshape(1, len(result))

        return result
    
    def serialize(self):
        r"""
        Documentation here
        """
        return {
            "wave": True,
            "wave_stats": True if self.stats else False,
            "wavelet_type": self.wavelet_type,
            "wavelet_lvl": self.wavelet_lvl
        }


class FeatureExtraction:
    r"""
    Documentation here
    """

    def __init__(
            self,
            stats:bool=False,
            stats_features:list=['kurt', 'skew', 'std', 'crest_factor'],
            wave:bool=False,
            wave_stats:bool=False,
            wavelet_type:str='db2',
            wavelet_lvl:str=3,
            time_window:int=10,
            variables_names:list=[]
            ):
        self.statistical = stats
        self.wave_stats = wave_stats
        self.wave = wave
        self.variables_names = variables_names
        self.time_window = time_window
        _stats = Stats(features=stats_features)
        if self.statistical:
            
            self.stats = _stats

        if self.wave:
            
            if self.wave_stats:

                self.wavelet = Wavelet(wavelet_type=wavelet_type, lvl=wavelet_lvl, stats=_stats)

            else:

                self.wavelet = Wavelet(wavelet_type=wavelet_type, lvl=wavelet_lvl)

    def append_variable_name(self, name:str):
        r"""
        Documentation here
        """
        if name not in self.variables_names:

            self.variables_names.append(name)

    def set_variables_names(self, variables_names:list):
        r"""
        Documentation here
        """
        self.variables_names = variables_names

    def __call__(self, data:np.array)->np.array:
        r"""
        Documentation here
        """
        features = list()

        if self.statistical:
            
            features.append(self.stats(data))

        if self.wave:
            
            features.append(self.wavelet(data))

        return np.concatenate(features, axis=1)
    
    def save(self, filename:str):
        r"""
        Documentation here
        """
        with open(f'{filename}.fe', 'wb') as file:

            pickle.dump(self.serialize(), file)

    @classmethod
    def load(cls, filename:str):
        r"""
        Documentation here
        """
        with open(f'{filename}', 'rb') as file:

            feature_config = pickle.load(file)

        return cls(**feature_config)
    
    def serialize(self):
        r"""
        Documentation here
        """
        result = {}
        if self.statistical:
            result = {
                **result,
                **self.stats.serialize()
            }

        if self.wave:

            result ={
                **result,
                **self.wavelet.serialize(),
            }

            if self.wave_stats:

                result = {
                    **result,
                    "stats_features": self.wavelet.stats.features
                }

        return {
            **result,
            'variables_names': self.variables_names,
            'time_window': self.time_window
        }