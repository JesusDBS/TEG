import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from rackio_AI import RackioAI


def load_pkl(
        path: str, 
        _to: int|bool = False, 
        _from:int=0,
        join:bool=False
        ) -> pd.DataFrame | list:
    r"""
    This function loads the data in pkl format.

    Parameters
    ----------
    path:str
        Path to the data.

    Returns
    -------
    Pandas dataframe.
    """
    app = RackioAI

    datalist = app.load(pathname=path, join_file=False)

    if join:
        return datalist

    if isinstance(_to, int) and _to > _from:
        datalist = datalist[_from:_to]

    series = list()
    for el in datalist:
        df = el['tpl']
        series.append(el['tpl'])

    df = pd.concat(series, axis=0)

    df = df.reset_index(drop=True)

    return df


def plot_data(*datas, subplot: bool = False) -> None:
    r"""
    This function plots both the original and the filtered data

    Parameters
    ----------
    datas: 
        list of data to be plotted.

    Returns
    -------
    None.
    """
    if subplot:
        _, axes = plt.subplots(len(datas))

        for num, data in enumerate(datas):
            axes[num].plot(data)
    else:
        for data in datas:
            plt.plot(data)

    plt.show()


def make_datas_list(
        df: pd.DataFrame, 
        variables_to_plot: list = [], 
        _to: int | Any = None, 
        _from: int=0) -> list:
    r"""
    This function returns a list of pd.Series from each variable to plot.
    """
    if variables_to_plot:
        return [df[variable][_from:_to] for variable in variables_to_plot] if _to and _from != 0\
                else [df[variable] for variable in variables_to_plot]

    return None