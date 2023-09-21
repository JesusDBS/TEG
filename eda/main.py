import os
import pandas as pd
from utils import load_pkl, plot_data, make_datas_list
from typing import Any

def main(
    path_to_data: str, 
    variables_to_plot: list = [], 
    subplot: bool = False,
    _to: int|bool = False, 
    _from: int = 0,
    slice_start: int = 0,
    slice_end: int | Any = None,
    ):
    r"""
    Main EDA function
    """

    df = load_pkl(
        path=path_to_data,
        _to = _to,
        _from = _from
        )
    
    if variables_to_plot:
        variables_to_plot = make_datas_list(
            df=df, variables_to_plot=variables_to_plot, _from=slice_start, _to=slice_end)

        plot_data(*variables_to_plot, subplot=subplot)


if __name__ == "__main__":

    EE_leak_data_path = os.path.join(
        "..", "data", "data", "No Fuga", "CV_ET_D0_R0.pkl"
    )

    variables_to_plot = [
        ('PT_POSITION_POS1378M', 'Pressure', 'PA'),
        ('GT_POSITION_POS1378M', 'Total_mass_flow', 'KG/S'),
        # ('GTLEAK_LEAK_LEAK', 'Leakage_total_mass_flow_rate', 'KG/S'),
        # ('CONTR_CONTROLLER_CONTROLLEAK', 'Controller_output', '')
        # ('STROKE', 'Stroke', 'S')
        # ('VALVOP_CHOKE_VOUT', 'Relative_valve_opening', '')
        ('VALVE_OPENING', 'Valve opening', 'CM')
    ]

    CASES_FROM = 1
    CASES_TO = 2

    SLICE_START = 10
    SLICE_END = 150

    main(
        path_to_data=EE_leak_data_path,
        variables_to_plot=variables_to_plot,
        subplot=True,
        _to = CASES_TO,
        _from = CASES_FROM,
        # slice_start = SLICE_START,
        # slice_end = SLICE_END
    )
