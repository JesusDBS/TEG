import json
from typing import Any

def parse_config(path: str = 'config.json') -> dict:
    """Parses config json to Python dict
    """
    assert isinstance(path, str), f'Jason path: {path} must be a string!'

    try:

        with open(path, "r") as j:
            config_json = json.load(j)

        return config_json

    except:

        raise FileNotFoundError("Json file not found")
    
def convert_list2tuple(array: list=[]) -> tuple | Any:
    """Converts a list into tuple
    """
    if not array:
        return None
    
    return tuple(array)
