from pathlib import Path
import pandas as pd
import yaml
def read_csv_data(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("Input a CSV file path")
    return pd.read_csv(Path(path))

def parse_yaml():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
    
def split_input_output_df(struct, problem_df):
    return problem_df[struct["inputs"]], problem_df[struct["output"]]
