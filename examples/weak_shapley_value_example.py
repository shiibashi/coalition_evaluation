import pandas
import os
DIR_PATH = os.path.dirname(__file__)
import sys
if DIR_PATH == "":
    DIR_PATH = "."
sys.path.append("{}/../".format(DIR_PATH))
from scoring.weak_shapley_scorer import WeakShapleyScorer


def read_log(file_path):
    df = pandas.read_csv(
        file_path, dtype={
            "player": object, "action": object,
            "step": "int64", "score": "float64"
        }
    )
    return df


if __name__ == "__main__":
    ws = WeakShapleyScorer()
    log = read_log("{}/../dev_data/shapley.csv".format(DIR_PATH))
    r = ws.attribute(log)
    ws.to_csv("attribution.csv")
    ws.to_json("evaluation.json")
    print(r)
    print(ws.df)
