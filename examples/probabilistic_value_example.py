import pandas
import os
DIR_PATH = os.path.dirname(__file__)
import sys
if DIR_PATH == "":
    DIR_PATH = "."
sys.path.append("{}/../".format(DIR_PATH))
from scoring.probabilistic_scorer import ProbabilisticScorer


def read_log(file_path):
    df = pandas.read_csv(
        file_path, dtype={
            "player": object, "action": object,
            "step": "int64", "score": "float64"
        }
    )
    return df


if __name__ == "__main__":
    ps = ProbabilisticScorer()
    log = read_log("{}/../dev_data/probabilistic.csv".format(DIR_PATH))
    r = ps.attribute(log)
    ps.to_csv("attribution.csv")
    ps.to_json("evaluation.json")
    print(r)
    print(ps.df)
