import pandas
import os
DIR_PATH = os.path.dirname(__file__)
import sys
if DIR_PATH == "":
    DIR_PATH = "."
sys.path.append("{}/../".format(DIR_PATH))
from scoring.banzhaf_scorer import BanzhafScorer


def read_log(file_path):
    df = pandas.read_csv(
        file_path, dtype={
            "player": object, "action": object,
            "step": "int64", "score": "float64"
        }
    )
    return df


if __name__ == "__main__":
    bs = BanzhafScorer()
    log = read_log("{}/../dev_data/banzhaf.csv".format(DIR_PATH))
    r = bs.attribute(log)
    bs.to_csv("attribution.csv")
    bs.to_json("evaluation.json")
    print(r)
    print(bs.df)
