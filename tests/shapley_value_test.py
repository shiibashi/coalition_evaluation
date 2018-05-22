import unittest
import pandas
import os
DIR_PATH = os.path.dirname(__file__)
import sys
if DIR_PATH == "":
    DIR_PATH = "."
sys.path.append("{}/../".format(DIR_PATH))
from scoring.shapley_scorer import ShapleyScorer


def read_log(file_path):
    df = pandas.read_csv(
        file_path, dtype={
            "player": object, "action": object,
            "step": "int64", "score": "float64"
        }
    )
    return df

class TestShapleyScorer(unittest.TestCase):

    def test_algorithm(self):
        # attribute_algorithm_test
        # evaluate_algorithm_test
        log = read_log("{}/data/shapley_1.csv".format(DIR_PATH))
        scorer = ShapleyScorer()
        res = scorer.attribute(log)
        self.assertAlmostEqual(res["attribution"].sum(), log["score"].sum())
        self.assertAlmostEqual(scorer.dict_data["111"]["a"], 12.5)
        self.assertAlmostEqual(scorer.dict_data["111"]["b"], 47.5)
        self.assertAlmostEqual(scorer.dict_data["111"]["c"], 60)
        log["score"] = 100
        res2 = scorer.evaluate(log)
        self.assertAlmostEqual(res2["attribution"].sum(), log["score"].sum())


    def test_large_data(self):
        # many player test
        log = read_log("{}/data/shapley_2.csv".format(DIR_PATH))
        scorer = ShapleyScorer()
        res = scorer.attribute(log)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
