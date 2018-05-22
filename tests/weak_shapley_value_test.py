import unittest
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

class TestWeakShapleyScorer(unittest.TestCase):

    def test_algorithm(self):
        # attribute_algorithm_test
        # evaluate_algorithm_test
        log = read_log("{}/data/weak_shapley_1.csv".format(DIR_PATH))
        scorer = WeakShapleyScorer()
        res = scorer.attribute(log)
        self.assertAlmostEqual(res["attribution"].sum(), log["score"].sum())
        self.assertAlmostEqual(scorer.dict_data["111"]["a"], 24.166666666)
        self.assertAlmostEqual(scorer.dict_data["111"]["b"], 105.8333333333333)
        self.assertAlmostEqual(scorer.dict_data["111"]["c"], 136.6666666666666)
        log["score"] = 100
        res2 = scorer.evaluate(log)
        self.assertAlmostEqual(res2["attribution"].sum(), log["score"].sum())


    def test_large_data(self):
        # many player test
        log = read_log("{}/data/weak_shapley_2.csv".format(DIR_PATH))
        scorer = WeakShapleyScorer()
        res = scorer.attribute(log)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
