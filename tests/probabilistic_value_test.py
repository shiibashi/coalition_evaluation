import unittest
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

class TestProbabilisticScorer(unittest.TestCase):

    def test_algorithm(self):
        # attribute_algorithm_test
        # evaluate_algorithm_test
        log = read_log("{}/data/probabilistic_1.csv".format(DIR_PATH))
        scorer = ProbabilisticScorer()
        res = scorer.attribute(log)
        self.assertAlmostEqual(res["attribution"].sum(), log["score"].sum())
        self.assertAlmostEqual(scorer.dict_data["item6"], 0)
        self.assertAlmostEqual(scorer.dict_data["item5"], scorer.dict_data["item4"])
        self.assertGreater(scorer.dict_data["item4"], scorer.dict_data["item2"])
        self.assertGreater(scorer.dict_data["item2"], scorer.dict_data["item1"])
        self.assertGreater(scorer.dict_data["item1"], scorer.dict_data["item3"])
        self.assertAlmostEqual(sum(list(scorer.dict_data.values())), 1)
        res2 = scorer.evaluate(log[0:3])
        for a1, a2 in zip(res["attribution"][0:3], res2["attribution"]):
            self.assertAlmostEqual(a1, a2)

    def test_large_data(self):
        # many player test
        log = read_log("{}/data/probabilistic_2.csv".format(DIR_PATH))
        scorer = ProbabilisticScorer()
        res = scorer.attribute(log)
        self.assertAlmostEqual(res["attribution"].sum(), log["score"].sum())

    def test_large_data2(self):
        # many player and item test
        log = read_log("{}/data/probabilistic_3.csv".format(DIR_PATH))
        scorer = ProbabilisticScorer()
        res = scorer.attribute(log)
        self.assertAlmostEqual(res["attribution"].sum(), log["score"].sum())
        self.assertAlmostEqual(len(scorer.dict_data), 6006)        


if __name__ == "__main__":
    unittest.main()
