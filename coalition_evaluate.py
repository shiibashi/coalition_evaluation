import pandas
from argparse import ArgumentParser
from scoring.probabilistic_scorer import ProbabilisticScorer
from scoring.shapley_scorer import ShapleyScorer
from scoring.weak_shapley_scorer import WeakShapleyScorer
from scoring.banzhaf_scorer import BanzhafScorer
#from scoring.markov_scorer import MarkovScorer
#from scoring.directed_shapley_scorer import DirectedShapleyScorer
#from scoring.pvp_scorer import PVPScorer


def parser():
    p = ArgumentParser(
        prog="coalition_evaluation",
        usage="python coalition_evaluate.py"
              + " -f {input_file_path}"
              + " -m {method}"
              + " -o {output_file_path}",
        add_help=True
    )
    p.add_argument(
        "-f", "--file",
        help="input file path",
        required=True
    )
    p.add_argument(
        "-m", "--method",
        help="evalution method",
        required=True,
        choices=["probabilistic", "markov", "shapley", "weakShapley",
                 "banzhaf", "pvp", "directedShapley"]
    )
    p.add_argument(
        "-o", "--out",
        help="output directory path",
        required=False,
        default="./"
    )
    return p


def read_log(file_path):
    df = pandas.read_csv(
        file_path, dtype={
            "player": object, "action": object,
            "step": "int64", "score": "float64"
        }
    )
    return df


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    log = read_log(args.file)
    method = args.method
    out = args.out

    if method == "probabilistic":
        scorer = ProbabilisticScorer()
    elif method == "shapley":
        scorer = ShapleyScorer()
    elif method == "weakShapley":
        scorer = WeakShapleyScorer()
    elif method == "banzhaf":
        scorer = BanzhafScorer()
    elif method == "markov":
        scorer = MarkovScorer()
    elif method == "directedShapely":
        scorer = DirectedShapleyScorer()
    elif method == "pvp":
        scorer = PVPScorer()

    scorer.attribute(log)
    scorer.to_csv("{}/attribution.csv".format(out))
    scorer.to_json("{}/evaluation.json".format(out))
