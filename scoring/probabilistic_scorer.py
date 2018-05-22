from . import converter
from collections import Counter


class ProbabilisticScorer(converter.Converter):

    def __init__(self):
        super().__init__()
        self.action_list = []

    def attribute(self, X):
        X = X.copy()
        self.action_list = sorted(set(X["action"]))
        player_w = set(X[X["score"] == 1]["player"])
        action_w = X[X["player"].apply(lambda x: x in player_w)]["action"]
        counter_w = Counter(action_w)
        counter_all = Counter(X["action"])

        w = [counter_w[action] / counter_all[action]
             for action in self.action_list]
        weight_dict = {k: weight / sum(w)
                       for k, weight in zip(self.action_list, w)}
        A = self._attribute(X, weight_dict, player_w)
        self.df = A
        self.dict_data = weight_dict
        return A

    def evaluate(self, X):
        player_w = set(X[X["score"] == 1]["player"])
        A = self._attribute(X, self.dict_data, player_w)
        return A

    @staticmethod
    def _attribute(X, dict_data, player_w):
        X = X.copy()
        X["win"] = X["player"].apply(lambda x: 1 if x in player_w else 0)
        X["weight"] = X["action"].apply(lambda x: dict_data.get(x, 0))
        X = X.merge(
            X.groupby("player")["weight"].sum().reset_index().rename(
                columns={"weight": "sum_weight"}),
            on="player", how="inner", copy=False
        )
        X["attribution"] = (X["win"] * X["weight"] / X["sum_weight"]).fillna(0)
        X = X.drop(["win", "weight", "sum_weight"], axis=1)
        return X
