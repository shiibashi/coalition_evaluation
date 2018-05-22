from . import converter
import math
import pandas


class BanzhafScorer(converter.Converter):

    def __init__(self):
        super().__init__()
        self.action_list = []
        self.cf = {}

    def attribute(self, X):
        self._update_matrix(X)
        self._update_characteristic_function()
        power_set = super().power_set(self.action_list)
        attribution_dict = {self._key(s): self._value_dict(s)
                            for s in power_set}
        self.dict_data = attribution_dict
        X["attribution"] = self._distribute(X).copy()
        return X

    def evaluate(self, X):
        X["attribution"] = self._distribute(X).copy()
        return X

    def _distribute(self, X):
        X = X.copy()
        X["key_int"] = X["action"].apply(lambda x: int(self._key(set(x)), 2))
        X_sum = X.groupby("player")["key_int", "score"
                                    ].sum().reset_index().rename(columns={"score": "score_sum"})
        X_sum["key"] = X_sum["key_int"].apply(
            lambda x: bin(x).split("0b")[1].zfill(len(self.action_list)))
        res = X.merge(X_sum, on="player", how="inner")
        return pandas.Series(list(self._generate_attribution(res)))

    def _generate_attribution(self, df):
        for i, row in df.iterrows():
            action = row["action"]
            key = row["key"]
            attribution_sum = sum(list(self.dict_data[key].values()))
            score_sum = row["score_sum"]
            if attribution_sum == 0:
                attribution = 0
            else:
                attribution = score_sum * \
                    (self.dict_data[key].get(action, 0) / attribution_sum)
            yield attribution

    def _update_matrix(self, X):
        log_matrix = super().log_to_matrix(X).groupby(
            "player").sum().reset_index(drop=True)
        self.action_list = list(log_matrix.columns[:-1])
        matrix = log_matrix.groupby(by=self.action_list)["score"].agg(
            {"mean": "mean"}).reset_index()
        self.df = matrix

    def _update_characteristic_function(self):
        for i, row in self.df.iterrows():
            binary_string = "".join(
                row[self.action_list].apply(lambda x: str(int(x))).values)
            self.cf[binary_string] = row["mean"]

    def _value_dict(self, action_set):
        power_set = super().power_set(action_set)
        return {action: self._value(action, power_set) for action in action_set}

    def _value(self, action, power_set):
        p = len(max(power_set))
        exclude_set = [s for s in power_set if action not in s]
        include_set = [s | set({action}) for s in exclude_set]
        v = 0
        for se, si in zip(exclude_set, include_set):
            binary_string_se = self._key(se)
            v_se = self.cf.get(binary_string_se, 0)
            binary_string_si = self._key(si)
            v_si = self.cf.get(binary_string_si, 0)
            c = (1 / 2) ** (p - 1)
            v += c * (v_si - v_se)
        return v

    def _key(self, action_set):
        key = "".join(
            ["0" if a not in action_set else "1" for a in self.action_list])
        return key
