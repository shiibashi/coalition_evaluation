import pandas
import json
from itertools import combinations
from abc import abstractmethod


class Converter(object):

    def __init__(self):
        self.df = pandas.DataFrame()
        self.dict_data = dict()

    @abstractmethod
    def attribute(self, X):
        """attribution method
            Args:
                X (pandas.DataFrame): log dataframe
        """
        pass

    @abstractmethod
    def evaluate(self, X):
        """evaluation method
            Args:
                X (pandas.DataFrame): log dataframe
        """
        pass

    def log_to_matrix(self, log):
        log = log.copy()
        action_list = sorted(set(log["action"]))
        for action in action_list:
            log[action] = log["action"].apply(
                lambda x: 1 if x == action else 0)
        matrix = log[action_list+["score", "player"]]
        return matrix

    def to_csv(self, file_name):
        self.df.to_csv(file_name, index=False)

    def to_json(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.dict_data, f, indent=4)

    @staticmethod
    def power_set(columns):
        power_set = [set(s) for i in range(len(columns)+1)
                     for s in combinations(columns, i+1)]
        power_set.append(set({}))
        return power_set
