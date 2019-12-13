import ast
import joblib
import json
import re
import unittest


def to_labels(genres_list: str):
    """ Convert the input string to a list of genre strings.

    """
    genres_list = ast.literal_eval(genres_list)
    return [g["name"] for g in genres_list]


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        with open("config/parameters.json") as fh:  # load test strings from .JSON file
            d = json.load(fh)

        self.genres_str = d["test"]["genres_string"]
        self.genres_list = d["test"]["genres_list"]
        self.description = d["test"]["overview"]
        self.no_overview_strings = d["no_overview_strings"]
        self.no_overview_pattern = d["no_overview_pattern"]

        self.path_model = d["train"]["path_model"]
        self.model_name = d["train"]["model_name"]
        self.label_binarizer_name = d["train"]["label_binarizer"]

    def test_genres_str_to_list(self):
        self.assertEqual(to_labels(self.genres_str), self.genres_list)

    def test_invalid_overview_pattern(self):
        pattern = re.compile(self.no_overview_pattern)
        outcome = []
        for io in self.no_overview_strings:
            if re.match(pattern, io):
                outcome.append(True)

        self.assertTrue(sum(outcome), len(self.no_overview_strings))

    def test_prediction(self):
        model = joblib.load(self.path_model + self.model_name)
        label_binarizer = joblib.load(self.path_model + self.label_binarizer_name)
        pred = model.predict([self.description])
        output = label_binarizer.classes_[pred.ravel().astype("bool")]

        self.assertEqual(output.tolist(), self.genres_list)


if __name__ == "__main__":
    unittest.main()
