import argparse
import joblib
import json


parser = argparse.ArgumentParser(prog="movie_classifier",
                                 description="Classify the genre of the input movie based on the title and description provided.")
parser.add_argument("--title", required=True, metavar='T', type=str, nargs=1, help="title of the input movie")
parser.add_argument("--description", required=True, metavar='D', type=str, nargs=1, help="brief description of the input movie")

# parse the arguments from command line
args = parser.parse_args()
title = args.title[0]
descr = args.description[0]

if title == '' or descr == '':
    raise ValueError("Input arguments cannot be the empty string.")

with open("./config/parameters.json") as fh:
    d = json.load(fh)

path_model = d["train"]["path_model"]
model_name = d["train"]["model_name"]
label_binarizer_name = d["train"]["label_binarizer"]
model = joblib.load(path_model + model_name)  # load the trained model
label_binarizer = joblib.load(path_model + label_binarizer_name)  # load the serialized multi-label binarizer

# compute prediction using input arguments
pred = model.predict([title + ' ' + descr])
output = label_binarizer.classes_[pred.ravel().astype("bool")]

# reformat output as a JSON string
movie = {"title": title,
         "description": descr,
         "genre": ", ".join(output)}  # adjust genre string according to specifications
movie_str = json.dumps(movie, indent=4)

print(movie_str)
