import joblib
import json
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from utils import to_labels


if __name__ == "__main__":
    with open("./config/parameters.json") as fh:
        d = json.load(fh)

    # read dataset
    filename = d["filename"]
    path_dataset = d["path_dataset"]
    no_overview_pattern = re.compile(d["no_overview_pattern"], re.IGNORECASE)  # regex to match invalid overview entries
    movies_df = pd.read_csv(path_dataset + filename, low_memory=False)[["title", "overview", "genres"]]
    # drop nan titles and overviews
    movies_df = movies_df[~movies_df["title"].isna()]
    movies_df = movies_df[~movies_df["overview"].isna()]
    movies_df = movies_df[~movies_df["overview"].str.match(no_overview_pattern)]  # drop rows with invalid overview

    has_genres_mask = movies_df["genres"] != "[]"
    genres = movies_df["genres"][has_genres_mask]  # drop empty genres
    genres_strings = genres.apply(to_labels)
    mlb = MultiLabelBinarizer()
    mlb.fit(genres_strings)
    # pick only annotated overview and titles
    x = movies_df["title"][has_genres_mask] + ' ' + movies_df["overview"][has_genres_mask]
    y = mlb.transform(genres_strings)  # encode multi-label annotations

    top_classes = sorted(list(zip(y.sum(axis=0), mlb.classes_)))[::-1]  # get the most present classes
    top_classes = sorted([tc[1] for tc in top_classes][1:10])  # drop `Drama` samples due to class imbalance

    top_mlb = MultiLabelBinarizer(classes=top_classes)
    top_mlb.fit(genres_strings)
    y = top_mlb.transform(genres_strings)  # encode the top classes
    no_labels_mask = y.sum(axis=1) == 0  # drop the samples which classes do not appear in `top_classes`
    x = x[~ no_labels_mask]
    y = y[~ no_labels_mask]
    # split dataset into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    tfidf = TfidfVectorizer(stop_words="english", lowercase=True, max_features=1000)
    mlp = MLPClassifier(warm_start=True, max_iter=5, hidden_layer_sizes=10)
    steps = [("transform", tfidf), ("estimator", mlp)]
    model = Pipeline(steps)
    # model training
    num_epochs = d["train"]["num_epochs"]
    for i in range(num_epochs):
        model.fit(x_train, y_train)
        print('epoch {0:3d} -- train {1:.3f} -- test {2:.3f}'.format(i,
                                                                     model.score(x_train, y_train),
                                                                     model.score(x_test, y_test)))

    # serialize the trained model and the top label binarizer to file
    path_model = d["train"]["path_model"]
    model_name = d["train"]["model_name"]
    binarizer_name = d["train"]["label_binarizer"]

    joblib.dump(model, path_model + model_name)
    joblib.dump(top_mlb, path_model + binarizer_name)
