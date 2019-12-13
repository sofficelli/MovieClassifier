# MovieClassifier
This is the public repository for the MovieClassifier challange.
In order to reproduce the results, make sure that `movies_dataset.csv` is present in folder `dataset`. If not, 
download it from the [MovieLens](https://www.kaggle.com/rounakbanik/the-movies-dataset) dataset. Your directory tree should look like this: 
``` 
.
├── config
│   ├── parameters.json
│   └── requirements.txt
├── dataset
│   └── movies_metadata.csv
├── main.py
├── models
│   ├── label_binarizer.joblib
│   └── movie_classifier.joblib
├── README.md
├── train.py
└── utils.py
```

## Dependencies
In order to reproduce the code, create a virtual environment in the directory root

```virtualenv -m python3 <your_virtualenv_name>```

Then, activate the virtual environment

```source <your_virtualenv_name>/bin/activate```

Finally, use `config/requirements.txt` to install all the required dependencies using `pip`

```pip install -r config/requirements.txt```

## Usage Example
Make sure the file `movie_classifier` is executable, if not

```chmod +x movie_classifer```

Then, to compute the genre prediction for the input title and description

```movie_classifier --title "Mega Python vs. Gatoroid" --description "A fanatical animal rights activist (Debbie Gibson) releases giant pythons into the Everglades, believing the wild animals should be set free. When they start decimating the native animal population, an over-zealous park ranger (Tiffany) feeds experimental steroids to wild alligators so they can fight back. The giant pythons and gargantuan alligators go on a killing spree, and it is now left up to the two feuding women to put aside their differences to put a stop to the creatures and the destruction."```

Note, if you execute the scripts from interpreter, you need to activate the virtual environment as explained above.

## Algorithm and Pre-processing 
`movie_classifier` implements a Multilayer Perceptron (MLP) using [scikit-learn](https://scikit-learn.org/stable/).
MLP with TF-IDF measure provides reasonable predictions for the genre to the input titles and descriptions. 
The dataset is heavily imbalanced in favor of the `Drama` genre, so removing this class from the label space dramatically
improves predictions. Moreover, the label space has been further reduced to the top 9 classes present in the dataset,
as some combinations of genres are really rare to occur.
In the pre-processing phase, `nan` and empty values are removed from the dataset. Text strings are lower cased and 
english stopwords are filtered out before training.

### Parameters
Parameters for training and tests are defined in the configuration file `parameters.json` located in `config`.

### Unit Testing
Auxiliary functions are defined in `utils.py` and are unit tested when directly executed. 
Furthermore, `test_prediction` is defined to test the model against the movie defined in `config/parameters.json`.
