# MovieClassifier
This is the public repository for the MovieClassifier challange.
In order to reproduce the results, download the [MovieLens](https://www.kaggle.com/rounakbanik/the-movies-dataset) dataset
and store `movies_dataset.csv` in the folder `dataset`. Your directory tree should look like this: 
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
```python main.py --title "Mega Python vs. Gatoroid" --description "A fanatical animal rights activist (Debbie Gibson) releases giant pythons into the Everglades, believing the wild animals should be set free. When they start decimating the native animal population, an over-zealous park ranger (Tiffany) feeds experimental steroids to wild alligators so they can fight back. The giant pythons and gargantuan alligators go on a killing spree, and it is now left up to the two feuding women to put aside their differences to put a stop to the creatures and the destruction."```