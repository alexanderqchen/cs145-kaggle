from config import *
import pandas
from data.training_data import TrainingData
from data.testing_data import TestingData
from ml.movie_avg import MovieAverage
import data.format as Format

training_data = pandas.read_csv(TRAINING_DATA_CSV)
testing_data = pandas.read_csv(TESTING_DATA_CSV)

clf = MovieAverage()

clf.train(training_data)
predictions = clf.test(testing_data)
print(predictions)
predictions.to_csv('csv/predictions.csv', index=False)