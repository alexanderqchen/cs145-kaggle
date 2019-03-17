from config import *
import pandas
from data.training_data import TrainingData
from data.testing_data import TestingData
from ml.both_avg import BothAverage
import data.format as Format
import numpy as np

training_data = pandas.read_csv(TRAINING_DATA_CSV)
testing_data = pandas.read_csv(TESTING_DATA_CSV)

clf = BothAverage()

clf.train(training_data)
predictions = clf.test(testing_data)

np.savetxt('csv/predictions.csv', predictions, header='id,rating', delimiter=",", fmt=['%i', '%f'])
