from config import *
from data.training_data import TrainingData
from data.testing_data import TestingData
from ml.average import Average
import data.format as Format

TrainingData = TrainingData(TRAINING_DATA_CSV)
TestingData = TestingData(TESTING_DATA_CSV)

clf = Average()

clf.train(TrainingData.data)
predictions = clf.test(TestingData.data)

Format.export_predictions(predictions)
