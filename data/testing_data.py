import csv

class TestingData:
	def __init__(self, filename):
		self.data = pandas.read_csv(filename)