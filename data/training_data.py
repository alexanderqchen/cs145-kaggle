import pandas

class TrainingData:
	def __init__(self, filename):
		self.data = pandas.read_csv(filename)