import csv

class TrainingData:
	"""
	self.data is a list of dictionaries of the form
	{
		'userId': int,
		'movieId': int,
		'rating': float
	}

	Note: movieId is stored as float in the csv, but is treated as an int
	"""
	def __init__(self, filename):
		self.filename = filename
		self.data = self.__parse_csv()

	def __parse_csv(self):
		train_data = []

		with open(self.filename) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			next(csv_reader)

			for row in csv_reader:
				userId = int(row[0])
				movieId = int(float(row[1]))
				rating = float(row[2])

				instance = {
					'userId': userId,
					'movieId': movieId,
					'rating': rating
				}

				train_data.append(instance)

		return train_data