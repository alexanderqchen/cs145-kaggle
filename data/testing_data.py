import csv

class TestingData:
	"""
	self.data is a list of dictionaries of the form
	{
		'id': int,
		'userId': int,
		'movieId': int
	}

	Note: movieId is stored as float in the csv, but is treated as an int
	"""
	def __init__(self, filename):
		self.filename = filename
		self.data = self.__parse_csv()

	def __parse_csv(self):
		test_data = []

		with open(self.filename) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			next(csv_reader)

			for row in csv_reader:
				id = int(row[0])
				userId = int(row[1])
				movieId = int(float(row[2]))

				instance = {
					'id': id,
					'userId': userId,
					'movieId': movieId,
				}

				test_data.append(instance)

		return test_data