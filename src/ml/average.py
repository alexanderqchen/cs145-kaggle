class Average:
	def train(self, data):
		total_rating = 0
		count = 0

		for instance in data:
			count += 1
			total_rating += instance['rating']

		self.average = total_rating / count

	def test(self, data):
		predictions = []

		for instance in data:
			prediction = {
				'id': instance['id'],
				'rating': self.average
			}

			predictions.append(prediction)

		return predictions
