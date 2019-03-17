import numpy as np

class MovieAverage:
	def train(self, data):
		self.avg = data.groupby(['movieId'])['rating'].mean()

	def test(self, data):
		predictions = np.zeros((data.shape[0], 2))

		for index, row in data.iterrows():
			id = int(row['Id'])
			movieId = row['movieId']
			prediction = self.avg[movieId] if movieId in self.avg else 3.535071681

			predictions[id][0] = id
			predictions[id][1] = prediction

			print(id)


		print(predictions)

		return predictions
