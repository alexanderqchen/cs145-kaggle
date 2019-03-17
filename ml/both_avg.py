import numpy as np

class BothAverage:
	def train(self, data):
		self.movie_avg = data.groupby(['movieId'])['rating'].mean()
		self.user_avg = data.groupby(['userId'])['rating'].mean()

	def test(self, data):
		predictions = np.zeros((data.shape[0], 2))

		for index, row in data.iterrows():
			id = int(row['Id'])
			movieId = row['movieId']
			userId = row['userId']
			movie_prediction = self.movie_avg[movieId] if movieId in self.movie_avg else 3.535071681
			user_prediction = self.user_avg[userId] if userId in self.user_avg else 3.535071681

			prediction = (movie_prediction + user_prediction) / 2

			predictions[id][0] = id
			predictions[id][1] = prediction

		return predictions
