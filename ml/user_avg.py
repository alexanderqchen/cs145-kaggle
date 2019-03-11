class MovieAverage:
	def train(self, data):
		self.avg = data.groupby(['userId'])['rating'].mean()

	def test(self, data):
		predictions = []

		for index, row in data.iterrows():
			userId = row['userId']
			prediction = self.avg[userId] if userId in self.avg else 3.535071681

			print(f"user {userId} got rating {prediction}")

			predictions.append(prediction)

		return predictions
