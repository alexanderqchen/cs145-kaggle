import pandas as pd

class MovieAverage:
	def train(self, data):
		self.avg = data.groupby(['movieId'])['rating'].mean()

	def test(self, data):
		predictions = pd.DataFrame(columns=['id', 'rating'])

		for index, row in data.iterrows():
			id = row['Id']
			movieId = row['movieId']
			prediction = self.avg[movieId] if movieId in self.avg else 3.535071681

			prediction_df = pd.DataFrame({'id': id, 'rating': prediction}, index=[0])

			predictions = predictions.append(prediction_df, ignore_index=True)
			print(id)

		return predictions
