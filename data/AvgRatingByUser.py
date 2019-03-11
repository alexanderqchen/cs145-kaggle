import csv
import pandas as pd
import numpy as np
import time

AVG_RATING = 3.535071681

class AvgRatingByUser:
	def __init__(self):
		self.df = pd.read_csv("./csv/train_ratings.csv")
		self.df['userId'] = self.df['userId'].apply(lambda x: int(x))
		self.df = self.df.groupby(['userId']).mean()
		
	def getTestRating(self):
		test_df = pd.read_csv("./csv/test_ratings.csv")
		res = []

		for idx, row in test_df.iterrows():
			# Periodic status update
			if idx % 300000 == 0:
				print(idx)
			userId = row['userId']
			
			try: 
				pred_val = self.df.loc[int(userId)]['rating']
				res.append([idx, pred_val])
			except:
				# Use the default average rating if no value can be found
				pred_val = AVG_RATING
				res.append([idx, pred_val])
		
		np.savetxt("user_avg.csv", np.array(res), delimiter=',', header="Id,rating", fmt="%i,%f", comments='')


if __name__ == '__main__':
	start_time = time.time()
	avgRating = AvgRatingByUser()
	avgRating.getTestRating()
	print("--- %s seconds ---" % (time.time() - start_time))
	