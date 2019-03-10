import numpy as np
import pandas as pd


class DataProcessor():
	def getDataFrame(self, filePath):
		data = pd.read_csv(filePath)
		return data

	def getMovieNameFromId(self, df, this_id):
		return df.loc[df['movieId'] == this_id]['title'].item()

	def getTagNameFromId(self, df, this_id):
		return df.loc[df['tagId'] == this_id]['tag'].item()

	def formMoviesTagsTuples(self, movie_names, tags_names, scores):
		res = dict()
		# vectorization
		for movieId in scores['movieId'].unique():
			movie_name = self.getMovieNameFromId(movie_names, movieId)
			relevant_entries = scores.loc[scores['movieId'] == movieId]
			list_of_relevant_tags = []
			for entry, row in relevant_entries.iterrows():
				item = row['relevance'].item()
				if item > 0.08:  # hyperparameter here
					list_of_relevant_tags.append(self.getTagNameFromId(tags_names, row['tagId'].item()))
			res[movie_name] = list_of_relevant_tags
		return res

	def formPseudoSentences(self, dictionary):
		res = list()
		for key, val in dictionary.items():
			prefix = key + ' ' + 'is a movie about '
			suffix = ''
			for tag in val:
				tag = tag.replace(' ', '_')
				tag = tag.replace('.', '_')
				suffix = suffix + tag + ','
			res.append(prefix + suffix[:len(suffix) - 2] + '.')
		return res

if __name__ == "__main__":
	ds = DataProcessor()
	movies_names = ds.getDataFrame('../../ref/movies.csv')
	movies_names = movies_names.drop(columns='genres')
	tags_names = ds.getDataFrame('../../ref/genome-tags.csv')
	scores = ds.getDataFrame('../../ref/genome-scores.csv')
	dict = ds.formMoviesTagsTuples(movies_names, tags_names, scores)
	res = ds.formPseudoSentences(dict)
	f = open("pseudo_sentences.txt", "w", encoding="utf-8")
	for sentence in res:
		f.write(sentence)
