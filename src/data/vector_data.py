import numpy as np
import pandas as pd


class DataProcessor():
    def getDataFrame(self, filePath):
        data = pd.read_csv(filePath)
        return data

    def convertMoviesToVectorsOfTags(self, genome_scores):
        # There are 1128 tags
        # The rows is determined by the maximum movieId found in genome_scores
        rows = np.amax(genome_scores['movieId'])
        movies = np.empty([rows, 1129])
        len = genome_scores.shape[0]
        # vectorization
        for i in range(len):
            entry = genome_scores.iloc[i]
            try:
                row = int(entry['movieId']) - 1
                col = int(entry['tagId'])
            except ValueError:
                print('The error is caught at row', i, '\n')
                continue
            movies[row][col] = entry['relevance']
        # assigning movieId to respecitve col
        for i in range(movies.shape[0]):
            movies[i][0] = i + 1
        return movies


if __name__ == '__main__':
    # Create a DataProcessor instance to access attribute functions
    ps = DataProcessor()
    # Accessing files
    genome_scores = ps.getDataFrame('../../csv/genome-scores.csv')
    # Transforming to vectors
    movies = ps.convertMoviesToVectorsOfTags(genome_scores)
    # Creating appropriate headers
    names = ['movieId']
    for i in range(1, 1129):
        tagId = 'tag' + str(i)
        names.append(tagId)
    # Export dataframe to csv files
    pd.DataFrame(movies).to_csv("movies_vectors.csv", header=names, index=False)
