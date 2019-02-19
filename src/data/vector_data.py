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

    def userFeatureVectorization(self, train_ratings, movies):
        # accquire the number of users
        maxNum_users = np.amax(train_ratings['userId'])
        # The columns are: average relevance per tag
        users = np.zeros((maxNum_users, 1130))
        # scale original train_ratings to [-2, 2]
        train_ratings['rating'] = train_ratings['rating'] - 3
        # begin vectorization
        for i in range(1, maxNum_users + 1):
            user_tuples = train_ratings.loc[train_ratings['userId'] == i]
            rated_movies = movies[movies['movieId'].isin(user_tuples['movieId'])]
            weighted_movies = rated_movies.mul(np.array(user_tuples['rating']), axis=0)
            average = weighted_movies.sum(axis=0)
            average[0] = i
            average_rating = user_tuples['rating'].sum() / user_tuples.shape[0]
            users[i - 1] = np.append(np.array(average), average_rating)
        return users


if __name__ == '__main__':
    ps = DataProcessor()
    # Accquire files
    train_ratings = ps.getDataFrame('../../csv/train_ratings.csv')
    movies = ps.getDataFrame('../../csv/movies_vectors.csv')
    # Transofrorming userId to vectors
    users = ps.userFeatureVectorization(train_ratings, movies)
    # Creating appropirate headers
    names = ['userId']
    for i in range(1, 1129):
        tagId = 'tag' + str(i)
        names.append(tagId)
    names.append('average_rating')
    # Export dataframe to csv files
    pd.DataFrame(users).to_csv("users.csv", header=names, index=False)

    # Legacy code: generates movies_vectors.csv
    '''
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
    '''
