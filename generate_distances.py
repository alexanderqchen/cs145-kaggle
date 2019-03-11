import pandas as pd

data_dir = "csv/"

if __name__ == "__main__":
    movies = pd.read_csv(data_dir + "genome-scores.csv")
    movies_pivot = movies.pivot(index="movieId", columns="tagId", values="relevance")
    train_ratings = pd.read_csv(data_dir + "train_ratings.csv")
    user_vectors = pd.read_csv(data_dir + "user_vectors.csv")
    for userId in train_ratings["userId"].unique():
        user_data = train_ratings.loc[train_ratings["userId"] == userId]
        movie_vectors = user_data.join(movies_pivot, on="movieId").drop(columns=["userId", "movieId", "rating"])
        user_vector = user_vectors.loc[user_vectors["userId"] == userId].drop(columns=["userId"])
        score = movie_vectors.mul(user_vector.loc[0], axis=1)
        score['rating'] = user_data['rating']
        if userId == 1:
            score.to_csv(data_dir + "train_scores.csv", mode='a', header=True, index=False)
        else:
            score.to_csv(data_dir + "train_scores.csv", mode='a', header=False, index=False)