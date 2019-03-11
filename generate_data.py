import pandas as pd

data_dir = "csv/"

if __name__ == "__main__":
    movies = pd.read_csv(data_dir + "genome-scores.csv")
    movies_pivot = movies.pivot(index="movieId", columns="tagId", values="relevance")
    train_ratings = pd.read_csv(data_dir + "train_ratings.csv")
    for userId in train_ratings["userId"].unique():
        user_data = train_ratings[train_ratings["userId"] == userId]
        weights = (user_data["rating"]-3)/2
        user_vector = user_data[["movieId"]].join(movies_pivot, on="movieId").mul(weights, axis=0).sum(axis=0).to_frame().transpose()
        user_vector = user_vector[user_vector.columns[1:]]
        user_vector.insert(0, "userId", userId)
        if userId == 1:
            user_vector.to_csv(data_dir + "user_vectors.csv", mode='a', header=True)
        else:
            user_vector.to_csv(data_dir + "user_vectors.csv", mode='a', header=False)
    pd.DataFrame(movies_pivot).to_csv("movies_vectors.csv", index=False)