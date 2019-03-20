import os
import pandas as pd

from common import script_dir, parquet_dir, initSpark

from pyspark.sql.functions import col
from sklearn.decomposition import IncrementalPCA

def main():
    context = initSpark()
    print("Loading data...")
    movie = context.read.parquet(os.path.join(script_dir, parquet_dir + "movie_vectors.parquet"))
    user = context.read.parquet(os.path.join(script_dir, parquet_dir + "user_vectors.parquet"))
    train = context.read.parquet(os.path.join(script_dir, parquet_dir + "train_ratings.parquet"))
    val = context.read.parquet(os.path.join(script_dir, parquet_dir + "val_ratings.parquet"))
    test = context.read.parquet(os.path.join(script_dir, parquet_dir + "test_ratings.parquet"))

    movie_alias = movie.select("movieId", *((col(c)).alias("movie_" + c) for c in movie.columns[1:]))
    user_alias = user.select("userId", *((col(c)).alias("user_" + c) for c in user.columns[1:]))

    print("Calculating training vector....")
    train_vec = train.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating validation vector...")
    val_vec = val.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating test vector....")
    test_vec = test.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "Id",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:])).orderBy("Id")

    print("Converting training data to pandas...")
    train_vec = train_vec.toPandas().fillna(0)
    train_x = train_vec.drop("rating", axis=1)
    train_y = train_vec["rating"]
    print("Converting validation data to pandas...")
    val_vec = val_vec.toPandas().fillna(0)
    val_x = val_vec.drop("rating", axis=1)
    val_y = val_vec["rating"]
    print("Converting test data to pandas...")
    test_vec = test_vec.toPandas().fillna(0)
    test_x = test_vec.drop("Id", axis=1)
    test_id = test_vec["Id"]

    pca = IncrementalPCA(n_components=20)
    print("Fitting PCA model...")
    model = pca.fit(train_x)
    print("Reducing training dimensionality...")
    train_vec = pd.DataFrame(model.transform(train_x)).astype("float32")
    train_vec.insert(0, "rating", train_y)
    train_vec.columns = train_vec.columns.astype(str)
    print("Reducing validation dimensionality...")
    val_vec = pd.DataFrame(model.transform(val_x)).astype("float32")
    val_vec.insert(0, "rating", val_y)
    val_vec.columns = val_vec.columns.astype(str)
    print("Reducing test dimensionality...")
    test_vec = pd.DataFrame(model.transform(test_x)).astype("float32")
    test_vec.insert(0, "Id", test_id)
    test_vec.columns = test_vec.columns.astype(str)

    print("Saving train vector...")
    train_vec.to_parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet"))
    print("Saving validation vector...")
    val_vec.to_parquet(os.path.join(script_dir, parquet_dir + "val_vec.parquet"))
    print("Saving test vector...")
    test_vec.to_parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))
    print("Done.")


if __name__ == "__main__":
    main()
