import os

from common import script_dir, parquet_dir, initSpark

from pyspark.sql.functions import col
from pyspark.ml.feature import PCA, VectorAssembler

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
    train_vec = train.sample(False, 0.1).join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating validation vector...")
    val_vec = val.sample(False, 0.01).join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "rating",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))
    print("Calculating test vector....")
    test_vec = test.join(movie_alias, "movieId", "left_outer").join(user_alias, "userId", "left_outer").select(
        "Id",
        *((col("movie_" + c) * col("user_" + c)).alias(c) for c in movie.columns[1:]))

    assembler = VectorAssembler(
        inputCols=[c for c in movie.columns[1:]],
        outputCol="features",
        handleInvalid="keep")

    print("Adding training feature column...")
    train_vec = assembler.transform(train_vec)
    print("Adding validation feature column...")
    val_vec = assembler.transform(val_vec)
    print("Adding test feature column...")
    test_vec = assembler.transform(test_vec)

    pca = PCA(k=100, inputCol="features", outputCol="pca_features")
    print("Fitting PCA model...")
    model = pca.fit(train_vec)
    print("Reducing training dimensionality...")
    train_vec = model.transform(train_vec).rdd.map(extractWithRating).toDF(["rating"])
    print("Reducing validation dimensionality...")
    val_vec = model.transform(val_vec).rdd.map(extractWithRating).toDF(["rating"])
    print("Reducing test dimensionality...")
    test_vec = model.transform(test_vec).rdd.map(extractWithId).toDF(["Id"])

    print("Saving train vector...")
    train_vec.write.parquet(os.path.join(script_dir, parquet_dir + "train_vec.parquet"))
    print("Saving validation vector...")
    val_vec.write.parquet(os.path.join(script_dir, parquet_dir + "val_vec.parquet"))
    print("Saving test vector...")
    test_vec.write.parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))
    print("Done.")


def extractWithRating(row):
    return (row.rating, ) + tuple(row.pca_features.toArray().tolist())


def extractWithId(row):
    return (row.Id, ) + tuple(row.pca_features.toArray().tolist())


if __name__ == "__main__":
    main()
