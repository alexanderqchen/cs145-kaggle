from __future__ import print_function

import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, col, avg
from pyspark.sql.types import ArrayType, StringType, IntegerType, BooleanType

script_dir = os.path.dirname(__file__)
space_dir = "/space"

# TASK 1
def get_initial_dfs(context, saved_initial_dfs):
    data_dir = "csv/"
    parquet_dir = "parquets/"

    if not saved_initial_dfs:
        # load csv and json files
        genome_scores = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "genome-scores.csv"))
        genome_scores = genome_scores.select(
            col("movieId").cast("int").alias("movieId"),
            col("tagId").cast("int").alias("tagId"),
            col("relevance").cast("float").alias("relevance"))
        train_ratings = context.read.format("csv").option("header", "true").load(
            os.path.join(script_dir, data_dir + "train_ratings.csv"))
        train_ratings = train_ratings.select(
            col("userId").cast("int").alias("userId"),
            col("movieId").cast("int").alias("movieId"),
            col("rating").cast("float").alias("rating"))
        print("Loaded genome scores and train ratings.")

        print("Calculating movie vector.")
        movie_vectors = genome_scores.groupBy("movieId").pivot("tagId").sum("relevance")
        print("Calculating user vector.")
        user_vectors = train_ratings.join(movie_vectors, train_ratings.movieId == movie_vectors.movieId).select(
            "userId",
            "rating",
            *((col(c) * (train_ratings.rating - 3) / 2).alias(c) for c in movie_vectors.columns[1:])
        ).groupBy("userId").agg(*(avg(c).alias(c) for c in movie_vectors.columns[1:]))

        print("Saving parquets.")
        movie_vectors.write.parquet(os.path.join(space_dir, parquet_dir + "movie_vectors.parquet"))
        user_vectors.write.parquet(os.path.join(space_dir, parquet_dir + "user_vectors.parquet"))
        train_ratings.write.parquet(os.path.join(space_dir, parquet_dir + "train_ratings.parquet"))
    else:
        # load parquets
        movie_vectors = context.read.parquet(os.path.join(space_dir, parquet_dir + "movie_vectors.parquet"))
        user_vectors = context.read.parquet(os.path.join(space_dir, parquet_dir + "user_vectors.parquet"))
        train_ratings = context.read.parquet(os.path.join(space_dir, parquet_dir + "train_ratings.parquet"))

    return movie_vectors, user_vectors, train_ratings


def compute_labeled_sanitized_comments(comments, labels):
    # TASK 2
    labelled_comments = comments.join(labels, comments.id == labels.Input_id)

    # TASK 6B
    def check_positive(x):
        if x == "1":
            return 1
        return 0

    def check_negative(x):
        if x == "-1":
            return 1
        return 0

    check_negative_udf = udf(check_negative, IntegerType())
    check_positive_udf = udf(check_positive, IntegerType())

    # TASKS 4, 5
    sanitize_udf = udf(sanitize_wrapper, ArrayType(StringType()))

    labelled_sanitized_comments = labelled_comments.select(
        sanitize_udf("body").alias("features"),
        check_positive_udf("labeldjt").alias("trump_pos"),
        check_negative_udf("labeldjt").alias("trump_neg"),
        check_positive_udf("labeldem").alias("dem_pos"),
        check_negative_udf("labeldem").alias("dem_neg"),
        check_positive_udf("labelgop").alias("rep_pos"),
        check_negative_udf("labelgop").alias("rep_neg"))

    # TASK 6A
    cv = CountVectorizer(inputCol="features", outputCol="vectors", binary=True, minDF=5)
    model = cv.fit(labelled_sanitized_comments)
    sanitized_comments = model.transform(labelled_sanitized_comments)
    return sanitized_comments, model


# TASK 7
def train_models(saved_trained_models, sanitized_comments):
    models_dir = "models/"
    parquet_dir = "parquets/"
    if not saved_trained_models:
        # Initialize six logistic regression models.
        pos_lr = LogisticRegression(labelCol="trump_pos", featuresCol="vectors", maxIter=10)
        neg_lr = LogisticRegression(labelCol="trump_neg", featuresCol="vectors", maxIter=10)
        dem_pos_lr = LogisticRegression(labelCol="dem_pos", featuresCol="vectors", maxIter=10)
        dem_neg_lr = LogisticRegression(labelCol="dem_neg", featuresCol="vectors", maxIter=10)
        rep_pos_lr = LogisticRegression(labelCol="rep_pos", featuresCol="vectors", maxIter=10)
        rep_neg_lr = LogisticRegression(labelCol="rep_neg", featuresCol="vectors", maxIter=10)

        # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
        pos_evaluator = BinaryClassificationEvaluator(labelCol="trump_pos")
        neg_evaluator = BinaryClassificationEvaluator(labelCol="trump_neg")
        dem_pos_evaluator = BinaryClassificationEvaluator(labelCol="dem_pos")
        dem_neg_evaluator = BinaryClassificationEvaluator(labelCol="dem_neg")
        rep_pos_evaluator = BinaryClassificationEvaluator(labelCol="rep_pos")
        rep_neg_evaluator = BinaryClassificationEvaluator(labelCol="rep_neg")

        # There are a few parameters associated with logistic regression. We do not know what they are a priori.
        # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
        # We will assume the parameter is 1.0. Grid search takes forever.
        pos_param_grid = ParamGridBuilder().addGrid(pos_lr.regParam, [1.0]).build()
        neg_param_grid = ParamGridBuilder().addGrid(neg_lr.regParam, [1.0]).build()
        dem_pos_param_grid = ParamGridBuilder().addGrid(dem_pos_lr.regParam, [1.0]).build()
        dem_neg_param_grid = ParamGridBuilder().addGrid(dem_neg_lr.regParam, [1.0]).build()
        rep_pos_param_grid = ParamGridBuilder().addGrid(rep_pos_lr.regParam, [1.0]).build()
        rep_neg_param_grid = ParamGridBuilder().addGrid(rep_neg_lr.regParam, [1.0]).build()

        # We initialize a 5 fold cross-validation pipeline.
        pos_cross_val = CrossValidator(
            estimator=pos_lr,
            evaluator=pos_evaluator,
            estimatorParamMaps=pos_param_grid,
            numFolds=5)
        neg_cross_val = CrossValidator(
            estimator=neg_lr,
            evaluator=neg_evaluator,
            estimatorParamMaps=neg_param_grid,
            numFolds=5)
        dem_pos_cross_val = CrossValidator(
            estimator=dem_pos_lr,
            evaluator=dem_pos_evaluator,
            estimatorParamMaps=dem_pos_param_grid,
            numFolds=5)
        dem_neg_cross_val = CrossValidator(
            estimator=dem_neg_lr,
            evaluator=dem_neg_evaluator,
            estimatorParamMaps=dem_neg_param_grid,
            numFolds=5)
        rep_pos_cross_val = CrossValidator(
            estimator=rep_pos_lr,
            evaluator=rep_pos_evaluator,
            estimatorParamMaps=rep_pos_param_grid,
            numFolds=5)
        rep_neg_cross_val = CrossValidator(
            estimator=rep_neg_lr,
            evaluator=rep_neg_evaluator,
            estimatorParamMaps=rep_neg_param_grid,
            numFolds=5)

        # Split the data 50/50
        pos_train, pos_test = sanitized_comments.randomSplit([0.5, 0.5])
        neg_train, neg_test = sanitized_comments.randomSplit([0.5, 0.5])
        dem_pos_train, dem_pos_test = sanitized_comments.randomSplit([0.5, 0.5])
        dem_neg_train, dem_neg_test = sanitized_comments.randomSplit([0.5, 0.5])
        rep_pos_train, rep_pos_test = sanitized_comments.randomSplit([0.5, 0.5])
        rep_neg_train, rep_neg_test = sanitized_comments.randomSplit([0.5, 0.5])

        # Train the models
        print("Training positive classifier...")
        pos_model = pos_cross_val.fit(pos_train)
        print("Training negative classifier...")
        neg_model = neg_cross_val.fit(neg_train)
        print("Training positive democrat classifier...")
        dem_pos_model = dem_pos_cross_val.fit(dem_pos_train)
        print("Training negative democrat classifier...")
        dem_neg_model = dem_neg_cross_val.fit(dem_neg_train)
        print("Training positive republican classifier...")
        rep_pos_model = rep_pos_cross_val.fit(rep_pos_train)
        print("Training negative republican classifier...")
        rep_neg_model = rep_neg_cross_val.fit(rep_neg_train)

        # Once we train the models, we don't want to do it again. We can save the models and load them again later.
        pos_model.save(models_dir + "pos.model")
        neg_model.save(models_dir + "neg.model")
        dem_pos_model.save(models_dir + "dem_pos.model")
        dem_neg_model.save(models_dir + "dem_neg.model")
        rep_pos_model.save(models_dir + "rep_pos.model")
        rep_neg_model.save(models_dir + "rep_neg.model")

        # save testing data
        pos_test.write.parquet(os.path.join(script_dir, parquet_dir + "pos_test.parquet"))
        neg_test.write.parquet(os.path.join(script_dir, parquet_dir + "neg_test.parquet"))
        dem_pos_test.write.parquet(os.path.join(script_dir, parquet_dir + "dem_pos_test.parquet"))
        dem_neg_test.write.parquet(os.path.join(script_dir, parquet_dir + "dem_neg_test.parquet"))
        rep_pos_test.write.parquet(os.path.join(script_dir, parquet_dir + "rep_pos_test.parquet"))
        rep_neg_test.write.parquet(os.path.join(script_dir, parquet_dir + "rep_neg_test.parquet"))
    else:
        # load models
        pos_model = CrossValidatorModel.load(models_dir + "pos.model")
        neg_model = CrossValidatorModel.load(models_dir + "neg.model")
        dem_pos_model = CrossValidatorModel.load(models_dir + "dem_pos.model")
        dem_neg_model = CrossValidatorModel.load(models_dir + "dem_neg.model")
        rep_pos_model = CrossValidatorModel.load(models_dir + "rep_pos.model")
        rep_neg_model = CrossValidatorModel.load(models_dir + "rep_neg.model")
    return pos_model, neg_model, dem_pos_model, dem_neg_model, rep_pos_model, rep_neg_model


def predict_sentiment(context, predicted_sentiment, comments, submissions, pos_model, neg_model, dem_pos_model,
                      dem_neg_model, rep_pos_model, rep_neg_model, cv_model):
    parquet_dir = "parquets/"

    # threshold functions
    def check_positive(x):
        if x[1] > .2:
            return 1
        return 0

    def check_negative(x):
        if x[1] > .25:
            return 1
        return 0

    def check_dem_positive(x):
        if x[1] > .025:
            return 1
        return 0

    def check_dem_negative(x):
        if x[1] > .42:
            return 1
        return 0

    def check_rep_positive(x):
        if x[1] > .17:
            return 1
        return 0

    def check_rep_negative(x):
        if x[1] > .23:
            return 1
        return 0

    if not predicted_sentiment:
        check_positive_udf = udf(check_positive, IntegerType())
        check_negative_udf = udf(check_negative, IntegerType())
        check_dem_positive_udf = udf(check_dem_positive, IntegerType())
        check_dem_negative_udf = udf(check_dem_negative, IntegerType())
        check_rep_positive_udf = udf(check_rep_positive, IntegerType())
        check_rep_negative_udf = udf(check_rep_negative, IntegerType())
        sanitize_udf = udf(sanitize_wrapper, ArrayType(StringType()))
        slice_link_id_udf = udf(lambda x: x[3:], StringType())

        # TASK 8
        comment_sanitized = comments.where(
            "NOT body LIKE '%/s%' AND NOT body LIKE '&gt;%'"
        ).select(
            "id",
            "created_utc",
            "score",
            "link_id",
            "author_flair_text",
            sanitize_udf("body").alias("features"))

        modded_submissions = submissions.selectExpr(
            "id as submission_id",
            "score as submission_score",
            "title")

        comment_sanitized = comment_sanitized.join(
            modded_submissions,
            slice_link_id_udf(comment_sanitized.link_id) == modded_submissions.submission_id)

        # TASK 9 (repeating 4, 5, 6A...)
        print("Counting vectors...")
        cv_result = cv_model.transform(comment_sanitized)

        print("Computing sentiment percentages...")
        pos_result = pos_model.transform(cv_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "probability as pos_probability")
        neg_result = neg_model.transform(pos_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "pos_probability",
            "probability as neg_probability")
        dem_pos_result = dem_pos_model.transform(neg_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "pos_probability",
            "neg_probability",
            "probability as dem_pos_probability")
        dem_neg_result = dem_neg_model.transform(dem_pos_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "pos_probability",
            "neg_probability",
            "dem_pos_probability",
            "probability as dem_neg_probability")
        rep_pos_result = rep_pos_model.transform(dem_neg_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "pos_probability",
            "neg_probability",
            "dem_pos_probability",
            "dem_neg_probability",
            "probability as rep_pos_probability")
        all_result = rep_neg_model.transform(rep_pos_result).selectExpr(
            "id", "created_utc", "score", "link_id", "author_flair_text", "features", "submission_id",
            "submission_score", "title", "vectors",
            "pos_probability",
            "neg_probability",
            "dem_pos_probability",
            "dem_neg_probability",
            "rep_pos_probability",
            "probability as rep_neg_probability")

        print("Converting percentage result to binary")
        predicted_comments = all_result.select(
            "id",
            "title",
            "submission_id",
            "score",
            "submission_score",
            "created_utc",
            "author_flair_text",
            check_positive_udf("pos_probability").alias("pos"),
            check_negative_udf("neg_probability").alias("neg"),
            check_dem_positive_udf("dem_pos_probability").alias("dem_pos"),
            check_dem_negative_udf("dem_neg_probability").alias("dem_neg"),
            check_rep_positive_udf("rep_pos_probability").alias("rep_pos"),
            check_rep_negative_udf("rep_neg_probability").alias("rep_neg"))
        print("Predicted Sentiment. Now save as parquet.")

        # save as parquet for future use
        predicted_comments.write.parquet(os.path.join(script_dir, parquet_dir + "sentiment.parquet"))
    else:
        # load parquets
        predicted_comments = context.read.parquet(os.path.join(script_dir, parquet_dir + "sentiment.parquet"))
    return predicted_comments


# TASK 10
def save_csvs(context, predicted_comments):
    csv_dir = "csv/"
    predicted_comments.registerTempTable("predicted")

    # 10.1
    percent_by_submission = context.sql(
        "SELECT FIRST(title), submission_id, "
        "AVG(pos), AVG(neg), AVG(dem_pos), AVG(dem_neg), AVG(rep_pos), AVG(rep_neg) "
        "FROM predicted "
        "GROUP BY submission_id")

    # 10.2
    percent_by_day = context.sql(
        "SELECT FROM_UNIXTIME(created_utc, 'YYYY-MM-dd') AS date, "
        "AVG(pos) AS Positive, AVG(neg) AS Negative, "
        "AVG(dem_pos) AS DemPositive, AVG(dem_neg) AS DemNegative, "
        "AVG(rep_pos) AS RepPositive, AVG(rep_neg) AS RepNegative "
        "FROM predicted "
        "GROUP BY date")

    # 10.3

    def is_state(x):
        states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
                  "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
                  "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
                  "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
                  "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
                  "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
                  "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
                  "Wyoming"]
        return x in states

    is_state_udf = udf(is_state, BooleanType())
    percent_by_state = context.sql(
        "SELECT author_flair_text AS state, "
        "AVG(pos) AS Positive, AVG(neg) AS Negative, "
        "AVG(dem_pos) AS DemPositive, AVG(dem_neg) AS DemNegative, "
        "AVG(rep_pos) AS RepPositive, AVG(rep_neg) AS RepNegative "
        "FROM predicted "
        "GROUP BY state").where(is_state_udf("state"))

    # 10.4
    percent_by_comment_score = context.sql(
        "SELECT score AS comment_score, "
        "AVG(pos) AS Positive, AVG(neg) AS Negative, "
        "AVG(dem_pos) AS DemPositive, AVG(dem_neg) AS DemNegative, "
        "AVG(rep_pos) AS RepPositive, AVG(rep_neg) AS RepNegative "
        "FROM predicted "
        "GROUP BY comment_score")
    percent_by_submission_score = context.sql(
        "SELECT submission_score, "
        "AVG(pos) AS Positive, AVG(neg) AS Negative, "
        "AVG(dem_pos) AS DemPositive, AVG(dem_neg) AS DemNegative, "
        "AVG(rep_pos) AS RepPositive, AVG(rep_neg) AS RepNegative "
        "FROM predicted "
        "GROUP BY submission_score")

    # save as csv
    print("Saving as csv.")
    percent_by_submission.orderBy("AVG(pos)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_pos_submissions.csv"))
    percent_by_submission.orderBy("AVG(neg)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_neg_submissions.csv"))
    percent_by_submission.orderBy("AVG(dem_pos)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_dem_pos_submissions.csv"))
    percent_by_submission.orderBy("AVG(dem_neg)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_dem_neg_submissions.csv"))
    percent_by_submission.orderBy("AVG(rep_pos)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_rep_pos_submissions.csv"))
    percent_by_submission.orderBy("AVG(rep_neg)", ascending=False).limit(10).repartition(1).write.format(
        "com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "top10_rep_neg_submissions.csv"))
    percent_by_day.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "time_data.csv"))
    percent_by_state.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "state_data.csv"))
    percent_by_comment_score.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "comment_score.csv"))
    percent_by_submission_score.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save(
        os.path.join(script_dir, csv_dir + "submission_score.csv"))


def main(context):
    saved_initial_dfs = False
    saved_trained_models = False
    predicted_sentiment = False

    movie_vectors, user_vectors, train_ratings = get_initial_dfs(context, saved_initial_dfs)


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]").set("spark.executor.memory", "16G").set("spark.driver.memory", "16G").set("spark.local.dir", space_dir)
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sc)
    main(sql_context)
