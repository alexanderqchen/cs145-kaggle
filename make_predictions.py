import os

from common import script_dir, data_dir, parquet_dir, model_dir, initSpark

from joblib import load

def main():
    context = initSpark()
    nn = load(os.path.join(script_dir, models_dir + "nn.model"))
    test = context.read.parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))

    test = test.toPandas()
    print("Making predictions...")
    test["rating"] = nn.predict(test.drop["Id"].fillna(0))
    print("Saving predictions...")
    test[["Id", "rating"]].to_csv(os.path.join(script_dir, data_dir + "results.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()