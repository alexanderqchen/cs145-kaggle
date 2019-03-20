import os

import pandas as pd
from joblib import load

from common import script_dir, data_dir, parquet_dir, model_dir


def main():
    print("Loading model...")
    nn = load(os.path.join(script_dir, model_dir + "nn.model"))
    print("Loading data...")
    test = pd.read_parquet(os.path.join(script_dir, parquet_dir + "test_vec.parquet"))

    print("Making predictions...")
    test["rating"] = nn.predict(test.drop("Id", axis=1).fillna(0))
    print("Saving predictions...")
    test[["Id", "rating"]].to_csv(os.path.join(script_dir, data_dir + "results.csv"), index=False)
    print("Done.")


if __name__ == "__main__":
    main()
