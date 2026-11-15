import joblib

datasets = ["FD001", "FD002", "FD003", "FD004"]

for dataset in datasets:
    path = f"models/xgboost_{dataset}.joblib"

    print("\n" + "=" * 45)
    print("DATASET:", dataset)
    print("=" * 45)

    model = joblib.load(path)

    print("Model type:", type(model).__name__)

    if hasattr(model, "get_params"):
        print("Model parameters:", model.get_params())

    if hasattr(model, "get_booster"):
        print("Number of boosting rounds:",
              model.get_booster().num_boosted_rounds())
