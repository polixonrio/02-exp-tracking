import os
import pickle
import click
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlzoomcamp_02-11")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle("C:/Users/Lyrian/projects/mlops-zoomcamp/cohorts/2023/02-experiment-tracking/output/train.pkl")
    X_val, y_val = load_pickle("C:/Users/Lyrian/projects/mlops-zoomcamp/cohorts/2023/02-experiment-tracking/output/val.pkl")

    with mlflow.start_run():
        mlflow.set_tag("developer","fluffy")
        max_depth=10
        mlflow.log_param("max_depth", max_depth)

        rf = RandomForestRegressor(max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_param("rmse", rmse)

if __name__ == '__main__':
    run_train()
