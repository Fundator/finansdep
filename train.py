import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import mlflow.pyfunc
import json
import shap
import argparse

categorical_feature_names = ["Suburb", "Type", "Method", "Regionname", "CouncilArea"]
columns_to_encode = ["Suburb", "Type", "Method", "Postcode", "Regionname", "CouncilArea"]
columns_to_drop = ["Address", "Date", "Price", "SellerG", "LogPrice"]

class HousePricePredictor(mlflow.pyfunc.PythonModel):

    def __init__(self, encoder, explainer, cat_col_idx, reg, log_target=False):
        self.encoder = encoder
        self.reg = reg
        self.explainer = explainer
        self.cat_col_idx = cat_col_idx
        self.log_target = log_target
        
    def predict(self, context, model_input):
        print(model_input)
        print(type(model_input))
        df = model_input
        df[categorical_feature_names] = df[categorical_feature_names].astype("category")
        new_cols = self.encoder.transform(df[columns_to_encode])
        for c in categorical_feature_names:
            df[c] = df[c].cat.codes.astype(int)
        df.loc[:, columns_to_encode] = new_cols
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year 
        df = df.drop("Date", axis=1)
        print(df)
        pred = self.reg.predict(df)
        shap_values = explainer.shap_values(Pool(df, pred, cat_features=cat_col_idx))
        print(shap_values)
        resp = {}
        if self.log_target:
            global_mean = np.exp(explainer.expected_value)
            resp["Predictions"] = np.exp(pred).tolist()
            resp["Explanations"] = [{name:(global_mean-(global_mean*np.exp(value)))*-1 for name, value in zip(df.columns, example_shap)} for example_shap in shap_values]
        else:
            global_mean = explainer.expected_value
            resp["Predictions"] = pred.tolist()
            resp["Explanations"] = [{name:value for name, value in zip(df.columns, example_shap)} for example_shap in shap_values]
        for e in resp["Explanations"]:
            e["GlobalMean"] = global_mean
        return resp

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="Specify seed for reproducibility", type=int, default=42)
parser.add_argument("-d", "--datafile", help="Path to datafile", type=str, default="MELBOURNE_HOUSE_PRICES_LESS.csv")

def winsorized_absolute_log_error(y_true, y_pred, epsilon=1e-7):
        return np.mean(np.minimum(np.abs(np.log(y_pred+epsilon) - np.log(y_true+epsilon)), 0.4))

def get_metrics(true, pred, method="unknown", print_metrics=True, log_target=True):
    """
    Get the four metrics specified.
    true: Ground true data
    pred: Predicted data
    method: Type of model
    print_metrics: If metrics should be printer, default=True
    log_target: If the target value is logaritmic, we should do an inverse transform to get metrics in original units, default=True
    """
    if log_target:
        true, pred = np.exp(true), np.exp(pred)

    WALE = winsorized_absolute_log_error(true,pred)
    RMSE = np.sqrt(mean_squared_error(true, pred))
    MAE = mean_absolute_error(true, pred)
    R2 = r2_score(true, pred)
    return_dict = {"WALE": WALE, "RMSE": RMSE, "MAE": MAE, "R2": R2}
    if print_metrics:
        for k,v in return_dict.items():
            print(f"{k}: {v}")
    return return_dict

def split_dataset(dataframe, val_index, holdout_test_index):
    train = dataframe.iloc[:val_index]
    val = dataframe.iloc[val_index:holdout_test_index]
    holdout_test = dataframe.iloc[holdout_test_index:]
    return train, val, holdout_test

if __name__ == "__main__":

    args = parser.parse_args()
    print("Using arguments:")
    print(args)
    seed = args.seed
    datafile = args.datafile
    
    conda_env_path = "environment.yml"
    model_prefix = "/models"
    data_file = args.datafile

    with mlflow.start_run() as run:
        # Get path to save model
        tracking_uri = mlflow.tracking.get_tracking_uri() 
        print("Logging to "+tracking_uri)
        artifact_uri = mlflow.get_artifact_uri()
        print("Saving artifacts to "+artifact_uri)
        model_path = artifact_uri+model_prefix
        mlflow.log_param("seed", seed)
        mlflow.log_param("data_file", data_file)
        
        df = pd.read_csv(datafile, parse_dates=["Date"])
        df = df.dropna(subset=["Price"])

        df[categorical_feature_names] = df[categorical_feature_names].astype("category")
        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        df["LogPrice"] = df["Price"].apply(np.log)

        df = df.sort_values(by="Date")
        holdout_test_idx = df.shape[0] - int(df.shape[0] * 0.1)
        val_idx = df.shape[0] - int(df.shape[0] * 0.1) * 2

        rfc_enc = OrdinalEncoder()
        rfc_df = df
        rfc_df[columns_to_encode] = rfc_enc.fit_transform(df[columns_to_encode])

        rfc_train, rfc_valid, rfc_holdout_test = split_dataset(rfc_df, val_idx, holdout_test_idx)

        rfc_X_train = rfc_train.drop(columns=columns_to_drop)
        rfc_y_train = rfc_train["Price"]

        rfc_X_valid = rfc_valid.drop(columns=columns_to_drop)
        rfc_y_valid = rfc_valid["Price"]

        rfc_X_test = rfc_holdout_test.drop(columns=columns_to_drop)
        rfc_y_test = rfc_holdout_test["Price"]

        cat_col_idx = np.array([rfc_X_train.columns.get_loc(c) for c in columns_to_encode])
        rfc = CatBoostRegressor(iterations=500, loss_function="RMSE", task_type="CPU", border_count=256, model_size_reg=0)
        rfc.fit(X=rfc_X_train, y=rfc_y_train, cat_features=cat_col_idx, eval_set=(rfc_X_valid, rfc_y_valid), silent=True, plot=False)
        explainer = shap.TreeExplainer(rfc)
        
        housereg = HousePricePredictor(rfc_enc, explainer, cat_col_idx, rfc, log_target=False)
        print("Saving model to "+model_path)
        mlflow.pyfunc.save_model(model_path, conda_env=conda_env_path, python_model=housereg)
        
        rfc_pred = rfc.predict(rfc_X_test)

        rfc_metrics = get_metrics(rfc_y_test, rfc_pred, log_target=False)
        for metric_name, metric_value in rfc_metrics.items():
            mlflow.log_metric(metric_name, metric_value) 