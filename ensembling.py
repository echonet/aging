import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
from sklearn.model_selection import train_test_split
import argparse

def mrn_fill(df):
    df['MRN'] = df['MRN'].astype(str).str.zfill(20)

def process_and_aggregate_data(df):
    df = df[['study_uid', 'view', 'MRN', 'Age', 'Age_preds']].copy()
    df.loc[:, 'view'] = df['view'].str.split('_').str[0]
    aggregated_df = (
        df
        .groupby('study_uid', as_index=False)
        .agg({
            'view': 'first',
            'MRN': 'first',
            'Age': 'first',
            'Age_preds': 'mean'
        })
    )
    return aggregated_df

def main(plax_csv, a4c_csv, a2c_csv, sc_csv, boosting_weights_path=None):
 
    plax_full = pd.read_csv(plax_csv)
    a4c_full = pd.read_csv(a4c_csv)
    a2c_full = pd.read_csv(a2c_csv)
    sc_full = pd.read_csv(sc_csv)

    for df in [plax_full, a4c_full, a2c_full, sc_full]:
        if 'frames' in df.columns:
            df.drop(columns=['frames'], inplace=True)


    mrn_fill(plax_full)
    mrn_fill(a4c_full)
    mrn_fill(a2c_full)
    mrn_fill(sc_full)

    plax_test_agg = process_and_aggregate_data(plax_full)
    a4c_test_agg = process_and_aggregate_data(a4c_full)
    a2c_test_agg = process_and_aggregate_data(a2c_full)
    sc_test_agg = process_and_aggregate_data(sc_full)

    plax_test_agg.rename(columns={'Age_preds': 'age_preds_plax'}, inplace=True)
    a4c_test_agg.rename(columns={'Age_preds': 'age_preds_a4c'}, inplace=True)
    a2c_test_agg.rename(columns={'Age_preds': 'age_preds_a2c'}, inplace=True)
    sc_test_agg.rename(columns={'Age_preds': 'age_preds_sc'}, inplace=True)

    for df in [plax_test_agg, a4c_test_agg, a2c_test_agg, sc_test_agg]:
        if 'view' in df.columns:
            df.drop(columns=['view'], inplace=True)

    combined_test_df = plax_test_agg.merge(a4c_test_agg, on=['study_uid', 'MRN', 'Age'], how='outer') \
        .merge(a2c_test_agg, on=['study_uid', 'MRN', 'Age'], how='outer') \
        .merge(sc_test_agg, on=['study_uid', 'MRN', 'Age'], how='outer')

    combined_test_df['MRN'] = combined_test_df['MRN'].astype(str).str.zfill(20)
    combined_test_df.drop_duplicates(inplace=True)

    X = combined_test_df[["age_preds_plax", "age_preds_a4c", "age_preds_a2c", "age_preds_sc"]]
    y = combined_test_df["Age"]
  
    model = joblib.load(boosting_weights_path)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    pearson_corr, p_value = pearsonr(y, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    print(f"Pearson Correlation: {pearson_corr} (p-value: {p_value:.10g})")

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "p_value": p_value
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistGradientBoostingRegressor on echo age predictions from multiple views")
    parser.add_argument("--plax_csv", required=True)
    parser.add_argument("--a4c_csv", required=True)
    parser.add_argument("--a2c_csv", required=True)
    parser.add_argument("--sc_csv", required=True)
    parser.add_argument("--boosting_weights_path", default=None)
    args = parser.parse_args()
    main(args.plax_csv, args.a4c_csv, args.a2c_csv, args.sc_csv, args.boosting_weights_path)