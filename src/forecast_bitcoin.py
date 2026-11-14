# dsci 352 final project part 2
# Bitcoin Forecast

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "bitcoin_daily_prices.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "bitcoin"

LOOKBACK = 30  # use the prior 30 days to predict the next day
TRAIN_END_YEAR = 2022  # train on 2019-2022, test on 2023+

EPOCHS = 50
BATCH_SIZE = 32
LSTM_UNITS = 64
DROPOUT_RATE = 0.2

# A - Build and Prepare Data
def load_and_clean(path):
    # read in the data
    df = pd.read_csv(path)
    # print(df.info())
    # convert date to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    # only keep needed columns, sort by date
    df = df[["Date", "High"]].sort_values("Date").reset_index(drop=True)
    # There are a small
    # number of multi-day gaps in the daily series;
    # we reindex to a continuous daily DatetimeIndex
    # forward-fill so the LSTM sees regular time steps
    df = df.set_index("Date").asfreq("D").ffill().reset_index()
    return df

# plot the price series, in a time-price line chart format
def plot_price_series(df, train_end_year):
    split_date = pd.Timestamp(f"{train_end_year}-12-31")
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["High"], color="#1f77b4", linewidth=1)
    plt.axvline(split_date, color="red", linestyle="--", label=f"Train/Test split ({split_date.date()})")
    plt.title("Bitcoin Daily High Price (USD)")
    plt.xlabel("Date")
    plt.ylabel("High Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("btc_price_series.png", dpi=150)
    plt.close()

def split_train_test(df, train_end_year):
    # Split chronologically: training = up to and including train_end_year
    train_df = df[df["Date"].dt.year <= train_end_year].reset_index(drop=True)
    test_df = df[df["Date"].dt.year > train_end_year].reset_index(drop=True)
    return train_df, test_df


# Fit MinMaxScaler on training data, transform both train and test.
# Returns the scaled arrays plus the fitted scaler
# we can invert the transform later when computing metrics in USD.
def scale_series(train_df, test_df, column="High"):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[[column]].values)
    test_scaled = scaler.transform(test_df[[column]].values)
    return train_scaled, test_scaled, scaler


# Window Construction
# Turn a 1-D series into supervised (X, y) pairs with a sliding window
# X[i] = series[i : i+lookback]    (shape: lookback x 1)
# y[i] = series[i+lookback]        (next-day value)
def make_windows(series, lookback):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i: i + lookback])
        y.append(series[i + lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


# Build (X_train, y_train) and (X_test, y_test) sliding-window arrays.
# To produce the very first test prediction we need the last `lookback`
# days of training history, so we prepend them to the test series before
# windowing. This avoids losing the first 30 test points.

def prepare_sequences(train_scaled, test_scaled, lookback):
    test_with_context = np.vstack([train_scaled[-lookback:], test_scaled])
    X_train, y_train = make_windows(train_scaled, lookback)
    X_test, y_test = make_windows(test_with_context, lookback)
    return X_train, y_train, X_test, y_test

def build_lstm_model(lookback, units, dropout):
    # Define the baseline LSTM:
    # A single LSTM layer with dropout regularization is a strong, simple
    # baseline for univariate next-step regression on ~1.4k training samples.
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# LSTM model training
def train_model(model, X_train, y_train, epochs, batch_size):

    # Fit the model with early stopping; return the History object.
    # Validation uses the last 10% of the training set (shuffle=False keeps
    # that split chronological). Early stopping reverts to the best weights.
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,           # preserve temporal order
        verbose=2,
    )
    return history

def plot_training_loss(history):
    # Plot training (and validation) loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train loss (MSE)")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("btc_training_loss.png", dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test, scaler):
    # Predict, invert MinMax scaling, and compute RMSE / MAE in USD
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return y_true, y_pred, rmse, mae


# Plot actual vs. predicted High on the test set; annotate metrics.
def plot_predictions(test_dates, y_true, y_pred, rmse, mae):
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, y_true, label="Actual High", color="#1f77b4", linewidth=1.5)
    plt.plot(test_dates, y_pred, label="Predicted High", color="#ff7f0e",
             linewidth=1.5, alpha=0.85)
    # Escape $ to avoid matplotlib's LaTeX math mode rendering.
    plt.title(
        f"BTC High: Actual vs. Predicted (Test Set)   "
        f"RMSE = \\${rmse:,.0f}   MAE = \\${mae:,.0f}"
    )
    plt.xlabel("Date")
    plt.ylabel("High Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bitcoin_predictions.png", dpi=150)
    plt.close()


def main():

    # Reproducibility: seed all the relevant RNGs.
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(OUTPUT_DIR)

    # A-1,2
    df = load_and_clean(DATA_PATH)
    # print(df.info())
    # print(df.head())

    # A-3 Visualize and split the data into Train/test
    plot_price_series(df, TRAIN_END_YEAR)
    train_df, test_df = split_train_test(df, TRAIN_END_YEAR)
    print(f"Train: {len(train_df)} rows  "
          f"({train_df['Date'].min().date()} -> {train_df['Date'].max().date()})")
    print(f"Test : {len(test_df)} rows  "
          f"({test_df['Date'].min().date()} -> {test_df['Date'].max().date()})")

    # A-4 Scale the data
    train_scaled, test_scaled, scaler = scale_series(train_df, test_df)

    # A-5 Create Window
    X_train, y_train, X_test, y_test = prepare_sequences(train_scaled, test_scaled, LOOKBACK)
    print(f"X_train shape: {X_train.shape}   y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}    y_test  shape: {y_test.shape}")

    # B-1,2 Define LSTM, Compile using Adam optimizer and MSE/MAE Loss
    model = build_lstm_model(LOOKBACK, LSTM_UNITS, DROPOUT_RATE)
    model.summary()

    # B-3 Train the model and plot loss over epochs
    history = train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE)
    plot_training_loss(history)


    # C-1 Predict Prices & plot Actual vs. Predicted
    y_true, y_pred, rmse, mae = evaluate_model(model, X_test, y_test, scaler)
    plot_predictions(
        test_df["Date"].values, y_true, y_pred, rmse, mae,
    )

    # C-2 Compute RMSE
    print(f"Test RMSE: ${rmse:,.2f}")
    print(f"Test MAE : ${mae:,.2f}")


if __name__ == "__main__":
    main()


