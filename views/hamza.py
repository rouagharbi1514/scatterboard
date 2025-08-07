import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import random


def set_seed(seed=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(50)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        seq_x = self.X[idx:idx + self.seq_len]
        seq_y = self.y[idx + self.seq_len]
        # Avoid torch.from_numpy to circumvent environments lacking NumPy bindings in Torch
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


def preprocess_data(filepath, sheet_name="Sheet1"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df[['Date', 'TotalRevenue']].dropna()   # change the second df variable with the variable that you want to forecast
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear

    features = ['day', 'month', 'dayofweek', 'dayofyear']
    target = 'TotalRevenue'  # this one too

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32).reshape(-1, 1)

    return df, X, y


def scale_data(X, y):
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y).flatten()

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_model(model, dataset, epochs=300, batch_size=32, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")


def evaluate_model(model, dataset, scaler_y, sequence_length):
    model.eval()
    with torch.no_grad():
        y_preds_scaled = []
        for i in range(len(dataset)):
            seq_x, _ = dataset[i]
            pred = model(seq_x.unsqueeze(0))
            y_preds_scaled.append(pred.item())

    y_true_scaled = dataset.y[sequence_length:]
    y_preds = scaler_y.inverse_transform(np.array(y_preds_scaled).reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, y_preds)
    r2 = r2_score(y_true, y_preds)
    print(f"Training MAE (LSTM): {mae:.2f}")
    print(f"Training R² (LSTM): {r2:.2f}")

    return y_true, y_preds


def forecast_future(model, X_scaled, df_dates, scaler_X, scaler_y, sequence_length, forecast_horizon=30):
    future_preds = []
    # Use torch.tensor instead of torch.from_numpy to remove the hard NumPy dependency inside torch
    current_seq = torch.tensor(X_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0)

    model.eval()
    future_dates = df_dates.copy()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            pred_scaled = model(current_seq).item()
            future_preds.append(pred_scaled)

            last_date = future_dates.iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            future_dates = pd.concat([future_dates, pd.Series([next_date])], ignore_index=True)

            day = next_date.day
            month = next_date.month
            dayofweek = next_date.dayofweek
            dayofyear = next_date.dayofyear
            next_features = np.array([[day, month, dayofweek, dayofyear]], dtype=np.float32)
            next_features_scaled = scaler_X.transform(next_features)

            next_tensor = torch.tensor(next_features_scaled, dtype=torch.float32).unsqueeze(0)
            current_seq = torch.cat((current_seq[:, 1:, :], next_tensor), dim=1)

    future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=df_dates.max() + pd.Timedelta(days=1), periods=forecast_horizon)

    return pd.DataFrame({'Date': forecast_dates, 'PredictedRevenue': future_preds})


# just plots
def plot_results(df, future_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['TotalRevenue'], label="Historical Revenue", color='blue')
    plt.plot(future_df['Date'], future_df['PredictedRevenue'], label="Forecast (30 days)", color='green')
    plt.title("TotalRevenue Forecast (Next 30 Days) - LSTM")
    plt.xlabel("Date")
    plt.ylabel("TotalRevenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    filepath = "/Users/hamzamathlouthi/Desktop/scatterboard02/hotel-dashboard/data/hotel_data.xlsx"
    sequence_length = 14

    df, X, y = preprocess_data(filepath)
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)

    dataset = TimeSeriesDataset(X_scaled, y_scaled, sequence_length)

    model = LSTMModel()

    train_model(model, dataset, epochs=300)

    evaluate_model(model, dataset, scaler_y, sequence_length)

    future_df = forecast_future(model, X_scaled, df['Date'], scaler_X, scaler_y, sequence_length)

    plot_results(df, future_df)


if __name__ == "__main__":
    main()
