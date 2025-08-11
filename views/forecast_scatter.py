# flake8: noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from io import StringIO
import random

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QThread, Signal
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import get_dataframe

# Set random seeds for reproducibility
def set_seed(seed=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(50)

# Dataset and Model classes
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        seq_x = self.X[idx:idx+self.seq_len]
        seq_y = self.y[idx+self.seq_len]
        return torch.tensor(seq_x.tolist(), dtype=torch.float32), torch.tensor(float(seq_y), dtype=torch.float32)

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

# Worker thread for model training
class ForecastWorker(QThread):
    finished = Signal(object, object, object, object, object, float, float)
    error = Signal(str)

    def __init__(self, df, target_col, seq_len=14):
        super().__init__()
        self.df = df
        self.target_col = target_col
        self.seq_len = seq_len

    def run(self):
        try:
            # Preprocess data
            df = self.df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Extract date features
            df['day'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month
            df['dayofweek'] = df['Date'].dt.dayofweek
            df['dayofyear'] = df['Date'].dt.dayofyear
            
            features = ['day', 'month', 'dayofweek', 'dayofyear']
            target = self.target_col

            X = df[features].values.astype(np.float32)
            y = df[target].values.astype(np.float32).reshape(-1, 1)
            
            # Scale data
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y).flatten()
            
            # Create dataset
            dataset = TimeSeriesDataset(X_scaled, y_scaled, self.seq_len)
            
            # Initialize and train model
            model = LSTMModel()
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(300):
                model.train()
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_X.size(0)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                y_preds_scaled = []
                for i in range(len(dataset)):
                    seq_x, _ = dataset[i]
                    pred = model(seq_x.unsqueeze(0))
                    y_preds_scaled.append(pred.item())
            
            y_true_scaled = dataset.y[self.seq_len:]
            y_preds = scaler_y.inverse_transform(np.array(y_preds_scaled).reshape(-1, 1)).flatten()
            y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
            
            # Forecast future
            future_preds = []
            current_seq = torch.tensor(X_scaled[-self.seq_len:].tolist(), dtype=torch.float32).unsqueeze(0)
            forecast_horizon = 30
            
            future_dates = df['Date'].copy()
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
                
                current_seq = torch.cat((current_seq[:, 1:, :], torch.tensor(next_features_scaled.tolist(), dtype=torch.float32).unsqueeze(0)), dim=1)
            
            future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
            forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_horizon)
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_preds)
            r2 = r2_score(y_true, y_preds)
            
            self.finished.emit(
                df['Date'], 
                df[target],
                forecast_dates,
                future_preds,
                y_preds,
                mae,
                r2
            )
            
        except Exception as e:
            self.error.emit(f"Error during forecasting: {str(e)}")

# Helper function to create Plotly view
def create_plotly_view(fig):
    """Converts a Plotly figure to a Qt web view widget."""
    html = StringIO()
    fig.write_html(html, include_plotlyjs="cdn", full_html=False)
    
    view = QWebEngineView()
    view.setHtml(html.getvalue())
    view.setMinimumHeight(500)
    return view

# Main view widget
class ForecastScatterView(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Explore the Future (LSTM)")
        title.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(title)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(200)
        control_layout.addWidget(QLabel("Target Column:"))
        control_layout.addWidget(self.target_combo)
        
        self.forecast_btn = QPushButton("Run Forecast")
        self.forecast_btn.clicked.connect(self.run_forecast)
        self.forecast_btn.setStyleSheet(
            "QPushButton {"
            "   background-color: #2e7d32;"
            "   color: white;"
            "   padding: 8px 16px;"
            "   font-weight: bold;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #1b5e20; }"
        )
        control_layout.addWidget(self.forecast_btn)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Plotly web view for visualization
        self.plot_view = QWebEngineView()
        self.plot_view.setMinimumHeight(500)
        layout.addWidget(self.plot_view)
        
        # Metrics display
        self.metrics_label = QLabel("Metrics will appear here after forecast")
        self.metrics_label.setStyleSheet("font-size: 12pt; margin-top: 10px;")
        layout.addWidget(self.metrics_label)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 10pt;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Initialize target options
        self.refresh_target_options()
    
    def refresh_target_options(self):
        df = get_dataframe()
        if df is not None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            self.target_combo.clear()
            self.target_combo.addItems(numeric_cols)
            if 'TotalRevenue' in numeric_cols:
                self.target_combo.setCurrentText('TotalRevenue')
    
    def run_forecast(self):
        df = get_dataframe()
        if df is None or df.empty:
            self.status_label.setText("No data available. Please upload data first.")
            return
            
        target_col = self.target_combo.currentText()
        if not target_col:
            self.status_label.setText("Please select a target column")
            return
            
        # Show status
        self.status_label.setText("Training model... This may take a few minutes")
        self.forecast_btn.setEnabled(False)
        
        # Create and start worker thread
        self.worker = ForecastWorker(df, target_col)
        self.worker.finished.connect(self.on_forecast_finished)
        self.worker.error.connect(self.on_forecast_error)
        self.worker.start()
    
    def on_forecast_finished(self, dates, actual, forecast_dates, forecast, predictions, mae, r2):
        # Create Plotly figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines',
            name='Historical Revenue',
            line=dict(color='blue'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Revenue</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Model predictions
        pred_dates = dates[len(dates) - len(predictions):]
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions,
            mode='lines',
            name='Model Predictions',
            line=dict(color='green', dash='dash'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Prediction</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name='30-Day Forecast',
            line=dict(color='red'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Highlight forecast region
        fig.add_vrect(
            x0=forecast_dates[0],
            x1=forecast_dates[-1],
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.target_combo.currentText()} Forecast",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Legend",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Display plot
        self.plot_view.setHtml("")  # Clear previous content
        html = StringIO()
        fig.write_html(html, include_plotlyjs="cdn", full_html=False)
        self.plot_view.setHtml(html.getvalue())
        
        # Update metrics
        self.metrics_label.setText(
            f"Training MAE: {mae:.2f} | R² Score: {r2:.2f}"
        )
        
        self.status_label.setText("Forecast completed successfully")
        self.forecast_btn.setEnabled(True)
    
    def on_forecast_error(self, error_msg):
        self.status_label.setText(error_msg)
        self.forecast_btn.setEnabled(True)