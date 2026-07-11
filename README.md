# Stock Price Prediction System 📈

A complete, production-ready Deep Learning solution for stock price prediction using LSTM Neural Networks. This system fetches historical market data, generates advanced technical indicators, trains an optimized LSTM model, and serves predictions via an interactive Streamlit dashboard.

## 🌟 Features

- **Automated Data Pipeline**: Fetch historical data for any ticker using `yfinance`.
- **Advanced Feature Engineering**: Automatically computes technical indicators (RSI, MACD, EMA, SMA, Bollinger Bands, ATR) as model features.
- **Deep LSTM Architecture**: Built with TensorFlow/Keras, featuring Dropout for regularization and configured for sequence prediction.
- **Robust Training Setup**: Includes EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
- **Interactive Dashboard**: A beautiful, responsive Streamlit dashboard with Plotly charts (Candlestick, Volume, Moving Averages).
- **Multi-Horizon Predictions**: Predict the next day, next 7 days, or next 30 days.
- **Production Code Quality**: Modular design, Type Hinting, Docstrings, standard Python Logging, and custom exception handling.

## 🏗️ Architecture & Folder Structure

```
Stock-Market-LSTM/
│
├── data/                  # Stores downloaded historical CSV data
├── models/                # Stores saved models (.keras) and fitted scalers (.pkl)
├── notebooks/             # For exploratory data analysis (optional)
├── src/                   # Core ML Backend modules
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and global settings
│   ├── data_loader.py     # Alternative data loading module (handled in utils)
│   ├── preprocessing.py   # Data scaling, sequence generation, cleaning
│   ├── model.py           # Deep LSTM Architecture
│   ├── trainer.py         # Training loop & callbacks
│   ├── predictor.py       # Inference and Evaluation metrics
│   ├── utils.py           # Logging, Fetching data, Technical Indicators
│
├── streamlit_app.py       # Streamlit Dashboard UI
├── train.py               # CLI script to train the model
├── predict.py             # CLI script to make predictions
├── requirements.txt       # Dependencies
├── README.md              # Project Documentation
├── .gitignore             
└── LICENSE                
```

## 🖼️ Screenshots
*(Add screenshots of your Streamlit Dashboard here)*
- Dashboard Overview
- Interactive Candlestick Charts
- Prediction Graphs

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock-Market-LSTM.git
   cd Stock-Market-LSTM
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Streamlit Dashboard (Recommended)

The easiest way to interact with the project is through the Streamlit UI.

```bash
streamlit run streamlit_app.py
```
This will open the web app in your browser where you can:
- Select a stock ticker and date range.
- Configure hyperparameters (Epochs, Batch Size).
- Train the model dynamically.
- Generate and visualize future predictions.

### Command Line Interface

You can also run the ML pipeline from the terminal.

**To train the model:**
```bash
python train.py --ticker AAPL --epochs 50 --batch_size 32
```

**To predict future prices:**
```bash
python predict.py --ticker AAPL --days 7
```

## 🧠 Model Explanation

The core model is a multi-layer Long Short-Term Memory (LSTM) network.
LSTMs are excellent for time-series forecasting as they can capture long-term dependencies in the data.
1. The historical data (Close prices + Technical Indicators) is scaled using `MinMaxScaler`.
2. The data is divided into sequences (e.g., the last 60 days of data).
3. The LSTM learns the patterns from these 60-day sequences to predict the price on the 61st day.
4. During inference for multi-day predictions, the model uses its own predictions as inputs to forecast further into the future.

## 🔮 Future Improvements

- Implementation of GRU and CNN-LSTM architectures for comparison.
- Adding SHAP (SHapley Additive exPlanations) for model explainability.
- Multi-stock correlation features.
- Deployment via Docker.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
