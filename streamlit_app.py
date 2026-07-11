import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from src.utils import fetch_stock_data, add_technical_indicators
from src.trainer import train_model
from src.predictor import predict_future, evaluate_model
from src.config import DEFAULT_TICKER, MODELS_DIR

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Stock Price Prediction AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS removed for compatibility ---


# --- Helper Functions for Visualization ---
def plot_historical_data(df, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Historical Price', 'Volume'),
                        row_width=[0.2, 0.7])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='Candlestick'), row=1, col=1)
    
    # Moving Averages
    if 'SMA_50' in df.columns and 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='blue', width=1), name='EMA 20'), row=1, col=1)

    # Volume bar chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, marker_color='cyan'), row=2, col=1)

    fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_predictions(historical_df, predictions_df, ticker):
    fig = go.Figure()
    
    # Plot last 100 days of historical data for context
    context_df = historical_df.tail(100)
    
    fig.add_trace(go.Scatter(x=context_df.index, y=context_df['Close'], 
                             mode='lines', name='Historical Close', line=dict(color='cyan')))
    
    # Plot predicted data
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Close'], 
                             mode='lines+markers', name='Predicted Close', line=dict(color='red', dash='dash')))
                             
    fig.update_layout(title=f"{ticker} Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      height=500,
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title="Model Training Loss",
                      xaxis_title="Epoch",
                      yaxis_title="Loss (MSE)",
                      height=400,
                      template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)


# --- Main Application ---
def main():
    st.title("📈 Stock Price Prediction System")
    st.markdown("Predict future stock prices using Deep LSTM Neural Networks and advanced technical indicators.")

    # Sidebar setup
    st.sidebar.header("📊 Stock Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", DEFAULT_TICKER).upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Model Hyperparameters")
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Prediction Settings")
    days_to_predict = st.sidebar.selectbox("Days to Predict", [1, 7, 30], index=1)
    
    # State management
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
        
    # Main area tabs
    tab1, tab2, tab3 = st.tabs(["Data Visualization", "Model Training", "Future Predictions"])
    
    # 1. Fetch Data
    try:
        if st.sidebar.button("Fetch Data"):
            with st.spinner(f"Fetching data for {ticker}..."):
                raw_df = fetch_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                st.session_state["data"] = add_technical_indicators(raw_df)
                st.success("Data fetched and processed successfully!")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

    df = st.session_state.get("data")
    
    with tab1:
        if df is not None:
            st.subheader(f"Historical Data & Technical Indicators: {ticker}")
            plot_historical_data(df, ticker)
            
            with st.expander("View Raw Data"):
                st.dataframe(df.tail(20))
        else:
            st.info("Click 'Fetch Data' in the sidebar to visualize historical data.")

    with tab2:
        if df is not None:
            st.subheader("Train LSTM Model")
            st.write("Train a deep neural network on the fetched historical data.")
            
            model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.keras")
            if os.path.exists(model_path):
                st.success(f"A trained model for {ticker} already exists. You can retrain it or go directly to predictions.")
                st.session_state["model_trained"] = True
                
            if st.button("Train Model"):
                with st.spinner("Training model... This might take a few minutes. Check terminal for epoch progress."):
                    try:
                        history, _ = train_model(
                            ticker=ticker,
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            epochs=epochs,
                            batch_size=batch_size
                        )
                        st.session_state["model_trained"] = True
                        st.success("Model trained successfully!")
                        plot_training_history(history.history)
                        
                        # Evaluate on recent data
                        st.subheader("Model Evaluation")
                        metrics, y_true, y_pred = evaluate_model(ticker, df)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("RMSE", f"{metrics['RMSE']:.2f}")
                        col2.metric("MAE", f"{metrics['MAE']:.2f}")
                        col3.metric("MSE", f"{metrics['MSE']:.2f}")
                        col4.metric("R² Score", f"{metrics['R2']:.4f}")
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
        else:
            st.info("Please fetch data first.")

    with tab3:
        if st.session_state.get("model_trained"):
            st.subheader(f"Forecast for the next {days_to_predict} days")
            if st.button("Generate Prediction"):
                with st.spinner("Generating predictions..."):
                    try:
                        pred_df = predict_future(ticker, df, days_to_predict=days_to_predict)
                        
                        plot_predictions(df, pred_df, ticker)
                        
                        st.subheader("Prediction Data")
                        st.dataframe(pred_df)
                        
                        # Download CSV
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name=f'{ticker}_predictions.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
        else:
            st.info("Please train the model first before generating predictions.")

if __name__ == "__main__":
    main()
