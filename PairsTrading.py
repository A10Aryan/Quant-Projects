import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Pairs Trading Analysis",
    layout="wide"
)

# Sidebar for user inputs
st.sidebar.title("Select Stocks")
num_stocks = st.sidebar.slider("Number of stocks", min_value=2, max_value=10, value=2)

stock_tickers = []
for i in range(num_stocks):
    ticker = st.sidebar.text_input(f"Enter stock ticker {i + 1}", value=f'AAPL' if i == 0 else f'MSFT' if i == 1 else f'STOCK_{i + 1}')
    stock_tickers.append(ticker.strip().upper())

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-01-01'))

st.sidebar.write("This app performs pairs trading analysis.")

# Function to load stock data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.warning(f"No data found for {ticker}. Please check the ticker symbol and date range.")
            return None
        
        # Extract the Close price and ensure it's a Series
        close_data = data['Close']
        if isinstance(close_data, pd.DataFrame):
            # If multiple columns, take the first one
            close_data = close_data.iloc[:, 0]
        
        # Ensure we have a proper Series with datetime index
        if not isinstance(close_data.index, pd.DatetimeIndex):
            close_data.index = pd.to_datetime(close_data.index)
            
        return close_data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

# Load stock data
st.write("Loading stock data...")
stock_data = {}
valid_tickers = []

for ticker in stock_tickers:
    if ticker and ticker not in ['STOCK_1', 'STOCK_2', 'STOCK_3', 'STOCK_4', 'STOCK_5', 'STOCK_6', 'STOCK_7', 'STOCK_8', 'STOCK_9', 'STOCK_10']:
        data = load_data(ticker, start_date, end_date)
        if data is not None and not data.empty and len(data) > 1:
            stock_data[ticker] = data
            valid_tickers.append(ticker)
            st.success(f"✓ Loaded data for {ticker}: {len(data)} data points")
        else:
            st.warning(f"⚠ Skipped {ticker}: insufficient data")

# Check if we have enough valid data
if len(stock_data) < 2:
    st.error("Need at least 2 stocks with valid data. Please enter valid stock tickers (e.g., AAPL, MSFT, GOOGL).")
    st.info("Try using these popular tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA")
else:
    st.success(f"Successfully loaded data for {len(stock_data)} stocks!")
    
    # Create DataFrame with proper alignment
    try:
        # Get the common date range for all stocks
        all_dates = None
        for ticker, data in stock_data.items():
            if all_dates is None:
                all_dates = data.index
            else:
                all_dates = all_dates.intersection(data.index)
        
        if len(all_dates) == 0:
            st.error("No overlapping dates found between the selected stocks.")
        else:
            # Create DataFrame with aligned data
            df_data = {}
            for ticker, data in stock_data.items():
                df_data[ticker] = data.reindex(all_dates)
            
            df = pd.DataFrame(df_data)
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                st.error("No valid data remaining after alignment and cleaning.")
            else:
                st.write(f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Calculate correlation matrix
                corr_matrix = df.corr()

                # Plot correlation matrix using Altair
                st.title("Correlation Matrix")
                st.write("Interactive heatmap of correlation matrix")

                # Convert correlation matrix to long format for Altair plotting
                corr_long = corr_matrix.stack().reset_index()
                corr_long.columns = ['Stock 1', 'Stock 2', 'Correlation']

                # Create heatmap with Altair
                heatmap = alt.Chart(corr_long).mark_rect().encode(
                    x='Stock 1:N',
                    y='Stock 2:N',
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                    tooltip=['Stock 1', 'Stock 2', alt.Tooltip('Correlation:Q', format='.2f')],
                ).properties(
                    width=600,
                    height=500,
                    title='Correlation Heatmap'
                ).interactive()

                # Add text marks for correlation values in each box
                text = alt.Chart(corr_long).mark_text(baseline='middle').encode(
                    x='Stock 1:N',
                    y='Stock 2:N',
                    text=alt.Text('Correlation:Q', format='.2f'),
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('white'),
                        alt.value('black')
                    )
                )

                heatmap_with_text = (heatmap + text).properties(
                    title='Correlation Heatmap with Values'
                )

                st.altair_chart(heatmap_with_text, use_container_width=True)

                # Display line chart with different colors
                st.title("Multiple Stock Prices Over Time")
                st.line_chart(df)

                # Backtesting function
                def backtest(spread, window=20):
                    spread_mean = spread.rolling(window=window).mean()
                    spread_std = spread.rolling(window=window).std()
                    
                    # Handle division by zero
                    z_score = pd.Series(index=spread.index, dtype=float)
                    mask = spread_std != 0
                    z_score[mask] = (spread[mask] - spread_mean[mask]) / spread_std[mask]
                    z_score[~mask] = 0

                    long = (z_score < -1).astype(int)
                    short = (z_score > 1).astype(int)
                    exit_signal = abs(z_score) < 0.5

                    positions = pd.DataFrame(index=spread.index)
                    positions['long'] = 0
                    positions['short'] = 0

                    positions.loc[long == 1, 'long'] = 1
                    positions.loc[short == 1, 'short'] = -1
                    positions['exit'] = exit_signal

                    positions.loc[positions['exit'], 'long'] = 0
                    positions.loc[positions['exit'], 'short'] = 0

                    positions['spread'] = spread
                    positions['z_score'] = z_score

                    return positions

                # Select two stocks for backtesting spread
                selected_stocks = st.sidebar.multiselect("Select Two Stocks for Spread Analysis", valid_tickers)

                if len(selected_stocks) == 2:
                    # Calculate spread and perform backtest
                    spread = df[selected_stocks[0]] - df[selected_stocks[1]]
                    positions = backtest(spread)

                    # Plot spread and signals
                    st.title(f"Spread Analysis: {selected_stocks[0]} - {selected_stocks[1]}")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot spread
                    spread.plot(ax=ax1, label='Spread', color='blue')
                    long_signals = positions[positions['long'] == 1]
                    short_signals = positions[positions['short'] == -1]
                    
                    if not long_signals.empty:
                        ax1.plot(long_signals.index, spread[long_signals.index], '^', 
                                markersize=8, color='green', label='Long Signal')
                    if not short_signals.empty:
                        ax1.plot(short_signals.index, spread[short_signals.index], 'v', 
                                markersize=8, color='red', label='Short Signal')
                    
                    ax1.set_title('Price Spread with Trading Signals')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Plot z-score
                    positions['z_score'].plot(ax=ax2, label='Z-Score', color='purple')
                    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Short Threshold')
                    ax2.axhline(y=-1, color='green', linestyle='--', alpha=0.7, label='Long Threshold')
                    ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Exit Threshold')
                    ax2.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.7)
                    ax2.set_title('Z-Score with Trading Thresholds')
                    ax2.legend()
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Display backtest summary
                    st.title("Backtest Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Long Signals", len(long_signals))
                    with col2:
                        st.metric("Total Short Signals", len(short_signals))
                    with col3:
                        st.metric("Spread Std Dev", f"{spread.std():.2f}")
                    
                    # Show detailed positions (last 10 rows)
                    st.subheader("Recent Trading Positions")
                    st.dataframe(positions.tail(10))
                    
                elif len(selected_stocks) > 2:
                    st.info("Please select exactly two stocks for spread analysis.")
                elif len(selected_stocks) == 1:
                    st.info("Please select one more stock for spread analysis.")
                else:
                    st.info("Select two stocks from the sidebar to perform spread analysis and backtesting.")
                    
    except Exception as e:
        st.error(f"Error creating DataFrame: {str(e)}")
        st.info("This might be due to mismatched data formats or empty datasets. Try using different stock tickers or date ranges.")
