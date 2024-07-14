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
    stock_tickers.append(st.sidebar.text_input(f"Enter stock ticker {i + 1}", value=f'STOCK_{i + 1}'))

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-01-01'))

st.sidebar.write("This app performs pairs trading analysis.")

# Function to load stock data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data['Close']
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

# Load stock data
stock_data = {}
for ticker in stock_tickers:
    stock_data[ticker] = load_data(ticker, start_date, end_date)

# Check if any stock data failed to load
if not all(stock_data[ticker] is not None for ticker in stock_tickers):
    st.error("Failed to load data. Please check the stock tickers and date range.")
else:
    # Prepare DataFrame for line_chart
    df = pd.DataFrame(stock_data)

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
    def backtest(spread, window=1):
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        z_score = (spread - spread_mean) / spread_std

        long = (z_score < -1).astype(int)
        short = (z_score > 1).astype(int)
        exit = abs(z_score) < 0.5

        positions = pd.DataFrame(index=spread.index).fillna(0)
        positions['long'] = 0
        positions['short'] = 0

        positions.loc[long == 1, 'long'] = 1
        positions.loc[short == 1, 'short'] = -1
        positions['exit'] = exit

        positions.loc[positions['exit'], 'long'] = 0
        positions.loc[positions['exit'], 'short'] = 0

        positions['spread'] = spread
        positions['z_score'] = z_score

        return positions

    # Select two stocks for backtesting spread
    selected_stocks = st.sidebar.multiselect("Select Two Stocks for Spread Analysis", stock_tickers)

    if len(selected_stocks) == 2:
        # Calculate spread and perform backtest
        spread = df[selected_stocks[0]] - df[selected_stocks[1]]
        positions = backtest(spread)

        # Plot spread and signals
        fig, ax = plt.subplots(figsize=(10, 6))
        spread.plot(ax=ax, label='Spread')
        ax.plot(positions[positions['long'] == 1].index, spread[positions['long'] == 1], '^', markersize=10, color='g', label='Long Signal')
        ax.plot(positions[positions['short'] == -1].index, spread[positions['short'] == -1], 'v', markersize=10, color='r', label='Short Signal')
        ax.legend()
        st.pyplot(fig)

        # Display backtest results
        st.title("Backtest Results")
        st.write(positions)
    elif len(selected_stocks) > 0:
        st.info("Select two stocks to perform spread analysis and backtesting.")
