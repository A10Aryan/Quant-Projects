import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title = "Pairs Trading Analysis",
    layout = "wide"
)

# Sidebar for user inputs
st.sidebar.title("Select Stocks")
num_stocks = st.sidebar.slider("Number of stocks", min_value = 2, max_value = 10, value = 2)

stock_tickers = []
for i in range(num_stocks):
    stock_tickers.append(st.sidebar.text_input(f"Enter stock ticker {i + 1}", value = f'STOCK_{i + 1}'))

start_date = st.sidebar.date_input("Start Date", value = pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value = pd.to_datetime('2023-01-01'))

st.sidebar.write("This app performs pairs trading analysis.")

# Function to load stock data
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start = start, end = end)
        if 'Close' in data and not data['Close'].empty:
            return data['Close']
        else:
            return None
    except Exception as e:
        return None

# Load stock data
stock_data = {}
failed_tickers = []

for ticker in stock_tickers:
    data = load_data(ticker, start_date, end_date)
    if data is not None and isinstance(data, pd.Series) and not data.empty:
        stock_data[ticker] = data
    else:
        failed_tickers.append(ticker)

# Check if any valid data was loaded
if not stock_data:
    st.error("No valid data was loaded. Please check the stock tickers and try again.")
else:
    if failed_tickers:
        st.warning(f"Failed to load data for: {', '.join(failed_tickers)}")

    # Align all Series by date index
    df = pd.concat(stock_data.values(), axis = 1)
    df.columns = list(stock_data.keys())

    # Drop any rows with missing data
    df.dropna(inplace = True)

    # Correlation matrix
    corr_matrix = df.corr()

    st.title("Correlation Matrix")
    st.write("Interactive heatmap of correlation matrix")

    # Convert to long format for Altair
    corr_long = corr_matrix.stack().reset_index()
    corr_long.columns = ['Stock 1', 'Stock 2', 'Correlation']

    heatmap = alt.Chart(corr_long).mark_rect().encode(
        x = 'Stock 1:N',
        y = 'Stock 2:N',
        color = alt.Color('Correlation:Q', scale = alt.Scale(scheme = 'redblue', domain = [-1, 1])),
        tooltip = ['Stock 1', 'Stock 2', alt.Tooltip('Correlation:Q', format = '.2f')],
    ).properties(
        width = 600,
        height = 500,
        title = 'Correlation Heatmap'
    ).interactive()

    text = alt.Chart(corr_long).mark_text(baseline = 'middle').encode(
        x = 'Stock 1:N',
        y = 'Stock 2:N',
        text = alt.Text('Correlation:Q', format = '.2f'),
        color = alt.condition(
            alt.datum.Correlation > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )

    st.altair_chart(heatmap + text, use_container_width = True)

    # Plot price chart
    st.title("Multiple Stock Prices Over Time")
    st.line_chart(df)

    # Backtesting function
    def backtest(spread, window = 1):
        spread_mean = spread.rolling(window = window).mean()
        spread_std = spread.rolling(window = window).std()
        z_score = (spread - spread_mean) / spread_std

        long = (z_score < -1).astype(int)
        short = (z_score > 1).astype(int)
        exit = abs(z_score) < 0.5

        positions = pd.DataFrame(index = spread.index).fillna(0)
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

    # Select two stocks for spread analysis
    selected_stocks = st.sidebar.multiselect("Select Two Stocks for Spread Analysis", list(df.columns))

    if len(selected_stocks) == 2:
        spread = df[selected_stocks[0]] - df[selected_stocks[1]]
        positions = backtest(spread)

        fig, ax = plt.subplots(figsize = (10, 6))
        spread.plot(ax = ax, label = 'Spread')
        ax.plot(positions[positions['long'] == 1].index, spread[positions['long'] == 1], '^', markersize = 10, color = 'g', label = 'Long Signal')
        ax.plot(positions[positions['short'] == -1].index, spread[positions['short'] == -1], 'v', markersize = 10, color = 'r', label = 'Short Signal')
        ax.legend()
        st.pyplot(fig)

        st.title("Backtest Results")
        st.write(positions)

    elif len(selected_stocks) > 0:
        st.info("Please select exactly two stocks to perform spread analysis and backtesting.")
