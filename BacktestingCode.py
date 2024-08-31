# import numpy as np
import yfinance as yf
import pandas as pd

# Function to fetch stock data

def load_stock_data(tickers, start, end):
    price1 = pd.DataFrame()
    price2 = pd.DataFrame()
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start = start, end = end)
            price1[ticker] = data['Open']
            price2[ticker] = data['Close']
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    return price1, price2


# Function to calculate the metrics

def metrics(price1, price2):
    monthly_ret = (price2 - price1.shift(1)) / price1.shift(1)
    monthly_ret_avg = monthly_ret.rolling(window = 12).mean()
    monthly_vol = monthly_ret.rolling(window = 12).std()
    risk_adj_ret_avg = monthly_ret_avg / monthly_vol
    return monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg

# Function to calculate inverse volatility weights
# def weights_vol(monthly_vol):
#     vol_inv = 1 / monthly_vol
#     weights = vol_inv.div(vol_inv.sum(axis = 1), axis = 0)
#     return weights


# Function to simulate trades based on quantile selection

def simulation(price1, price2, monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg, form_prd = 12, hold_prd = 1, top_quantile = 0.1, bottom_quantile = 0.1):
    
    trades = pd.DataFrame(columns=['Year', 'Month', 'Stock', 'Position', 'Buy', 'Sell', 'Profit', 'Drawdown', 'Upside', 'Monthly Return', 'Cumulative Return', 'Mean Monthly Return', 'Monthly Volatility', 'Risk-Adjusted Mean Return'])
    ret_cum = (price2.shift(-form_prd) / price1 - 1).fillna(0)
    
    for end in ret_cum.index[form_prd:]:
        start = end - pd.DateOffset(months = form_prd)
        formation_window = ret_cum.loc[start:end]
        
        ret_avg = formation_window.mean()
        sorted_stocks = ret_avg.sort_values(ascending = False)
        
        n_top = int(len(sorted_stocks) * top_quantile)
        n_bottom = int(len(sorted_stocks) * bottom_quantile)
        
        top_stocks = sorted_stocks.index[:n_top]
        bottom_stocks = sorted_stocks.index[-n_bottom:]
        
        # Calculate inverse volatility weights for the current formation period
        # formation_volatility = monthly_vol.loc[start:end]
        # weights = weights_vol(formation_volatility)
        
        
        # Evaluating stocks for long position
        
        for stock in top_stocks:
            buy_price = price1.loc[end, stock]
            sell_date = end + pd.DateOffset(months = hold_prd)
            if sell_date in price2.index:
                sell_price = price2.loc[sell_date, stock]
                profit = (sell_price - buy_price) / buy_price
                drawdown = (price1.loc[end:sell_date, stock].min() - buy_price) / buy_price
                upside = (price1.loc[end:sell_date, stock].max() - buy_price) / buy_price
                new_trade = pd.DataFrame({
                    'Year': [end.year],
                    'Month': [end.month],
                    'Stock': [stock],
                    'Position': ['buy'],
                    'Buy': [buy_price],
                    'Sell': [sell_price],
                    'Profit': [profit],
                    'Drawdown': [drawdown],
                    'Upside': [upside],
                    'Monthly Return': [monthly_ret.loc[sell_date, stock]],
                    'Cumulative Return': [ret_cum.loc[sell_date, stock]],
                    'Mean Monthly Return': [monthly_ret_avg.loc[sell_date, stock]],
                    'Monthly Volatility': [monthly_vol.loc[sell_date, stock]],
                    'Risk-Adjusted Mean Return': [risk_adj_ret_avg.loc[sell_date, stock]]})
                
                trades = pd.concat([trades, new_trade], ignore_index = True)
        
        # Evaluating stocks for short position
        
        for stock in bottom_stocks:
            sell_price = price1.loc[end, stock]
            buy_date = end + pd.DateOffset(months = hold_prd)
            if buy_date in price2.index:
                buy_price = price2.loc[buy_date, stock]
                profit = (sell_price - buy_price) / sell_price
                drawdown = (price1.loc[end:buy_date, stock].max() - sell_price) / sell_price
                upside = (price1.loc[end:buy_date, stock].min() - sell_price) / sell_price
                new_trade = pd.DataFrame({
                    'Year': [end.year],
                    'Month': [end.month],
                    'Stock': [stock],
                    'Position': ['sell'],
                    'Buy': [buy_price],
                    'Sell': [sell_price],
                    'Profit': [profit],
                    'Drawdown': [drawdown],
                    'Upside': [upside],
                    'Monthly Return': [monthly_ret.loc[buy_date, stock]],
                    'Cumulative Return': [ret_cum.loc[buy_date, stock]],
                    'Mean Monthly Return': [monthly_ret_avg.loc[buy_date, stock]],
                    'Monthly Volatility': [monthly_vol.loc[buy_date, stock]],
                    'Risk-Adjusted Mean Return': [risk_adj_ret_avg.loc[buy_date, stock]]})
                
                trades = pd.concat([trades, new_trade], ignore_index = True)
                
    return trades


# Function to run backtest with specified parameters

def run_backtest(start, end, tickers, price1, price2, monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg, params):
    form_prd_range = params.get('form_prd', [12])
    hold_prd_range = params.get('hold_prd', [1])
    top_quantile_range = params.get('top_quantile', [0.1])
    bottom_quantile_range = params.get('bottom_quantile', [0.1])
    

    results = []
    best_performance = -float('inf')
    best_params = None
    
    for form_prd in form_prd_range:
        for hold_prd in hold_prd_range:
            for top_quantile in top_quantile_range:
                for bottom_quantile in bottom_quantile_range:
                    trades = simulation(price1, price2, monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg, form_prd = form_prd, hold_prd = hold_prd, top_quantile = top_quantile, bottom_quantile = bottom_quantile)
                    
                    portfolio_summary = trades.groupby(['Year', 'Month']).agg({
                        'Profit': 'sum',
                        'Stock': 'count'}).reset_index()
                    
                    portfolio_summary.columns = ['Year', 'Month', 'Portfolio Returns', 'Number of Trades']


                    # Calculate the weightage of each stock for each month's portfolio
                    
                    weightage = trades.groupby(['Year', 'Month', 'Stock'])['Profit'].sum().unstack(fill_value = 0)
                    weightage = weightage.div(weightage.sum(axis = 1), axis = 0).reset_index()


                    # Merge weightage with portfolio summary
                    
                    portfolio_summary = portfolio_summary.merge(weightage, on = ['Year', 'Month'], how = 'left')
                    
                    
                    # Calculate performance metric (e.g., cumulative return)
                    cumulative_return = portfolio_summary['Portfolio Returns'].sum()
                    
                    results.append({
                        'Formation Period': form_prd,
                        'Holding Period': hold_prd,
                        'Top Quantile': top_quantile,
                        'Bottom Quantile': bottom_quantile,
                        'Trades': trades,
                        'Portfolio Summary': portfolio_summary,
                        'Cumulative Return': cumulative_return})
                    
                    # Check if current combination is the best
                    if cumulative_return > best_performance:
                       best_performance = cumulative_return
                       best_params = {
                           'Formation Period': form_prd,
                           'Holding Period': hold_prd,
                           'Top Quantile': top_quantile,
                           'Bottom Quantile': bottom_quantile
                       }
   
   # Print the best parameter combination and its performance metrics
   
    if best_params:
       print("Best Parameter Combination:")
       print(f"Formation Period: {best_params['Formation Period']}")
       print(f"Holding Period: {best_params['Holding Period']}")
       print(f"Top Quantile: {best_params['Top Quantile']}")
       print(f"Bottom Quantile: {best_params['Bottom Quantile']}")
       print(f"Cumulative Return: {best_performance}")

    return results


# Function to save backtest results to Excel

def save_backtest_results_to_excel(trades, portfolio_summary):
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    
    with pd.ExcelWriter('backtest_results.xlsx', engine = 'xlsxwriter') as writer:
        
        # Write the trades DataFrame to the 'Trades' sheet
        
        trades.to_excel(writer, sheet_name = 'Trades', index = False)

        # Write each stock's trades to its own sheet
        
        for stock in trades['Stock'].unique():
            stock_trades = trades[trades['Stock'] == stock]
            stock_trades.to_excel(writer, sheet_name = stock, index = False)

        # Write the portfolio summary DataFrame to the 'Overall Portfolio Sheet'
        
        portfolio_summary.to_excel(writer, sheet_name = 'Portfolio Sheet', index = False)

# Define parameters for optimization

params = {
    'form_prd': [6, 9, 12, 15, 18],
    'hold_prd': [1, 2, 3],
    'top_quantile': [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],  # Adjust quantile values as needed
    'bottom_quantile': [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]}


# Specify tickers and date range

tickers = ["ASIANPAINT.NS", "BRITANNIA.NS", "CIPLA.NS", "EICHERMOT.NS", "NESTLEIND.NS", 
    "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ITC.NS", 
    "LT.NS", "M&M.NS", "RELIANCE.NS", "TATACONSUM.NS", "TATAMOTORS.NS", 
    "TATASTEEL.NS", "WIPRO.NS", "APOLLOHOSP.NS", "DRREDDY.NS", "TITAN.NS", 
    "SBIN.NS", "SHRIRAMCIT.NS", "BPCL.NS", "KOTAKBANK.NS", "INFY.NS", 
    "BAJFINANCE.NS", "ADANIENT.NS", "SUNPHARMA.NS", "JSWSTEEL.NS", "HDFCBANK.NS",
    "TCS.NS", "ICICIBANK.NS", "POWERGRID.NS", "MARUTI.NS", "INDUSINDBK.NS",
    "AXISBANK.NS", "HCLTECH.NS", "ONGC.NS", "NTPC.NS", "COALINDIA.NS",
    "BHARTIARTL.NS", "TECHM.NS", "MINDTREE.NS", "DIVISLAB.NS", "ADANIPORTS.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "ULTRACEMCO.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS"]

start_date = '2020-01-01'
end_date = '2024-06-30'


# Load the stock data

price1, price2 = load_stock_data(tickers, start_date, end_date)

# Calculate metrics

monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg = metrics(price1, price2)

# Run backtest

results = run_backtest(start_date, end_date, tickers, price1, price2, monthly_ret, monthly_ret_avg, monthly_vol, risk_adj_ret_avg, params)

# Save results to Excel

for result in results:
    save_backtest_results_to_excel(result['Trades'], result['Portfolio Summary'])
   
print("Backtest completed successfully. Results saved to 'backtest_results.xlsx'.")
