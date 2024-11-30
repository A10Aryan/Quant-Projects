# Load required libraries
library(ggplot2)

# Parameters
S_0 <- 100      # Initial stock price
K <- 105        # Strike price
T <- 1          # Time to maturity (in years)
r <- 0.05       # Risk-free rate
sigma <- 0.2    # Volatility
n_simulations <- 10000  # Number of simulations
n_steps <- 252          # Number of time steps (daily)
dt <- T / n_steps       # Time step

# Function to simulate Geometric Brownian Motion
simulate_paths <- function(S_0, r, sigma, T, n_steps, n_simulations) {
  Z <- matrix(rnorm(n_simulations * n_steps), nrow = n_simulations, ncol = n_steps)
  dt <- T / n_steps
  paths <- matrix(NA, nrow = n_simulations, ncol = n_steps + 1)
  paths[, 1] <- S_0
  
  for (t in 2:(n_steps + 1)) {
    paths[, t] <- paths[, t - 1] * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z[, t - 1])
  }
  
  return(paths)
}

# Simulate price paths
set.seed(42)
paths <- simulate_paths(S_0, r, sigma, T, n_steps, n_simulations)

# Compute Option Payoff
option_payoff <- function(paths, K, r, T, option_type = "call") {
  final_prices <- paths[, ncol(paths)]
  if (option_type == "call") {
    payoffs <- pmax(final_prices - K, 0)
  } else if (option_type == "put") {
    payoffs <- pmax(K - final_prices, 0)
  } else {
    stop("Invalid option type. Use 'call' or 'put'.")
  }
  
  discounted_payoffs <- exp(-r * T) * payoffs
  price <- mean(discounted_payoffs)
  return(list(price = price, payoffs = discounted_payoffs))
}

# Compute Call Option Price
call_result <- option_payoff(paths, K, r, T, option_type = "call")
call_price <- call_result$price

# Compute Greeks (Delta, Gamma, Vega)
compute_greeks <- function(S_0, K, r, sigma, T, n_simulations, n_steps) {
  dS <- 1  # Small change in stock price
  dSigma <- 0.01  # Small change in volatility
  
  # Base option price
  base_paths <- simulate_paths(S_0, r, sigma, T, n_steps, n_simulations)
  base_price <- option_payoff(base_paths, K, r, T)$price
  
  # Delta (change in option price with respect to S_0)
  paths_up <- simulate_paths(S_0 + dS, r, sigma, T, n_steps, n_simulations)
  price_up <- option_payoff(paths_up, K, r, T)$price
  delta <- (price_up - base_price) / dS
  
  # Gamma (second derivative of option price with respect to S_0)
  paths_down <- simulate_paths(S_0 - dS, r, sigma, T, n_steps, n_simulations)
  price_down <- option_payoff(paths_down, K, r, T)$price
  gamma <- (price_up - 2 * base_price + price_down) / (dS^2)
  
  # Vega (change in option price with respect to sigma)
  paths_vega <- simulate_paths(S_0, r, sigma + dSigma, T, n_steps, n_simulations)
  price_vega <- option_payoff(paths_vega, K, r, T)$price
  vega <- (price_vega - base_price) / dSigma
  
  return(list(delta = delta, gamma = gamma, vega = vega))
}

# Compute Greeks
greeks <- compute_greeks(S_0, K, r, sigma, T, n_simulations, n_steps)

# Plot Simulated Price Paths
plot_paths <- function(paths, n_paths = 10) {
  mat <- paths[1:n_paths, ]
  df <- data.frame(
    Time = rep(1:ncol(mat), each = nrow(mat)),
    Path = as.vector(t(mat)),
    ID = rep(1:n_paths, each = ncol(mat))
  )
  
  ggplot(df, aes(x = Time, y = Path, group = ID, color = factor(ID))) +
    geom_line() +
    theme_minimal() +
    labs(title = "Simulated Price Paths", x = "Time Steps", y = "Price")
}

plot_paths(paths)

# Output Results
cat("Call Option Price: ", call_price, "\n")
cat("Delta: ", greeks$delta, "\n")
cat("Gamma: ", greeks$gamma, "\n")
cat("Vega: ", greeks$vega, "\n")

