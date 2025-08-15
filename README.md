# ML Trading Strategy Backtester

A comprehensive Streamlit application that implements sophisticated trading strategies using machine learning. This application combines two powerful strategies from Jupyter notebooks into an interactive dashboard with object-oriented programming design.

## 🎯 Features

### 📈 Trading Strategies

1. **Moving Average Crossover with ML Filter**
   - Classic MA crossover strategy with intelligent signal filtering
   - Machine learning model predicts successful entry signals
   - Gradient boosting classifier for signal quality assessment
   - Comparative analysis between baseline and ML-enhanced strategies

2. **Volatility Forecasting Strategy**
   - Risk-adjusted position sizing based on predicted volatility
   - GARCH(1,1) and Random Forest volatility forecasting
   - Dynamic leverage adjustment with volatility targeting
   - Rolling rebalancing with transaction cost modeling

### 🎛️ Interactive Features

- **Real-time Parameter Adjustment**: Modify strategy parameters and see results instantly
- **Multi-Strategy Comparison**: Run and compare different strategies side-by-side
- **Advanced Visualizations**: Interactive Plotly charts for comprehensive analysis
- **Performance Analytics**: Detailed metrics including CAGR, Sharpe ratio, drawdowns
- **Data Export**: Download results as CSV files for further analysis

### 📊 Analytics Dashboard

- Equity curves with hover details
- Price charts with trading signals
- Volatility forecasts visualization
- Drawdown analysis
- Returns distribution analysis
- ML probability tracking

## 🚀 Quick Start

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account
2. **Visit [share.streamlit.io](https://share.streamlit.io)**
3. **Click "New app"** and connect your GitHub account
4. **Select your forked repository** and set:
   - Main file path: `app.py`
   - Python version: 3.9+ (recommended)
5. **Click "Deploy"** - the app will automatically install dependencies

**🌐 Live Demo**: [View the deployed app](https://ml-trading-strategies.streamlit.app)

### Option 2: Automated Local Setup

```bash
# Make the setup script executable
chmod +x run_app.sh

# Run the setup and launch script
./run_app.sh
```

### Option 3: Manual Local Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

## 🔧 Deployment Configuration

The app is optimized for Streamlit Cloud deployment with:

- **Configuration**: `.streamlit/config.toml` for production settings
- **Dependencies**: `requirements.txt` with pinned versions for reproducibility
- **System packages**: `packages.txt` for additional system dependencies
- **Caching**: `@st.cache_data` decorators for optimal performance
- **Error handling**: Comprehensive exception handling and user feedback

### Environment Variables (Optional)

For enhanced functionality, you can set these environment variables in Streamlit Cloud:

```bash
# Optional: Custom default settings
DEFAULT_TICKER=AAPL
DEFAULT_START_DATE=2020-01-01
```

## 📋 Requirements

- Python 3.8+
- Required packages (automatically installed):
  - streamlit
  - plotly
  - pandas
  - numpy
  - yfinance
  - scikit-learn
  - ta (technical analysis)
  - arch (GARCH modeling)
  - matplotlib

## 🏗️ Architecture

The application follows object-oriented design principles with clean separation of concerns:

```
├── app.py              # Streamlit dashboard interface
├── strategies.py       # Strategy implementations (OOP)
├── requirements.txt    # Package dependencies
├── run_app.sh         # Setup and launch script
└── README.md          # This file
```

### Core Classes

- **BaseStrategy**: Abstract base class defining the strategy interface
- **MACrossoverStrategy**: Moving average crossover with ML filtering
- **VolatilityStrategy**: Volatility forecasting with risk management
- **StrategyFactory**: Factory pattern for strategy creation
- **Metrics**: Performance metrics data container

## 📈 Strategy Details

### Moving Average Crossover Strategy

**Parameters:**
- Short/Long MA windows
- ML holding period
- Probability threshold for signal filtering
- Slippage and transaction costs

**Features:**
- Technical indicators (MACD, RSI, Bollinger Bands)
- Time-series cross-validation
- Signal quality prediction
- Risk-adjusted performance metrics

### Volatility Forecasting Strategy

**Parameters:**
- Volatility target
- Maximum leverage
- Rebalancing frequency
- Transaction costs

**Features:**
- GARCH(1,1) volatility modeling
- Random Forest ML forecasting
- Dynamic position sizing
- Risk-based portfolio management

## 🎮 Usage Guide

1. **Select Strategy**: Choose between MA Crossover or Volatility Forecasting
2. **Configure Parameters**: Adjust strategy parameters in the sidebar
3. **Set Market Data**: Enter ticker symbol and date range
4. **Run Backtest**: Click the "Run Backtest" button
5. **Analyze Results**: Review performance metrics and interactive charts
6. **Export Data**: Download results for further analysis

## 📊 Performance Metrics

The dashboard provides comprehensive performance analytics:

- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Total Return**: Cumulative performance
- **Volatility**: Annualized return volatility
- **Trade Count**: Number of executed trades

## 🔧 Customization

The modular design allows easy extension:

1. **Add New Strategies**: Inherit from `BaseStrategy` class
2. **Custom Indicators**: Extend indicator calculation methods
3. **New Visualizations**: Add custom Plotly charts
4. **Enhanced Analytics**: Implement additional performance metrics

## 📝 Example Usage

```python
# Create strategy instance
from strategies import StrategyFactory

strategy = StrategyFactory.create_strategy(
    "MA Crossover",
    ticker="AAPL",
    start_date="2020-01-01",
    short_window=20,
    long_window=100
)

# Run backtest
results = strategy.run_backtest()

# Access results
baseline_equity = results["baseline"]["equity"]
ml_equity = results["ml_filtered"]["equity"]
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new trading strategies
- Improving visualizations
- Enhancing performance metrics
- Adding new data sources
- Optimizing performance

## ⚠️ Disclaimer

This application is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk assessment before implementing any trading strategy with real capital.

## 📄 License

This project is provided as-is for educational purposes. Use at your own risk and ensure compliance with relevant financial regulations in your jurisdiction.

---

**Built with Streamlit, Plotly, and scikit-learn**
