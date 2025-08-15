import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import sys
import traceback
warnings.filterwarnings("ignore")

# Check for required packages
try:
    from strategies import StrategyFactory, Metrics
except ImportError as e:
    st.error(f"Error importing strategies module: {e}")
    st.stop()

# Validate essential packages
missing_packages = []
try:
    import yfinance
except ImportError:
    missing_packages.append("yfinance")

try:
    import sklearn
except ImportError:
    missing_packages.append("scikit-learn")

try:
    import ta
except ImportError:
    missing_packages.append("ta")

if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.error("Please install missing packages and restart the app.")
    st.stop()

# Streamlit Cache for data loading
@st.cache_data(ttl=3600, show_spinner=True)
def load_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data with caching for better performance"""
    try:
        import yfinance as yf
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Page configuration
st.set_page_config(
    page_title="ML Trading Strategy Backtester",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def format_metric(value, metric_type):
    """Format metrics for display"""
    if pd.isna(value):
        return "N/A"
    
    if metric_type in ["cagr", "total_return", "max_dd", "realized_vol"]:
        return f"{value:.2%}"
    elif metric_type == "sharpe":
        return f"{value:.2f}"
    elif metric_type == "trade_count":
        return f"{int(value) if not pd.isna(value) else 'N/A'}"
    else:
        return f"{value:.4f}"

def plot_equity_curves(results_dict, title="Strategy Performance"):
    """Create interactive equity curve plot"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, df) in enumerate(results_dict.items()):
        if "equity" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["equity"],
                mode='lines',
                name=name.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Equity: %{{y:.3f}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Equity (Growth of $1)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    return fig

def plot_price_with_signals(df, strategy_type="MA Crossover"):
    """Plot price chart with trading signals"""
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Price with Signals', 'Equity Curve'),
                       vertical_spacing=0.08,
                       row_heights=[0.7, 0.3])
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode='lines', name='Price',
        line=dict(color='black', width=1),
        hovertemplate="Price: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    if strategy_type == "MA Crossover":
        # Moving averages
        if "sma_s" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_s"],
                mode='lines', name='Short MA',
                line=dict(color='blue', width=1.5),
                hovertemplate="Short MA: $%{y:.2f}<extra></extra>"
            ), row=1, col=1)
        
        if "sma_l" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["sma_l"],
                mode='lines', name='Long MA',
                line=dict(color='red', width=1.5),
                hovertemplate="Long MA: $%{y:.2f}<extra></extra>"
            ), row=1, col=1)
        
        # Buy/sell signals
        if "signal" in df.columns:
            buy_signals = df[df["signal"] == 1]
            sell_signals = df[df["signal"] == -1]
            
            if len(buy_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index, y=buy_signals["Close"],
                    mode='markers', name='Buy Signal',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    hovertemplate="Buy: $%{y:.2f}<extra></extra>"
                ), row=1, col=1)
            
            if len(sell_signals) > 0:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index, y=sell_signals["Close"],
                    mode='markers', name='Sell Signal',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    hovertemplate="Sell: $%{y:.2f}<extra></extra>"
                ), row=1, col=1)
    
    # Equity curve
    if "equity" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["equity"],
            mode='lines', name='Equity',
            line=dict(color='purple', width=2),
            hovertemplate="Equity: %{y:.3f}<extra></extra>"
        ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    
    return fig

def plot_volatility_forecasts(forecasts_dict, title="Volatility Forecasts"):
    """Plot volatility forecasts comparison"""
    fig = go.Figure()
    
    colors = {'realized': '#1f77b4', 'garch': '#ff7f0e', 'ml': '#2ca02c'}
    
    for name, series in forecasts_dict.items():
        if series is not None and not series.isna().all():
            # Annualize if needed
            if name != 'realized':
                series = series * np.sqrt(252)
            
            fig.add_trace(go.Scatter(
                x=series.dropna().index,
                y=series.dropna(),
                mode='lines',
                name=name.replace('_', ' ').title(),
                line=dict(color=colors.get(name, '#9467bd'), width=2),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Volatility: %{{y:.2%}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400
    )
    
    return fig

def plot_drawdowns(results_dict):
    """Plot drawdown comparison"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, df) in enumerate(results_dict.items()):
        if "equity" in df.columns:
            dd = (df["equity"] / df["equity"].cummax() - 1) * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=dd,
                mode='lines',
                name=name.replace('_', ' ').title(),
                fill='tonexty' if i == 0 else 'tonexty',
                line=dict(color=colors[i % len(colors)], width=1),
                hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>"
            ))
    
    fig.update_layout(
        title=dict(text="Strategy Drawdowns", x=0.5, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">ML Trading Strategy Backtester</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for strategy selection and parameters
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Strategy Configuration</div>', 
                    unsafe_allow_html=True)
        
        # Strategy selection
        strategy_name = st.selectbox(
            "Select Strategy",
            StrategyFactory.get_available_strategies(),
            index=0
        )
        
        st.markdown("---")
        
        # Common parameters
        st.subheader("Market Data")
        ticker = st.text_input("Ticker Symbol", value="MSFT", help="Enter stock ticker (e.g., MSFT, AAPL, SPY)")
        
        # Store ticker in session state for caching
        st.session_state.ticker = ticker
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2015, 1, 1),
                max_value=datetime.now() - timedelta(days=30)
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                min_value=start_date + timedelta(days=365),
                max_value=datetime.now()
            )
        
        # Store dates in session state for caching
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        
        st.markdown("---")
        
        # Strategy-specific parameters
        if strategy_name == "MA Crossover":
            st.subheader("MA Crossover Parameters")
            short_window = st.slider("Short Window", 5, 50, 20)
            long_window = st.slider("Long Window", 50, 200, 100)
            ml_hold_days = st.slider("ML Holding Period (days)", 5, 30, 10)
            proba_threshold = st.slider("ML Probability Threshold", 0.5, 0.8, 0.55, 0.01)
            slippage_bps = st.slider("Slippage (bps)", 0.0, 10.0, 1.0, 0.1)
            
            strategy_params = {
                "ticker": ticker,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "short_window": short_window,
                "long_window": long_window,
                "ml_hold_days": ml_hold_days,
                "proba_threshold": proba_threshold,
                "slippage_bps": slippage_bps
            }
            
        elif strategy_name == "Volatility Forecasting":
            st.subheader("Volatility Strategy Parameters")
            vol_target = st.slider("Volatility Target", 0.05, 0.30, 0.15, 0.01)
            max_leverage = st.slider("Max Leverage", 1.0, 5.0, 2.0, 0.1)
            rebalance_freq = st.slider("Rebalance Frequency (days)", 1, 20, 5)
            slippage_bps = st.slider("Slippage (bps)", 0.0, 10.0, 1.0, 0.1)
            
            strategy_params = {
                "ticker": ticker,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "vol_target": vol_target,
                "max_leverage": max_leverage,
                "rebalance_freq": rebalance_freq,
                "slippage_bps": slippage_bps
            }
        
        st.markdown("---")
    run_backtest = st.button("Run Backtest", type="primary", use_container_width=True)
    
    # Main content area
    if run_backtest:
        with st.spinner("Running backtest... This may take a moment."):
            try:
                # Create and run strategy
                strategy = StrategyFactory.create_strategy(strategy_name, **strategy_params)
                results = strategy.run_backtest()
                
                # Store in session state
                st.session_state.results = results
                st.session_state.strategy = strategy
                st.session_state.strategy_name = strategy_name
                st.success("Backtest completed successfully!")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                if st.checkbox("Show detailed error information"):
                    st.code(traceback.format_exc())
                return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state.results
        strategy = st.session_state.strategy
        strategy_name = st.session_state.strategy_name

        # --- ML availability check and user banner (MA Crossover only) ---
        if strategy_name == "MA Crossover":
            baseline_df = results.get("baseline") if isinstance(results.get("baseline"), pd.DataFrame) else None
            ml_prob_series = results.get("ml_probabilities") if results.get("ml_probabilities") is not None else None

            # Use getattr on strategy object
            ml_hold_days = getattr(strategy, "ml_hold_days", 10)
            proba_threshold = getattr(strategy, "proba_threshold", 0.55)

            events_count = 0
            if baseline_df is not None:
                if len(baseline_df) > ml_hold_days:
                    cutoff_idx = baseline_df.index[-ml_hold_days]
                    events_count = int(((baseline_df["signal"] == 1) & (baseline_df.index <= cutoff_idx)).sum())
                else:
                    events_count = 0

            proba_mean = float(ml_prob_series.dropna().mean()) if ml_prob_series is not None and not ml_prob_series.dropna().empty else float("nan")
            proba_std = float(ml_prob_series.dropna().std()) if ml_prob_series is not None and not ml_prob_series.dropna().empty else float("nan")

            if events_count < 10:
                st.warning(
                    f"ML filter unavailable — not enough historical signals\n\n"
                    f"We need at least 10 past entry events to train the ML filter. Found {events_count}. "
                    "Try an earlier Start Date, reduce the Long MA or holding period, or lower the ML threshold to see ML-filtered results."
                )
                with st.expander("ML training details"):
                    st.write(f"Training set size: {events_count} labeled entries")
                    st.write(f"ML holding period: {ml_hold_days} days")
                    st.write(f"Probability threshold: {proba_threshold}")
                    st.write("Model: Gradient Boosting (time-series CV, 5 splits)")
                    st.write("Recommended actions: start earlier, reduce long MA, shorten ML holding period, or lower the threshold.")
            else:
                st.success(f"ML filter active — trained on {events_count} entry events")
                with st.expander("ML training details"):
                    st.write(f"Training set size: {events_count} labeled entries")
                    st.write(f"ML holding period: {ml_hold_days} days")
                    st.write(f"Probability threshold: {proba_threshold}")
                    if not np.isnan(proba_mean):
                        st.write(f"Sample ML probability: mean={proba_mean:.3f}, std={proba_std:.3f}")
                    st.write("Model: Gradient Boosting (time-series CV, 5 splits)")

        # Performance metrics section
        st.subheader("Performance Metrics")

        # Calculate metrics for all strategies
        metrics_data = []
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and "equity" in df.columns:
                metrics = strategy.compute_metrics(df)
                metrics_data.append({
                    "Strategy": name.replace('_', ' ').title(),
                    "CAGR": format_metric(metrics.cagr, "cagr"),
                    "Sharpe": format_metric(metrics.sharpe, "sharpe"),
                    "Max DD": format_metric(metrics.max_dd, "max_dd"),
                    "Total Return": format_metric(metrics.total_return, "total_return"),
                    "Volatility": format_metric(metrics.realized_vol, "realized_vol") if hasattr(metrics, 'realized_vol') else "N/A",
                    "Trades": format_metric(metrics.trade_count, "trade_count") if hasattr(metrics, 'trade_count') else "N/A"
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No metrics available to display.")

        # --- Additional diagnostics from notebooks: win-rates, selection stats, forecast errors ---
        st.subheader("Model & Trade Diagnostics")

        # MA Crossover diagnostics
        if strategy_name == "MA Crossover":
            st.markdown("### MA Crossover Diagnostics")
            baseline_df = results.get("baseline") if isinstance(results.get("baseline"), pd.DataFrame) else None
            ml_prob_series = results.get("ml_probabilities") if results.get("ml_probabilities") is not None else None

            if baseline_df is not None:
                # forward returns used for labels
                fwd_ret_col = "fwd_ret_N"
                baseline_df[fwd_ret_col] = baseline_df["Close"].pct_change(ml_hold_days).shift(-ml_hold_days)
                buy_signals = baseline_df[baseline_df["signal"] == 1]
                sell_signals = baseline_df[baseline_df["signal"] == -1]
                n_buy = len(buy_signals)
                n_sell = len(sell_signals)

                # baseline win rate on entry events
                events = buy_signals.copy()
                events = events.dropna(subset=[fwd_ret_col])
                baseline_win_rate = float((events[fwd_ret_col] > 0).mean()) if len(events) else float('nan')

                # ML selection stats
                selection_rate = float('nan')
                selection_win_rate = float('nan')
                n_selected = 0
                proba_at_events_mean = float('nan')
                proba_at_events_std = float('nan')
                if ml_prob_series is not None and len(events):
                    proba_at_events = ml_prob_series.reindex(events.index).dropna()
                    proba_at_events_mean = float(proba_at_events.mean()) if not proba_at_events.empty else float('nan')
                    proba_at_events_std = float(proba_at_events.std()) if not proba_at_events.empty else float('nan')
                    selected_idx = proba_at_events[proba_at_events > proba_threshold].index
                    n_selected = len(selected_idx)
                    selection_rate = n_selected / n_buy if n_buy else float('nan')
                    selected_events = events.reindex(selected_idx).dropna(subset=[fwd_ret_col])
                    selection_win_rate = float((selected_events[fwd_ret_col] > 0).mean()) if len(selected_events) else float('nan')

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buy Signals", f"{n_buy}")
                    st.metric("Sell Signals", f"{n_sell}")
                with col2:
                    st.metric("Baseline Win Rate", f"{baseline_win_rate:.2%}" if not pd.isna(baseline_win_rate) else "N/A")
                    st.metric("Selected Trades", f"{n_selected}")
                with col3:
                    st.metric("Selection Rate", f"{selection_rate:.2%}" if not pd.isna(selection_rate) else "N/A")
                    st.metric("Selection Win Rate", f"{selection_win_rate:.2%}" if not pd.isna(selection_win_rate) else "N/A")

                with st.expander("ML probability details"):
                    st.write(f"Mean ML probability at entry events: {proba_at_events_mean:.3f} (std {proba_at_events_std:.3f})")
                    st.write(f"Probability threshold: {proba_threshold}")
                    st.write("Notes: Baseline win rate is measured over all buy signals; selection win rate is measured only on buys where ML proba > threshold.")

        elif strategy_name == "Volatility Forecasting":
            st.markdown("### Volatility Strategy Diagnostics")
            forecasts = results.get("forecasts", {})
            realized = forecasts.get("realized") if forecasts is not None else None
            garch_preds = forecasts.get("garch") if forecasts is not None else None
            ml_preds = forecasts.get("ml") if forecasts is not None else None

            def forecast_stats(pred):
                if pred is None or pred.isna().all() or realized is None:
                    return None
                # Annualize predictions to match notebook conventions (realized is rv_20d already annualized)
                try:
                    pred_ann = pred * np.sqrt(252)
                except Exception:
                    pred_ann = pred

                valid = pd.concat([realized.rename('realized'), pred_ann.rename('pred')], axis=1).dropna()
                if valid.shape[0] < 10:
                    return None
                y = valid.iloc[:, 0].values
                yhat = valid.iloc[:, 1].values
                mse = float(((y - yhat) ** 2).mean())
                rmse = float(np.sqrt(mse))
                ss_res = float(((y - yhat) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
                return {"rmse": rmse, "r2": r2, "n": len(valid), "mean_pred": float(yhat.mean()), "mean_realized": float(y.mean())}

            # Forecast performance
            garch_stats = forecast_stats(garch_preds)
            ml_stats = forecast_stats(ml_preds)

            # Extended stats: MAE and Pearson correlation + prepare data for scatter plots
            def forecast_extended_stats(pred):
                if pred is None or pred.isna().all() or realized is None:
                    return None, None
                # Annualize predictions to match realized (rv_20d) used in notebook
                try:
                    pred_ann = pred * np.sqrt(252)
                except Exception:
                    pred_ann = pred

                valid = pd.concat([realized.rename('realized'), pred_ann.rename('pred')], axis=1).dropna()
                if valid.shape[0] < 5:
                    return None, valid
                y = valid['realized'].values
                yhat = valid['pred'].values
                mae = float(np.mean(np.abs(y - yhat)))
                # Pearson correlation
                try:
                    pearson = float(np.corrcoef(y, yhat)[0,1])
                except Exception:
                    pearson = float('nan')
                return {"mae": mae, "pearson": pearson}, valid

            garch_ext, garch_valid = forecast_extended_stats(garch_preds)
            ml_ext, ml_valid = forecast_extended_stats(ml_preds)

            # Backtest hit rates and trade counts
            buy_hold = results.get("buy_hold")
            garch_df = results.get("garch")
            ml_df = results.get("ml")

            def hit_rate_and_trades(df):
                if df is None or "strat_ret" not in df.columns:
                    return None
                # Hit rate over active days (where active_position exists) else over all days
                if "active_position" in df.columns:
                    active_mask = df["active_position"].notna()
                    if active_mask.sum() > 0:
                        hit = float((df.loc[active_mask, "strat_ret"] > 0).mean())
                        trades = int(df["rebalance"].sum()) if "rebalance" in df.columns else int(active_mask.sum())
                        avg_pos = float(df.loc[active_mask, "active_position"].mean())
                        return {"hit_rate": hit, "trades": trades, "avg_position": avg_pos}
                # fallback: daily positive return rate
                hit = float((df["strat_ret"] > 0).mean())
                return {"hit_rate": hit, "trades": 0, "avg_position": float('nan')}

            bh_stats = hit_rate_and_trades(buy_hold)
            g_stats = hit_rate_and_trades(garch_df)
            m_stats = hit_rate_and_trades(ml_df)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Buy & Hold")
                if bh_stats is None:
                    st.info("No buy & hold results")
                else:
                    st.metric("Positive Day Rate", f"{bh_stats['hit_rate']:.2%}")
            with c2:
                st.subheader("GARCH Strategy")
                if g_stats is None:
                    st.info("GARCH strategy not available or insufficient data")
                else:
                    st.metric("Hit Rate (active days)", f"{g_stats['hit_rate']:.2%}")
                    st.metric("Rebalance Count", f"{g_stats['trades']}")
                    st.write(f"Avg position size: {g_stats['avg_position']:.3f}")
            with c3:
                st.subheader("ML Volatility Strategy")
                if m_stats is None:
                    st.info("ML volatility strategy not available or insufficient data")
                else:
                    st.metric("Hit Rate (active days)", f"{m_stats['hit_rate']:.2%}")
                    st.metric("Rebalance Count", f"{m_stats['trades']}")
                    st.write(f"Avg position size: {m_stats['avg_position']:.3f}")

            # Forecast summary table (GARCH vs ML)
            summary_rows = []
            if garch_stats is not None:
                summary_rows.append({
                    "model": "GARCH",
                    "n": garch_stats["n"],
                    "rmse": garch_stats["rmse"],
                    "r2": garch_stats["r2"],
                    "mae": garch_ext["mae"] if garch_ext is not None else float('nan'),
                    "pearson": garch_ext["pearson"] if garch_ext is not None else float('nan')
                })
            if ml_stats is not None:
                summary_rows.append({
                    "model": "ML",
                    "n": ml_stats["n"],
                    "rmse": ml_stats["rmse"],
                    "r2": ml_stats["r2"],
                    "mae": ml_ext["mae"] if ml_ext is not None else float('nan'),
                    "pearson": ml_ext["pearson"] if ml_ext is not None else float('nan')
                })

            if summary_rows:
                st.markdown("#### Forecast Performance Summary")
                sum_df = pd.DataFrame(summary_rows)
                sum_df = sum_df.set_index('model')
                # format numbers
                sum_df_display = sum_df.copy()
                sum_df_display['rmse'] = sum_df_display['rmse'].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                sum_df_display['r2'] = sum_df_display['r2'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                sum_df_display['mae'] = sum_df_display['mae'].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                sum_df_display['pearson'] = sum_df_display['pearson'].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.dataframe(sum_df_display, use_container_width=True)

            # Scatter plots: Predicted vs Realized
            if (garch_valid is not None and not garch_valid.empty) or (ml_valid is not None and not ml_valid.empty):
                sp_col1, sp_col2 = st.columns(2)
                if garch_valid is not None and not garch_valid.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=garch_valid['realized'], y=garch_valid['pred'], mode='markers', name='GARCH', marker=dict(size=6)))
                    fig.add_trace(go.Scatter(x=garch_valid['realized'], y=garch_valid['realized'], mode='lines', name='y=x', line=dict(color='red', dash='dash')))
                    fig.update_layout(title='GARCH: Predicted vs Realized Volatility', xaxis_title='Realized', yaxis_title='Predicted', height=400)
                    sp_col1.plotly_chart(fig, use_container_width=True)

                if ml_valid is not None and not ml_valid.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=ml_valid['realized'], y=ml_valid['pred'], mode='markers', name='ML', marker=dict(size=6)))
                    fig2.add_trace(go.Scatter(x=ml_valid['realized'], y=ml_valid['realized'], mode='lines', name='y=x', line=dict(color='red', dash='dash')))
                    fig2.update_layout(title='ML: Predicted vs Realized Volatility', xaxis_title='Realized', yaxis_title='Predicted', height=400)
                    sp_col2.plotly_chart(fig2, use_container_width=True)

        # Charts section
        st.subheader("Strategy Performance")

        # Filter results for plotting
        plot_results = {k: v for k, v in results.items() if isinstance(v, pd.DataFrame) and "equity" in v.columns}

        if plot_results:
            # Equity curves
            eq_fig = plot_equity_curves(plot_results, f"{strategy_name} - Equity Curves")
            st.plotly_chart(eq_fig, use_container_width=True)

            # Strategy-specific plots
            if strategy_name == "MA Crossover":
                # Price chart with signals (baseline strategy)
                if "baseline" in results:
                    st.subheader("Price Chart with Trading Signals")
                    price_fig = plot_price_with_signals(results["baseline"], "MA Crossover")
                    st.plotly_chart(price_fig, use_container_width=True)

                # ML probabilities
                if "ml_probabilities" in results:
                    st.subheader("ML Signal Probabilities")
                    prob_fig = go.Figure()
                    proba = results["ml_probabilities"].dropna()
                    prob_fig.add_trace(go.Scatter(
                        x=proba.index, y=proba,
                        mode='lines', name='ML Probability',
                        line=dict(color='blue', width=1),
                        hovertemplate="Probability: %{y:.3f}<extra></extra>"
                    ))
                    prob_fig.add_hline(y=strategy_params.get("proba_threshold", 0.55), line_dash="dash", line_color="red",
                                      annotation_text="Threshold")
                    prob_fig.update_layout(
                        title="ML Entry Signal Probabilities",
                        xaxis_title="Date",
                        yaxis_title="Probability",
                        height=300
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)

            elif strategy_name == "Volatility Forecasting":
                # Volatility forecasts
                if "forecasts" in results:
                    st.subheader("Volatility Forecasts")
                    vol_fig = plot_volatility_forecasts(results["forecasts"])
                    st.plotly_chart(vol_fig, use_container_width=True)

            # Drawdowns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Drawdowns")
                dd_fig = plot_drawdowns(plot_results)
                st.plotly_chart(dd_fig, use_container_width=True)

            with col2:
                st.subheader("Returns Distribution")

                # Create returns distribution plot
                returns_fig = go.Figure()

                for name, df in plot_results.items():
                    if "strat_ret" in df.columns:
                        returns = df["strat_ret"].dropna() * 100  # Convert to percentage
                        returns_fig.add_trace(go.Histogram(
                            x=returns,
                            name=name.replace('_', ' ').title(),
                            opacity=0.7,
                            nbinsx=50
                        ))

                returns_fig.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Returns (%)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )

                st.plotly_chart(returns_fig, use_container_width=True)

        # Export section
        st.subheader("Export Results")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Download CSV"):
                # Prepare export data
                export_data = pd.DataFrame()
                for name, df in plot_results.items():
                    export_data[f"{name}_equity"] = df["equity"]
                    if "strat_ret" in df.columns:
                        export_data[f"{name}_returns"] = df["strat_ret"]

                csv = export_data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{strategy_name.lower().replace(' ', '_')}_results.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("Copy Metrics"):
                metrics_text = metrics_df.to_string(index=False) if 'metrics_df' in locals() else ''
                st.code(metrics_text)

    else:
        # Welcome message
        st.info(
            """
        Welcome to the ML Trading Strategies Dashboard!

        This application implements two sophisticated trading strategies using machine learning:

        MA Crossover Strategy:
        - Classic moving average crossover with ML-enhanced signal filtering
        - Uses gradient boosting to predict successful entry signals
        - Compares baseline vs. ML-filtered performance

        Volatility Forecasting Strategy:
        - Risk-adjusted position sizing based on predicted volatility
        - Combines GARCH models with ML forecasting
        - Implements volatility targeting for better risk management

        Get Started:
        1. Select a strategy from the sidebar
        2. Configure parameters for your analysis
        3. Choose a ticker and date range
        4. Click "Run Backtest" to see results

        The strategies use object-oriented programming for clean, maintainable code and provide comprehensive performance analytics with interactive visualizations.
        """
        )

        # Sample results showcase
        st.subheader("Sample Analytics")

        # Create sample equity curve
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        returns1 = np.random.normal(0.0008, 0.015, len(dates))
        returns2 = np.random.normal(0.0010, 0.012, len(dates))

        equity1 = pd.Series((1 + returns1).cumprod(), index=dates)
        equity2 = pd.Series((1 + returns2).cumprod(), index=dates)

        sample_fig = go.Figure()
        sample_fig.add_trace(go.Scatter(x=dates, y=equity1, name="Baseline Strategy", line=dict(color='blue')))
        sample_fig.add_trace(go.Scatter(x=dates, y=equity2, name="ML-Enhanced Strategy", line=dict(color='green')))
        sample_fig.update_layout(
            title="Sample Strategy Comparison",
            xaxis_title="Date",
            yaxis_title="Equity (Growth of $1)",
            height=400
        )

        st.plotly_chart(sample_fig, use_container_width=True)

if __name__ == "__main__":
    main()
