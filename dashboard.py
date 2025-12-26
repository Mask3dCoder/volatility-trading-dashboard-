import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import warnings
from scipy import stats  # Moved import here (was inside function)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        self.data_end_received = {}  # To track historicalDataEnd

    def error(self, reqId, errorCode, errorString):
        print(f"Error {reqId} | Code: {errorCode} | {errorString}")

    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'close': bar.close,
            'high': bar.high,
            'low': bar.low,
            'volume': bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data received for request ID {reqId} from {start} to {end}")
        self.data_end_received[reqId] = True  # Mark as complete


class ImpliedVolatilityDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Implied Volatility Trading Dashboard")
        self.root.geometry("1400x900")  # Reduced height slightly for better fit

        self.equity_data = None        # Stores raw IV data from IB
        self.volatility_data = None    # Processed DataFrame with IV and percentile
        self.current_implied_vol = None

        self.ib_app = IBApp()
        self.connected = False

        self.vol_annualization = 252

        self.setup_ui()

    def create_equity_contract(self, symbol):
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.primaryExchange = "NASDAQ"  # Helps routing for some symbols
        return contract

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)

        # Connection Frame
        conn_frame = ttk.LabelFrame(main_frame, text="Interactive Brokers Connection", padding="5")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=15).grid(row=0, column=1, padx=(0, 10))

        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
        self.port_var = tk.StringVar(value="7497")  # 7497 = Live, 7496 = Paper
        ttk.Entry(conn_frame, textvariable=self.port_var, width=10).grid(row=0, column=3, padx=(0, 10))

        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_to_ib)
        self.connect_btn.grid(row=0, column=4, padx=(0, 10))

        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", command=self.disconnect_from_ib, state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=(0, 10))

        # Data Query Frame
        data_frame = ttk.LabelFrame(main_frame, text="Data Query", padding="5")
        data_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(data_frame, text="Symbol:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(data_frame, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=(0, 10))

        ttk.Label(data_frame, text="Duration:").grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
        self.duration_var = tk.StringVar(value="2 Y")
        ttk.Entry(data_frame, textvariable=self.duration_var, width=10).grid(row=0, column=3, padx=(0, 10))

        self.query_btn = ttk.Button(data_frame, text="Query IV Data", command=self.query_data, state="disabled")
        self.query_btn.grid(row=0, column=4, padx=(0, 10))

        self.analyze_btn = ttk.Button(data_frame, text="Analyze Implied Volatility", command=self.analyze_volatility, state="disabled")
        self.analyze_btn.grid(row=0, column=5, padx=(0, 10))

        # Current IV Frame
        vol_frame = ttk.LabelFrame(main_frame, text="Current Implied Volatility", padding="10")
        vol_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(vol_frame, text="Current IV:").grid(row=0, column=0, padx=(0, 10))
        self.current_vol_label = ttk.Label(vol_frame, text="N/A", font=("Arial", 16, "bold"))
        self.current_vol_label.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(vol_frame, text="Computation:").grid(row=0, column=2, padx=(0, 10))
        self.vol_computation_label = ttk.Label(vol_frame, text="No data", font=("Arial", 10))
        self.vol_computation_label.grid(row=0, column=3, padx=(0, 10))

        ttk.Label(vol_frame, text="Vol Range:").grid(row=0, column=4, padx=(0, 10))
        self.vol_range_label = ttk.Label(vol_frame, text="N/A", font=("Arial", 10))
        self.vol_range_label.grid(row=0, column=5)

        # Regime Analysis Frame
        regime_frame = ttk.LabelFrame(main_frame, text="Volatility Regime Analysis", padding="5")
        regime_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(regime_frame, text="Current Regime:").grid(row=0, column=0, padx=(0, 10))
        self.regime_label = ttk.Label(regime_frame, text="N/A", font=("Arial", 12, "bold"))
        self.regime_label.grid(row=0, column=1, padx=(0, 20))

        ttk.Label(regime_frame, text="Percentile:").grid(row=0, column=2, padx=(0, 10))
        self.percentile_label = ttk.Label(regime_frame, text="N/A", font=("Arial", 10))
        self.percentile_label.grid(row=0, column=3, padx=(0, 20))

        ttk.Label(regime_frame, text="Mean Reversion Signal:").grid(row=0, column=4, padx=(0, 10))
        self.mean_reversion_label = ttk.Label(regime_frame, text="N/A", font=("Arial", 10))
        self.mean_reversion_label.grid(row=0, column=5)

        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="5")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=6)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Plot Frame
        plot_frame = ttk.LabelFrame(main_frame, text="Implied Volatility Analysis Results", padding="5")
        plot_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def connect_to_ib(self):
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get())

            self.log_message(f"Attempting connection to IBKR at {host}:{port}...")

            def run_client():
                self.ib_app.connect(host, port, clientId=0)
                self.ib_app.run()

            thread = threading.Thread(target=run_client, daemon=True)
            thread.start()

            # Wait up to 5 seconds for connection
            for _ in range(50):
                if self.ib_app.isConnected():
                    self.connected = True
                    self.connect_btn.config(state="disabled")
                    self.disconnect_btn.config(state="normal")
                    self.query_btn.config(state="normal")
                    self.log_message("Successfully connected to IBKR.")
                    return
                time.sleep(0.1)

            self.log_message("Connection timeout. Check TWS/IB Gateway settings.")
        except Exception as e:
            self.log_message(f"Connection failed: {e}")

    def disconnect_from_ib(self):
        try:
            self.ib_app.disconnect()
            self.connected = False
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            self.query_btn.config(state="disabled")
            self.analyze_btn.config(state="disabled")
            self.log_message("Disconnected from IBKR.")
            self.clear_display()
        except Exception as e:
            self.log_message(f"Error during disconnect: {e}")

    def clear_display(self):
        self.current_implied_vol = None
        self.equity_data = None
        self.volatility_data = None
        self.update_current_vol_display()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas.draw()

    def query_data(self):
        if not self.connected:
            messagebox.showerror("Error", "Not connected to IBKR.")
            return

        symbol = self.symbol_var.get().strip().upper()
        duration = self.duration_var.get().strip()

        if not symbol:
            messagebox.showerror("Error", "Please enter a symbol.")
            return

        self.log_message(f"Requesting implied volatility data for {symbol} ({duration})...")

        self.ib_app.historical_data.clear()
        self.ib_app.data_end_received.clear()

        contract = self.create_equity_contract(symbol)

        self.ib_app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="OPTION_IMPLIED_VOLATILITY",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # Wait for data or timeout
        timeout = 20
        start_time = time.time()

        while time.time() - start_time < timeout:
            if 1 in self.ib_app.data_end_received:
                break
            time.sleep(0.2)

        if 1 in self.ib_app.historical_data and len(self.ib_app.historical_data[1]) > 0:
            data = self.ib_app.historical_data[1]
            self.equity_data = pd.DataFrame(data)
            self.equity_data['date'] = pd.to_datetime(self.equity_data['date'])
            self.equity_data.set_index('date', inplace=True)
            self.equity_data.sort_index(inplace=True)

            # IB returns IV already annualized; do NOT multiply again
            self.equity_data["implied_vol"] = self.equity_data["close"]

            self.log_message(f"Received {len(self.equity_data)} IV data points for {symbol}")
            self.log_message(f"Date range: {self.equity_data.index.min().date()} to {self.equity_data.index.max().date()}")

            self.process_implied_volatility()
            self.analyze_btn.config(state="normal")
        else:
            self.log_message("No implied volatility data received (symbol may not support IV or market closed).")
            self.equity_data = None
            self.analyze_btn.config(state="disabled")

    def process_implied_volatility(self):
        if self.equity_data is None or len(self.equity_data) == 0:
            return

        df = self.equity_data.copy()
        df["iv_percentile"] = df["implied_vol"].rolling(window=252, min_periods=50).rank(pct=True)

        self.current_implied_vol = df["implied_vol"].iloc[-1]
        self.volatility_data = df[['implied_vol', 'iv_percentile']].copy()

        self.update_current_vol_display()

        self.log_message(f"Current IV: {self.current_implied_vol:.4f} ({self.current_implied_vol*100:.2f}%)")

    def update_current_vol_display(self):
        if self.current_implied_vol is not None:
            self.current_vol_label.config(text=f"{self.current_implied_vol:.4f} ({self.current_implied_vol*100:.2f}%)")
            self.vol_computation_label.config(text="From IBKR OPTION_IMPLIED_VOLATILITY (annualized)")

            if self.volatility_data is not None:
                vol_min = self.volatility_data["implied_vol"].min()
                vol_max = self.volatility_data["implied_vol"].max()
                vol_mean = self.volatility_data["implied_vol"].mean()
                vol_25 = self.volatility_data["implied_vol"].quantile(0.25)
                vol_75 = self.volatility_data["implied_vol"].quantile(0.75)
                range_text = f"25%: {vol_25:.3f} | Mean: {vol_mean:.3f} | 75%: {vol_75:.3f} | Max: {vol_max:.3f}"
                self.vol_range_label.config(text=range_text)

            # Color coding
            if self.current_implied_vol > 0.40:
                color = "red"
            elif self.current_implied_vol < 0.15:
                color = "green"
            else:
                color = "black"
            self.current_vol_label.config(foreground=color)

            self.update_regime_analysis()
        else:
            self.current_vol_label.config(text="N/A", foreground="black")
            self.vol_computation_label.config(text="No data")
            self.vol_range_label.config(text="N/A")
            self.regime_label.config(text="N/A")
            self.percentile_label.config(text="N/A")
            self.mean_reversion_label.config(text="N/A")

    def update_regime_analysis(self):
        if self.volatility_data is None or self.current_implied_vol is None:
            return

        current_pct = self.volatility_data["iv_percentile"].iloc[-1]

        if current_pct > 0.8:
            regime = "HIGH VOLATILITY"
            color = "red"
        elif current_pct > 0.6:
            regime = "ELEVATED"
            color = "orange"
        elif current_pct > 0.4:
            regime = "NORMAL"
            color = "black"
        elif current_pct > 0.2:
            regime = "LOW"
            color = "blue"
        else:
            regime = "VERY LOW VOLATILITY"
            color = "green"

        self.regime_label.config(text=regime, foreground=color)
        self.percentile_label.config(text=f"{current_pct:.1%}")

        if current_pct > 0.8:
            reversion = "EXPECT MEAN REVERSION DOWN"
            rcolor = "green"
        elif current_pct < 0.2:
            reversion = "EXPECT MEAN REVERSION UP"
            rcolor = "red"
        else:
            reversion = "NEUTRAL"
            rcolor = "black"

        self.mean_reversion_label.config(text=reversion, foreground=rcolor)

    def analyze_volatility(self):
        if self.volatility_data is None or len(self.volatility_data) < 60:
            messagebox.showwarning("Insufficient Data", "Need at least 60 days of IV data for analysis.")
            return

        self.log_message("Running volatility regime and forward analysis...")

        df = self.volatility_data.copy()
        df["forward_30d_vol"] = df["implied_vol"].shift(-30)
        df["vol_diff"] = df["forward_30d_vol"] - df["implied_vol"]
        analysis_df = df.dropna()

        # Overall regression: current vs forward
        slope1, intercept1, r1, p1, std1 = stats.linregress(analysis_df["implied_vol"], analysis_df["forward_30d_vol"])

        # Find intersection with y=x
        intersection_x = intercept1 / (1 - slope1) if abs(1 - slope1) > 1e-8 else analysis_df["implied_vol"].median()

        high_vol = analysis_df["implied_vol"] > intersection_x
        low_vol = analysis_df["implied_vol"] <= intersection_x

        # Regime-specific regressions
        if high_vol.sum() > 10:
            res_high = stats.linregress(analysis_df.loc[high_vol, "implied_vol"], analysis_df.loc[high_vol, "vol_diff"])
            slope_h, int_h, r_h = res_high.slope, res_high.intercept, res_high.rvalue
        else:
            slope_h = int_h = r_h = None

        if low_vol.sum() > 10:
            res_low = stats.linregress(analysis_df.loc[low_vol, "implied_vol"], analysis_df.loc[low_vol, "vol_diff"])
            slope_l, int_l, r_l = res_low.slope, res_low.intercept, res_low.rvalue
        else:
            slope_l = int_l = r_l = None

        # Plot 1: Forward vs Current
        self.ax1.clear()
        self.ax1.scatter(analysis_df["implied_vol"], analysis_df["forward_30d_vol"], alpha=0.6, s=20)
        x_line = np.linspace(analysis_df["implied_vol"].min(), analysis_df["implied_vol"].max(), 100)
        self.ax1.plot(x_line, slope1 * x_line + intercept1, "r-", lw=2,
                      label=f"R² = {r1**2:.3f}")
        min_v, max_v = analysis_df["implied_vol"].min(), analysis_df["implied_vol"].max()
        self.ax1.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, label="y = x")
        self.ax1.set_xlabel("Current IV")
        self.ax1.set_ylabel("30-Day Forward IV")
        self.ax1.set_title(f"Forward vs Current IV\ny = {slope1:.3f}x + {intercept1:.3f}")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Plot 2: Vol change vs current level
        self.ax2.clear()
        self.ax2.scatter(analysis_df.loc[high_vol, "implied_vol"], analysis_df.loc[high_vol, "vol_diff"],
                         color="red", alpha=0.6, s=20, label="High Vol Regime")
        self.ax2.scatter(analysis_df.loc[low_vol, "implied_vol"], analysis_df.loc[low_vol, "vol_diff"],
                         color="blue", alpha=0.6, s=20, label="Low Vol Regime")

        if slope_h is not None:
            xh = analysis_df.loc[high_vol, "implied_vol"]
            xr_h = np.linspace(xh.min(), xh.max(), 100)
            self.ax2.plot(xr_h, slope_h * xr_h + int_h, "r-", lw=2, label=f"High R² = {r_h**2:.3f}")
        if slope_l is not None:
            xl = analysis_df.loc[low_vol, "implied_vol"]
            xr_l = np.linspace(xl.min(), xl.max(), 100)
            self.ax2.plot(xr_l, slope_l * xr_l + int_l, "b-", lw=2, label=f"Low R² = {r_l**2:.3f}")

        self.ax2.axhline(0, color="black", linestyle="--", alpha=0.7)
        self.ax2.axvline(intersection_x, color="green", linestyle=":", alpha=0.8,
                         label=f"Split at {intersection_x:.3f}")
        self.ax2.set_xlabel("Current Implied Volatility")
        self.ax2.set_ylabel("30-Day Forward Change")
        self.ax2.set_title("Regime-Dependent Mean Reversion")
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        # Plot 3: Time series
        self.ax3.clear()
        self.ax3.plot(df.index, df["implied_vol"], label="Implied Volatility", linewidth=2)
        q25 = df["implied_vol"].quantile(0.25)
        q75 = df["implied_vol"].quantile(0.75)
        self.ax3.axhline(q75, color="red", linestyle="--", alpha=0.7, label="75th Percentile")
        self.ax3.axhline(q25, color="green", linestyle="--", alpha=0.7, label="25th Percentile")
        self.ax3.scatter(df.index[-1], self.current_implied_vol, color="red", s=100, zorder=5, label="Latest")
        self.ax3.set_xlabel("Date")
        self.ax3.set_ylabel("Implied Volatility")
        self.ax3.set_title("IV Time Series with Percentile Bands")
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(axis='x', rotation=45)

        self.fig.tight_layout()
        self.canvas.draw()
        self.log_message("Analysis complete.")


def main():
    root = tk.Tk()
    app = ImpliedVolatilityDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()