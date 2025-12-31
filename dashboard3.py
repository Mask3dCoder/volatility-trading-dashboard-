import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm
import json
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-darkgrid')


# ==================== Black-Scholes Greeks ====================
class BlackScholesGreeks:
    """Calculate theoretical option Greeks using Black-Scholes model"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return BlackScholesGreeks.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_delta(S, K, T, r, sigma):
        """Calculate call option delta"""
        return norm.cdf(BlackScholesGreeks.d1(S, K, T, r, sigma))
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate gamma (same for calls and puts)"""
        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate vega (same for calls and puts) - returns per 1% change"""
        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def call_theta(S, K, T, r, sigma):
        """Calculate call option theta - returns per day"""
        d1 = BlackScholesGreeks.d1(S, K, T, r, sigma)
        d2 = BlackScholesGreeks.d2(S, K, T, r, sigma)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        return theta / 365  # Per day


# ==================== Interactive Brokers API ====================
class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data = {}
        self.next_req_id = 1
        self.data_end_flags = {}
        self.lock = threading.Lock()

    def error(self, reqId, errorCode, errorString):
        if errorCode == 2176 and 'fractional share' in errorString.lower():
            return
        if errorCode == 162:  # Historical data farm connection OK
            return
        print(f"Error {reqId} {errorCode} {errorString}")

    def nextValidId(self, orderId):
        self.connected = True
        self.next_req_id = orderId
        print("Connected to IBKR TWS")

    def historicalData(self, reqId, bar):
        with self.lock:
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
        with self.lock:
            self.data_end_flags[reqId] = True
        print(f"Historical data received for reqId {reqId}")

    def get_next_req_id(self):
        with self.lock:
            req_id = self.next_req_id
            self.next_req_id += 1
            return req_id


# ==================== Main Dashboard ====================
class AdvancedVolatilityDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Implied Volatility & Greeks Trading Dashboard")
        self.root.state('zoomed')  # Fullscreen on Windows
        
        # Data containers
        self.equity_data = None
        self.vix_data = None
        self.price_data = None
        self.volatility_data = None
        self.current_implied_vol = None
        self.current_vix = None
        self.current_price = None
        
        # IB Connection
        self.ib_app = IBApp()
        self.connected = False
        self.connection_thread = None
        
        # Parameters
        self.vol_annualization = 252
        self.risk_free_rate = 0.045  # 4.5% current approximation
        
        self.setup_ui()

    def create_equity_contract(self, symbol):
        """Create equity contract"""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def create_index_contract(self, symbol):
        """Create index contract (for VIX)"""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        return contract

    def setup_ui(self):
        """Setup comprehensive UI with paned layout"""
        
        # Main container with PanedWindow for resizable layout
        main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top control panel
        control_frame = ttk.Frame(main_paned)
        main_paned.add(control_frame, weight=0)
        
        # Bottom content with horizontal panes
        content_paned = ttk.PanedWindow(main_paned, orient=tk.HORIZONTAL)
        main_paned.add(content_paned, weight=1)
        
        # Left panel (charts)
        chart_frame = ttk.Frame(content_paned)
        content_paned.add(chart_frame, weight=3)
        
        # Right panel (info & controls)
        info_frame = ttk.Frame(content_paned)
        content_paned.add(info_frame, weight=1)
        
        self._setup_control_panel(control_frame)
        self._setup_chart_area(chart_frame)
        self._setup_info_panel(info_frame)

    def _setup_control_panel(self, parent):
        """Setup connection and query controls"""
        
        # Connection Frame
        conn_frame = ttk.LabelFrame(parent, text="IBKR Connection", padding=10)
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Host:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(conn_frame, text="Port:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.port_var = tk.StringVar(value="7497")
        ttk.Entry(conn_frame, textvariable=self.port_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(conn_frame, text="Client ID:").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.client_id_var = tk.StringVar(value="2")
        ttk.Entry(conn_frame, textvariable=self.client_id_var, width=6).grid(row=0, column=5, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.connect_ib)
        self.connect_btn.grid(row=0, column=4, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect", 
                                         command=self.disconnect_ib, state="disabled")
        self.disconnect_btn.grid(row=0, column=5, padx=5)
        
        # Status indicator
        self.conn_status_label = ttk.Label(conn_frame, text="● Disconnected", 
                                           foreground="red", font=("Arial", 10, "bold"))
        self.conn_status_label.grid(row=0, column=6, padx=10)
        
        # Query Frame
        query_frame = ttk.LabelFrame(parent, text="Data Query", padding=10)
        query_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(query_frame, text="Symbol:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(query_frame, textvariable=self.symbol_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(query_frame, text="Duration:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.duration_var = tk.StringVar(value="3 Y")
        ttk.Combobox(query_frame, textvariable=self.duration_var, 
                     values=["1 Y", "2 Y", "3 Y", "5 Y"], width=8).grid(row=0, column=3, padx=5)
        
        self.query_btn = ttk.Button(query_frame, text="Load All Data", 
                                    command=self.query_all_data, state='disabled')
        self.query_btn.grid(row=0, column=4, padx=10)
        
        self.analyze_btn = ttk.Button(query_frame, text="Run Full Analysis", 
                                      command=self.run_full_analysis, state="disabled")
        self.analyze_btn.grid(row=0, column=5, padx=5)
        
        self.export_btn = ttk.Button(query_frame, text="Export Data", 
                                     command=self.export_data, state="disabled")
        self.export_btn.grid(row=0, column=6, padx=5)
        
        ttk.Separator(query_frame, orient=tk.VERTICAL).grid(row=0, column=7, sticky='ns', padx=10)
        
        # Live data labels
        ttk.Label(query_frame, text="Live Price:", font=("Arial", 9, "bold")).grid(row=0, column=8, padx=5)
        self.live_price_label = ttk.Label(query_frame, text="--", 
                                          font=("Arial", 11, "bold"), foreground="blue")
        self.live_price_label.grid(row=0, column=9, padx=5)
        
        ttk.Label(query_frame, text="Live VIX:", font=("Arial", 9, "bold")).grid(row=0, column=10, padx=5)
        self.live_vix_label = ttk.Label(query_frame, text="--", 
                                        font=("Arial", 11, "bold"), foreground="purple")
        self.live_vix_label.grid(row=0, column=11, padx=5)

    def _setup_chart_area(self, parent):
        """Setup 3x3 grid of professional charts"""
        
        chart_container = ttk.LabelFrame(parent, text="Multi-Panel Volatility Analysis", padding=5)
        chart_container.pack(fill=tk.BOTH, expand=True)
        
        # Create 3x3 grid
        self.fig = plt.Figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # Create 9 subplots
        self.ax_iv_ts = self.fig.add_subplot(gs[0, 0])           # IV time series
        self.ax_iv_vix = self.fig.add_subplot(gs[0, 1])          # IV vs VIX scatter
        self.ax_iv_vix_premium = self.fig.add_subplot(gs[0, 2])  # IV-VIX premium
        
        self.ax_correlation = self.fig.add_subplot(gs[1, 0])     # Rolling correlation
        self.ax_percentile = self.fig.add_subplot(gs[1, 1])      # IV percentile
        self.ax_regime = self.fig.add_subplot(gs[1, 2])          # Regime analysis
        
        self.ax_delta = self.fig.add_subplot(gs[2, 0])           # Delta surface
        self.ax_gamma = self.fig.add_subplot(gs[2, 1])           # Gamma profile
        self.ax_vega_theta = self.fig.add_subplot(gs[2, 2])      # Vega & Theta
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._initialize_empty_charts()

    def _initialize_empty_charts(self):
        """Initialize all charts with placeholder text"""
        axes = [self.ax_iv_ts, self.ax_iv_vix, self.ax_iv_vix_premium,
                self.ax_correlation, self.ax_percentile, self.ax_regime,
                self.ax_delta, self.ax_gamma, self.ax_vega_theta]
        
        titles = ["IV Time Series", "IV vs VIX", "IV-VIX Premium",
                 "Rolling Correlation", "IV Percentile", "Regime Analysis",
                 "ATM Delta Evolution", "Gamma Profile", "Vega & Theta"]
        
        for ax, title in zip(axes, titles):
            ax.text(0.5, 0.5, f'{title}\n(Load data to populate)', 
                   ha='center', va='center', fontsize=12, alpha=0.5)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.canvas.draw_idle()

    def _setup_info_panel(self, parent):
        """Setup info and metrics panel"""
        
        # Current Metrics
        metrics_frame = ttk.LabelFrame(parent, text="Current Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        metrics = [
            ("Implied Vol:", "iv_label"),
            ("VIX Level:", "vix_label"),
            ("IV Percentile:", "percentile_label"),
            ("IV-VIX Spread:", "spread_label"),
            ("IV/VIX Ratio:", "ratio_label"),
            ("252d Correlation:", "corr_label")
        ]
        
        for idx, (label_text, attr) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label_text, font=("Arial", 9)).grid(
                row=idx, column=0, sticky=tk.W, pady=3, padx=5)
            setattr(self, attr, ttk.Label(metrics_frame, text="--", 
                                          font=("Arial", 10, "bold")))
            getattr(self, attr).grid(row=idx, column=1, sticky=tk.W, pady=3, padx=5)
        
        # Regime Analysis
        regime_frame = ttk.LabelFrame(parent, text="Regime & Strategy", padding=10)
        regime_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(regime_frame, text="Current Regime:", 
                 font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, pady=3)
        self.regime_label = ttk.Label(regime_frame, text="--", 
                                      font=("Arial", 11, "bold"))
        self.regime_label.grid(row=0, column=1, sticky=tk.W, pady=3)
        
        ttk.Label(regime_frame, text="Mean Reversion:", 
                 font=("Arial", 9)).grid(row=1, column=0, sticky=tk.W, pady=3)
        self.reversion_label = ttk.Label(regime_frame, text="--", 
                                         font=("Arial", 9))
        self.reversion_label.grid(row=1, column=1, sticky=tk.W, pady=3, columnspan=2)
        
        ttk.Label(regime_frame, text="Strategy:", 
                 font=("Arial", 9, "bold")).grid(row=2, column=0, sticky=tk.NW, pady=3)
        self.strategy_text = tk.Text(regime_frame, height=4, width=35, 
                                     wrap=tk.WORD, font=("Arial", 9))
        self.strategy_text.grid(row=2, column=1, columnspan=2, pady=3, padx=5)
        
        # Greeks Summary
        greeks_frame = ttk.LabelFrame(parent, text="ATM Option Greeks (30d)", padding=10)
        greeks_frame.pack(fill=tk.X, padx=5, pady=5)
        
        greeks = [
            ("Delta:", "delta_label"),
            ("Gamma:", "gamma_label"),
            ("Vega:", "vega_label"),
            ("Theta:", "theta_label")
        ]
        
        for idx, (label_text, attr) in enumerate(greeks):
            ttk.Label(greeks_frame, text=label_text, font=("Arial", 9)).grid(
                row=idx, column=0, sticky=tk.W, pady=3, padx=5)
            setattr(self, attr, ttk.Label(greeks_frame, text="--", 
                                          font=("Arial", 10, "bold")))
            getattr(self, attr).grid(row=idx, column=1, sticky=tk.W, pady=3, padx=5)
        
        # Summary Statistics
        summary_frame = ttk.LabelFrame(parent, text="Summary Statistics", padding=10)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, 
                                                      height=15, 
                                                      font=("Courier", 9))
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Status Log
        status_frame = ttk.LabelFrame(parent, text="Status Log", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, 
                                                     height=10, 
                                                     font=("Courier", 8))
        self.status_text.pack(fill=tk.BOTH, expand=True)

    # ==================== Connection Methods ====================
    
    def log_message(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def connect_ib(self):
        """Connect to Interactive Brokers"""
        try:
            host = self.host_var.get()
            port = int(self.port_var.get())
            client_id = int(self.client_id_var.get())
            
            self.log_message(f"Connecting to IBKR at {host}:{port} (Client {client_id})...")
            
            def connect_thread():
                try:
                    self.ib_app.connect(host, port, client_id)
                    self.ib_app.run()
                except Exception as e:
                    self.log_message(f"Connection error: {e}")
            
            self.connection_thread = threading.Thread(target=connect_thread, daemon=True)
            self.connection_thread.start()
            
            # Wait for connection
            for _ in range(50):
                if self.ib_app.connected:
                    break
                time.sleep(0.1)
            
            if self.ib_app.connected:
                self.connected = True
                self.connect_btn.config(state='disabled')
                self.disconnect_btn.config(state='normal')
                self.query_btn.config(state='normal')
                self.conn_status_label.config(text="● Connected", foreground="green")
                self.log_message("Successfully connected to IBKR TWS/Gateway")
            else:
                self.log_message("Connection timeout - ensure TWS/Gateway is running")
                
        except Exception as e:
            self.log_message(f"Connection error: {e}")
            messagebox.showerror("Connection Error", str(e))

    def disconnect_ib(self):
        """Disconnect from Interactive Brokers"""
        try:
            self.ib_app.disconnect()
            self.connected = False
            self.connect_btn.config(state='normal')
            self.disconnect_btn.config(state='disabled')
            self.query_btn.config(state='disabled')
            self.analyze_btn.config(state='disabled')
            self.export_btn.config(state='disabled')
            self.conn_status_label.config(text="● Disconnected", foreground="red")
            self.log_message("Disconnected from IBKR")
        except Exception as e:
            self.log_message(f"Disconnect error: {e}")

    # ==================== Data Query Methods ====================
    
    def request_historical_data(self, contract, what_to_show, duration="3 Y"):
        """Generic method to request historical data"""
        req_id = self.ib_app.get_next_req_id()
        self.ib_app.data_end_flags[req_id] = False
        
        self.ib_app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow=what_to_show,
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        
        # Wait for data
        timeout = 30
        start_time = time.time()
        while not self.ib_app.data_end_flags.get(req_id, False):
            if time.time() - start_time > timeout:
                self.log_message(f"Timeout waiting for {what_to_show} data")
                return None
            time.sleep(0.1)
        
        # Get data
        with self.ib_app.lock:
            data = self.ib_app.historical_data.get(req_id, [])
        
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        return None

    def query_all_data(self):
        """Query all required data: IV, VIX, and underlying prices"""
        if not self.connected:
            messagebox.showerror("Error", "Not connected to IBKR")
            return
        
        symbol = self.symbol_var.get().upper()
        duration = self.duration_var.get()
        
        self.log_message(f"Querying comprehensive data for {symbol}...")
        self.log_message("This may take 30-60 seconds...")
        
        # Clear previous data
        self.ib_app.historical_data.clear()
        self.ib_app.data_end_flags.clear()
        
        # Query IV data
        self.log_message(f"1/3: Requesting Implied Volatility data...")
        contract = self.create_equity_contract(symbol)
        self.equity_data = self.request_historical_data(
            contract, "OPTION_IMPLIED_VOLATILITY", duration)
        
        if self.equity_data is not None and len(self.equity_data) > 0:
            self.log_message(f"   ✓ Received {len(self.equity_data)} IV data points")
        else:
            self.log_message("   ✗ Failed to get IV data")
            messagebox.showwarning("Data Error", 
                                  "Could not retrieve IV data. Ensure options exist for this symbol.")
            return
        
        # Query price data
        self.log_message(f"2/3: Requesting price data...")
        self.price_data = self.request_historical_data(contract, "TRADES", duration)
        
        if self.price_data is not None and len(self.price_data) > 0:
            self.log_message(f"   ✓ Received {len(self.price_data)} price data points")
            self.current_price = self.price_data['close'].iloc[-1]
            self.live_price_label.config(text=f"${self.current_price:.2f}")
        else:
            self.log_message("   ✗ Failed to get price data")
        
        # Query VIX data
        self.log_message(f"3/3: Requesting VIX data...")
        vix_contract = self.create_index_contract("VIX")
        self.vix_data = self.request_historical_data(vix_contract, "TRADES", duration)
        
        if self.vix_data is not None and len(self.vix_data) > 0:
            self.log_message(f"   ✓ Received {len(self.vix_data)} VIX data points")
            self.current_vix = self.vix_data['close'].iloc[-1]
            self.live_vix_label.config(text=f"{self.current_vix:.2f}")
        else:
            self.log_message("   ⚠ VIX data not available (will proceed without VIX analysis)")
        
        # Process all data
        self.process_all_data()
        
        self.analyze_btn.config(state="normal")
        self.export_btn.config(state="normal")
        
        self.log_message("✓ All data loaded successfully!")

    def process_all_data(self):
        """Process and align all data"""
        if self.equity_data is None:
            return
        
        self.log_message("Processing data...")
        
        # Process IV data
        self.equity_data['implied_vol'] = self.equity_data['close']
        
        # Align indices if we have VIX data
        if self.vix_data is not None:
            common_dates = self.equity_data.index.intersection(self.vix_data.index)
            self.equity_data = self.equity_data.loc[common_dates]
            self.vix_data = self.vix_data.loc[common_dates]
            self.vix_data['vix'] = self.vix_data['close']
            
            # Calculate IV-VIX metrics
            self.equity_data['vix'] = self.vix_data['vix']
            self.equity_data['iv_vix_spread'] = self.equity_data['implied_vol'] - (self.equity_data['vix'] / 100)
            self.equity_data['iv_vix_ratio'] = self.equity_data['implied_vol'] / (self.equity_data['vix'] / 100)
            
            # Rolling correlation
            window = 252
            self.equity_data['iv_vix_correlation'] = (
                self.equity_data['implied_vol']
                .rolling(window)
                .corr(self.equity_data['vix'] / 100)
            )
        
        # Calculate IV percentile
        self.equity_data['iv_percentile'] = (
            self.equity_data['implied_vol']
            .rolling(window=252, min_periods=60)
            .rank(pct=True)
        )
        
        # Realized volatility if we have prices
        if self.price_data is not None:
            common_dates = self.equity_data.index.intersection(self.price_data.index)
            self.equity_data = self.equity_data.loc[common_dates]
            self.price_data = self.price_data.loc[common_dates]
            
            self.price_data['log_return'] = np.log(self.price_data['close'] / self.price_data['close'].shift(1))
            self.price_data['realized_vol_30'] = (
                self.price_data['log_return']
                .rolling(30)
                .std() * np.sqrt(self.vol_annualization)
            )
            
            self.equity_data['realized_vol'] = self.price_data['realized_vol_30']
            self.equity_data['iv_rv_spread'] = self.equity_data['implied_vol'] - self.equity_data['realized_vol']
        
        self.volatility_data = self.equity_data.copy()
        self.current_implied_vol = self.equity_data['implied_vol'].iloc[-1]
        
        self.update_metrics_display()
        self.log_message("✓ Data processing complete")

    # ==================== Analysis Methods ====================
    
    def run_full_analysis(self):
        """Run comprehensive volatility analysis"""
        if self.volatility_data is None:
            messagebox.showerror("Error", "No data available")
            return
        
        self.log_message("Running full analysis...")
        
        # Clear all axes
        for ax in [self.ax_iv_ts, self.ax_iv_vix, self.ax_iv_vix_premium,
                   self.ax_correlation, self.ax_percentile, self.ax_regime,
                   self.ax_delta, self.ax_gamma, self.ax_vega_theta]:
            ax.clear()
        
        # Generate all charts
        self.plot_iv_timeseries()
        self.plot_iv_vix_scatter()
        self.plot_iv_vix_premium()
        self.plot_rolling_correlation()
        self.plot_iv_percentile()
        self.plot_regime_analysis()
        self.plot_greeks_delta()
        self.plot_greeks_gamma()
        self.plot_greeks_vega_theta()
        
        self.canvas.draw_idle()
        
        # Update summary
        self.update_summary_statistics()
        
        self.log_message("✓ Full analysis complete!")

    def plot_iv_timeseries(self):
        """Chart 1: IV Time Series with bands"""
        df = self.volatility_data
        
        self.ax_iv_ts.plot(df.index, df['implied_vol'], 
                          label='Implied Vol', linewidth=1.5, color='blue')
        
        if 'realized_vol' in df.columns:
            self.ax_iv_ts.plot(df.index, df['realized_vol'], 
                              label='Realized Vol (30d)', linewidth=1.5, 
                              color='orange', alpha=0.7)
        
        # Percentile bands
        q25 = df['implied_vol'].quantile(0.25)
        q75 = df['implied_vol'].quantile(0.75)
        mean = df['implied_vol'].mean()
        
        self.ax_iv_ts.axhline(q75, color='red', linestyle='--', alpha=0.5, label='75th %ile')
        self.ax_iv_ts.axhline(mean, color='black', linestyle='--', alpha=0.5, label='Mean')
        self.ax_iv_ts.axhline(q25, color='green', linestyle='--', alpha=0.5, label='25th %ile')
        
        # Current point
        if self.current_implied_vol:
            self.ax_iv_ts.scatter(df.index[-1], self.current_implied_vol, 
                                 color='red', s=100, zorder=5, label='Current')
        
        self.ax_iv_ts.set_title("Implied Volatility Time Series", fontweight='bold')
        self.ax_iv_ts.set_xlabel("Date")
        self.ax_iv_ts.set_ylabel("Volatility")
        self.ax_iv_ts.legend(fontsize=8)
        self.ax_iv_ts.grid(alpha=0.3)
        self.ax_iv_ts.tick_params(axis='x', rotation=45)

    def plot_iv_vix_scatter(self):
        """Chart 2: IV vs VIX Scatter with regression"""
        if 'vix' not in self.volatility_data.columns:
            self.ax_iv_vix.text(0.5, 0.5, 'VIX data not available', 
                               ha='center', va='center')
            self.ax_iv_vix.set_title("IV vs VIX Scatter")
            return
        
        df = self.volatility_data.dropna(subset=['implied_vol', 'vix'])
        
        # Scatter plot
        self.ax_iv_vix.scatter(df['vix'], df['implied_vol'] * 100, 
                              alpha=0.5, s=20)
        
        # Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['vix'], df['implied_vol'] * 100)
        
        x_line = np.linspace(df['vix'].min(), df['vix'].max(), 100)
        y_line = slope * x_line + intercept
        
        self.ax_iv_vix.plot(x_line, y_line, 'r-', linewidth=2, 
                           label=f'R² = {r_value**2:.3f}')
        
        # Reference line (y=x)
        max_val = max(df['vix'].max(), (df['implied_vol'] * 100).max())
        min_val = min(df['vix'].min(), (df['implied_vol'] * 100).min())
        self.ax_iv_vix.plot([min_val, max_val], [min_val, max_val], 
                           'k--', alpha=0.5, label='y=x')
        
        # Current point
        if self.current_vix and self.current_implied_vol:
            self.ax_iv_vix.scatter(self.current_vix, self.current_implied_vol * 100,
                                  color='red', s=150, marker='*', zorder=5,
                                  label='Current')
        
        self.ax_iv_vix.set_title("IV vs VIX Relationship", fontweight='bold')
        self.ax_iv_vix.set_xlabel("VIX Level")
        self.ax_iv_vix.set_ylabel("Implied Vol (%)")
        self.ax_iv_vix.legend(fontsize=8)
        self.ax_iv_vix.grid(alpha=0.3)

    def plot_iv_vix_premium(self):
        """Chart 3: IV-VIX Premium (Relative Richness)"""
        if 'iv_vix_spread' not in self.volatility_data.columns:
            self.ax_iv_vix_premium.text(0.5, 0.5, 'VIX data not available', 
                                       ha='center', va='center')
            self.ax_iv_vix_premium.set_title("IV-VIX Premium")
            return
        
        df = self.volatility_data
        spread = df['iv_vix_spread'] * 100  # Convert to percentage points
        
        self.ax_iv_vix_premium.plot(df.index, spread, 
                                   linewidth=1.5, color='purple')
        self.ax_iv_vix_premium.axhline(0, color='black', linestyle='-', 
                                      linewidth=1, alpha=0.5)
        self.ax_iv_vix_premium.axhline(spread.mean(), color='blue', 
                                      linestyle='--', alpha=0.5, label='Mean')
        
        # Fill areas
        self.ax_iv_vix_premium.fill_between(df.index, 0, spread, 
                                           where=(spread > 0), 
                                           alpha=0.3, color='green', 
                                           label='IV Rich vs VIX')
        self.ax_iv_vix_premium.fill_between(df.index, 0, spread, 
                                           where=(spread < 0), 
                                           alpha=0.3, color='red', 
                                           label='IV Cheap vs VIX')
        
        self.ax_iv_vix_premium.set_title("IV-VIX Premium (Relative Value)", 
                                        fontweight='bold')
        self.ax_iv_vix_premium.set_xlabel("Date")
        self.ax_iv_vix_premium.set_ylabel("IV - VIX (pp)")
        self.ax_iv_vix_premium.legend(fontsize=8)
        self.ax_iv_vix_premium.grid(alpha=0.3)
        self.ax_iv_vix_premium.tick_params(axis='x', rotation=45)

    def plot_rolling_correlation(self):
        """Chart 4: Rolling 252-day IV-VIX Correlation"""
        if 'iv_vix_correlation' not in self.volatility_data.columns:
            self.ax_correlation.text(0.5, 0.5, 'VIX data not available', 
                                    ha='center', va='center')
            self.ax_correlation.set_title("Rolling IV-VIX Correlation")
            return
        
        df = self.volatility_data
        corr = df['iv_vix_correlation'].dropna()
        
        self.ax_correlation.plot(corr.index, corr, linewidth=1.5, color='darkblue')
        self.ax_correlation.axhline(0, color='black', linestyle='-', linewidth=1)
        self.ax_correlation.axhline(corr.mean(), color='red', linestyle='--', 
                                   alpha=0.5, label=f'Mean: {corr.mean():.2f}')
        
        # Shade high/low correlation regions
        self.ax_correlation.fill_between(corr.index, 0, corr, 
                                        where=(corr > 0.5), 
                                        alpha=0.2, color='blue', 
                                        label='High Correlation')
        self.ax_correlation.fill_between(corr.index, 0, corr, 
                                        where=(corr < 0.3), 
                                        alpha=0.2, color='orange', 
                                        label='Low Correlation')
        
        self.ax_correlation.set_title("Rolling 252-Day IV-VIX Correlation", 
                                     fontweight='bold')
        self.ax_correlation.set_xlabel("Date")
        self.ax_correlation.set_ylabel("Correlation")
        self.ax_correlation.set_ylim(-0.2, 1.0)
        self.ax_correlation.legend(fontsize=8)
        self.ax_correlation.grid(alpha=0.3)
        self.ax_correlation.tick_params(axis='x', rotation=45)

    def plot_iv_percentile(self):
        """Chart 5: IV Percentile with regime zones"""
        df = self.volatility_data
        percentile = df['iv_percentile'].dropna()
        
        self.ax_percentile.plot(percentile.index, percentile * 100, 
                               linewidth=1.5, color='darkgreen')
        
        # Regime zones
        self.ax_percentile.axhspan(80, 100, alpha=0.2, color='red', label='High Vol')
        self.ax_percentile.axhspan(60, 80, alpha=0.2, color='orange')
        self.ax_percentile.axhspan(40, 60, alpha=0.2, color='gray', label='Normal')
        self.ax_percentile.axhspan(20, 40, alpha=0.2, color='lightblue')
        self.ax_percentile.axhspan(0, 20, alpha=0.2, color='green', label='Low Vol')
        
        # Current point
        if len(percentile) > 0:
            current = percentile.iloc[-1] * 100
            self.ax_percentile.scatter(percentile.index[-1], current,
                                      color='red', s=150, marker='*', zorder=5)
            self.ax_percentile.text(percentile.index[-1], current + 5,
                                   f'{current:.1f}%', ha='center', fontweight='bold')
        
        self.ax_percentile.set_title("IV Percentile (252d Rolling)", fontweight='bold')
        self.ax_percentile.set_xlabel("Date")
        self.ax_percentile.set_ylabel("Percentile (%)")
        self.ax_percentile.set_ylim(0, 100)
        self.ax_percentile.legend(fontsize=8, loc='upper left')
        self.ax_percentile.grid(alpha=0.3)
        self.ax_percentile.tick_params(axis='x', rotation=45)

    def plot_regime_analysis(self):
        """Chart 6: Regime-based return analysis"""
        df = self.volatility_data
        
        if 'iv_percentile' not in df.columns:
            return
        
        # Create regime bins
        df_clean = df.dropna(subset=['iv_percentile'])
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['Very Low', 'Low', 'Normal', 'Elevated', 'High']
        df_clean['regime'] = pd.cut(df_clean['iv_percentile'], bins=bins, labels=labels)
        
        # Calculate forward returns if we have price data
        if self.price_data is not None and 'close' in self.price_data.columns:
            # Align with price data
            common_idx = df_clean.index.intersection(self.price_data.index)
            df_clean = df_clean.loc[common_idx]
            prices = self.price_data.loc[common_idx]
            
            # 30-day forward return
            df_clean['forward_return'] = (
                np.log(prices['close'].shift(-30) / prices['close']) * 100
            )
            
            regime_stats = df_clean.groupby('regime')['forward_return'].agg(['mean', 'std', 'count'])
            
            # Bar plot
            x = range(len(regime_stats))
            colors = ['green', 'lightgreen', 'gray', 'orange', 'red']
            bars = self.ax_regime.bar(x, regime_stats['mean'], color=colors, alpha=0.7)
            
            # Error bars
            self.ax_regime.errorbar(x, regime_stats['mean'], 
                                   yerr=regime_stats['std'], 
                                   fmt='none', color='black', capsize=5)
            
            self.ax_regime.set_xticks(x)
            self.ax_regime.set_xticklabels(regime_stats.index, rotation=45)
            self.ax_regime.axhline(0, color='black', linestyle='-', linewidth=1)
            
            self.ax_regime.set_title("30d Forward Returns by IV Regime", fontweight='bold')
            self.ax_regime.set_ylabel("Mean Return (%)")
            self.ax_regime.grid(alpha=0.3, axis='y')
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, regime_stats['count'])):
                height = bar.get_height()
                self.ax_regime.text(bar.get_x() + bar.get_width()/2., height,
                                   f'n={int(count)}', ha='center', va='bottom', fontsize=8)
        else:
            # Just show regime distribution
            regime_counts = df_clean['regime'].value_counts().sort_index()
            colors = ['green', 'lightgreen', 'gray', 'orange', 'red']
            regime_counts.plot(kind='bar', ax=self.ax_regime, color=colors, alpha=0.7)
            self.ax_regime.set_title("IV Regime Distribution", fontweight='bold')
            self.ax_regime.set_ylabel("Count")
            self.ax_regime.set_xlabel("")
            self.ax_regime.tick_params(axis='x', rotation=45)

    def plot_greeks_delta(self):
        """Chart 7: ATM Delta evolution over time to expiry"""
        if self.current_price is None or self.current_implied_vol is None:
            self.ax_delta.text(0.5, 0.5, 'Price/IV data required', 
                              ha='center', va='center')
            self.ax_delta.set_title("ATM Delta Evolution")
            return
        
        S = self.current_price
        K = S  # ATM
        r = self.risk_free_rate
        sigma = self.current_implied_vol
        
        # Time to expiry from 90 days to 1 day
        days = np.linspace(90, 1, 90)
        T = days / 365
        
        deltas = [BlackScholesGreeks.call_delta(S, K, t, r, sigma) for t in T]
        
        self.ax_delta.plot(days, deltas, linewidth=2, color='blue')
        self.ax_delta.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Delta = 0.5')
        self.ax_delta.fill_between(days, 0.5, deltas, alpha=0.3, color='lightblue')
        
        # Mark key tenors
        for tenor_days in [30, 60, 90]:
            idx = np.argmin(np.abs(days - tenor_days))
            self.ax_delta.scatter(days[idx], deltas[idx], s=100, color='red', zorder=5)
            self.ax_delta.text(days[idx], deltas[idx] + 0.02, f'{tenor_days}d', 
                              ha='center', fontsize=8)
        
        self.ax_delta.set_title(f"ATM Call Delta Evolution (IV={sigma*100:.1f}%)", 
                               fontweight='bold')
        self.ax_delta.set_xlabel("Days to Expiry")
        self.ax_delta.set_ylabel("Delta")
        self.ax_delta.legend(fontsize=8)
        self.ax_delta.grid(alpha=0.3)
        self.ax_delta.set_ylim(0.4, 0.6)

    def plot_greeks_gamma(self):
        """Chart 8: Gamma profile across strikes"""
        if self.current_price is None or self.current_implied_vol is None:
            self.ax_gamma.text(0.5, 0.5, 'Price/IV data required', 
                              ha='center', va='center')
            self.ax_gamma.set_title("Gamma Profile")
            return
        
        S = self.current_price
        r = self.risk_free_rate
        sigma = self.current_implied_vol
        T = 30 / 365  # 30 days
        
        # Strike range around ATM
        strike_range = np.linspace(S * 0.9, S * 1.1, 100)
        gammas = [BlackScholesGreeks.gamma(S, K, T, r, sigma) for K in strike_range]
        
        self.ax_gamma.plot(strike_range, gammas, linewidth=2, color='purple')
        self.ax_gamma.axvline(S, color='red', linestyle='--', alpha=0.5, label='Current Price')
        
        # Fill under curve
        self.ax_gamma.fill_between(strike_range, 0, gammas, alpha=0.3, color='purple')
        
        # Mark ATM
        gamma_atm = BlackScholesGreeks.gamma(S, S, T, r, sigma)
        self.ax_gamma.scatter(S, gamma_atm, s=150, color='red', zorder=5, marker='*')
        self.ax_gamma.text(S, gamma_atm * 1.1, f'ATM\nΓ={gamma_atm:.4f}', 
                          ha='center', fontsize=8)
        
        self.ax_gamma.set_title(f"Gamma Profile (30d, IV={sigma*100:.1f}%)", 
                               fontweight='bold')
        self.ax_gamma.set_xlabel("Strike Price")
        self.ax_gamma.set_ylabel("Gamma")
        self.ax_gamma.legend(fontsize=8)
        self.ax_gamma.grid(alpha=0.3)

    def plot_greeks_vega_theta(self):
        """Chart 9: Vega and Theta across time"""
        if self.current_price is None or self.current_implied_vol is None:
            self.ax_vega_theta.text(0.5, 0.5, 'Price/IV data required', 
                                   ha='center', va='center')
            self.ax_vega_theta.set_title("Vega & Theta")
            return
        
        S = self.current_price
        K = S  # ATM
        r = self.risk_free_rate
        sigma = self.current_implied_vol
        
        days = np.linspace(90, 1, 90)
        T = days / 365
        
        vegas = [BlackScholesGreeks.vega(S, K, t, r, sigma) for t in T]
        thetas = [BlackScholesGreeks.call_theta(S, K, t, r, sigma) for t in T]
        
        # Twin axes
        ax2 = self.ax_vega_theta.twinx()
        
        line1 = self.ax_vega_theta.plot(days, vegas, linewidth=2, color='green', 
                                        label='Vega')
        line2 = ax2.plot(days, thetas, linewidth=2, color='red', label='Theta')
        
        self.ax_vega_theta.set_xlabel("Days to Expiry")
        self.ax_vega_theta.set_ylabel("Vega (per 1% IV)", color='green')
        ax2.set_ylabel("Theta (per day)", color='red')
        
        self.ax_vega_theta.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax_vega_theta.legend(lines, labels, fontsize=8)
        
        self.ax_vega_theta.set_title(f"ATM Vega & Theta (IV={sigma*100:.1f}%)", 
                                     fontweight='bold')
        self.ax_vega_theta.grid(alpha=0.3)

    # ==================== Display Update Methods ====================
    
    def update_metrics_display(self):
        """Update all metric labels"""
        if self.current_implied_vol:
            self.iv_label.config(text=f"{self.current_implied_vol*100:.2f}%")
        
        if self.current_vix:
            self.vix_label.config(text=f"{self.current_vix:.2f}")
        
        if self.volatility_data is not None:
            # Percentile
            pct = self.volatility_data['iv_percentile'].iloc[-1]
            if pd.notna(pct):
                self.percentile_label.config(text=f"{pct*100:.1f}%")
            
            # Spread and ratio (if VIX available)
            if 'iv_vix_spread' in self.volatility_data.columns:
                spread = self.volatility_data['iv_vix_spread'].iloc[-1]
                ratio = self.volatility_data['iv_vix_ratio'].iloc[-1]
                
                if pd.notna(spread):
                    color = 'green' if spread > 0 else 'red'
                    self.spread_label.config(text=f"{spread*100:.2f}pp", foreground=color)
                
                if pd.notna(ratio):
                    self.ratio_label.config(text=f"{ratio:.2f}")
            
            # Correlation
            if 'iv_vix_correlation' in self.volatility_data.columns:
                corr = self.volatility_data['iv_vix_correlation'].iloc[-1]
                if pd.notna(corr):
                    self.corr_label.config(text=f"{corr:.3f}")
        
        self.update_regime_display()
        self.update_greeks_display()

    def update_regime_display(self):
        """Update regime analysis display"""
        if self.volatility_data is None:
            return
        
        pct = self.volatility_data['iv_percentile'].iloc[-1]
        if pd.isna(pct):
            return
        
        # Determine regime
        if pct > 0.8:
            regime = "EXTREME HIGH"
            color = "red"
            reversion = "Strong Mean Reversion Expected Down"
            strategy = ("• Short volatility strategies\n"
                       "• Sell straddles/strangles\n"
                       "• Credit spreads\n"
                       "• Iron condors\n"
                       "• Calendar spreads (sell near)")
        elif pct > 0.6:
            regime = "ELEVATED"
            color = "orange"
            reversion = "Modest Mean Reversion Expected Down"
            strategy = ("• Cautious short vol\n"
                       "• Credit spreads with defined risk\n"
                       "• Ratio spreads\n"
                       "• Consider hedged positions")
        elif pct > 0.4:
            regime = "NORMAL"
            color = "black"
            reversion = "Neutral / Range-Bound"
            strategy = ("• Directional trades\n"
                       "• Balanced strategies\n"
                       "• Delta-neutral positions\n"
                       "• Standard spreads")
        elif pct > 0.2:
            regime = "DEPRESSED"
            color = "blue"
            reversion = "Modest Mean Reversion Expected Up"
            strategy = ("• Cautious long vol\n"
                       "• Debit spreads\n"
                       "• Calendar spreads (buy far)\n"
                       "• Protective positions")
        else:
            regime = "EXTREME LOW"
            color = "green"
            reversion = "Strong Mean Reversion Expected Up"
            strategy = ("• Long volatility strategies\n"
                       "• Buy straddles/strangles\n"
                       "• Debit spreads\n"
                       "• Volatility expansion plays\n"
                       "• Protective puts cheap")
        
        # Additional context from VIX if available
        if 'iv_vix_spread' in self.volatility_data.columns:
            spread = self.volatility_data['iv_vix_spread'].iloc[-1]
            if pd.notna(spread):
                if spread > 0.05:
                    strategy += "\n• IV rich vs VIX - favor selling"
                elif spread < -0.05:
                    strategy += "\n• IV cheap vs VIX - favor buying"
        
        self.regime_label.config(text=regime, foreground=color)
        self.reversion_label.config(text=reversion)
        self.strategy_text.delete(1.0, tk.END)
        self.strategy_text.insert(1.0, strategy)

    def update_greeks_display(self):
        """Update Greeks labels for 30-day ATM option"""
        if self.current_price is None or self.current_implied_vol is None:
            return
        
        S = self.current_price
        K = S  # ATM
        T = 30 / 365
        r = self.risk_free_rate
        sigma = self.current_implied_vol
        
        delta = BlackScholesGreeks.call_delta(S, K, T, r, sigma)
        gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
        vega = BlackScholesGreeks.vega(S, K, T, r, sigma)
        theta = BlackScholesGreeks.call_theta(S, K, T, r, sigma)
        
        self.delta_label.config(text=f"{delta:.3f}")
        self.gamma_label.config(text=f"{gamma:.4f}")
        self.vega_label.config(text=f"{vega:.3f}")
        self.theta_label.config(text=f"{theta:.3f}")

    def update_summary_statistics(self):
        """Update comprehensive summary statistics"""
        if self.volatility_data is None:
            return
        
        df = self.volatility_data
        
        summary = "=" * 50 + "\n"
        summary += "COMPREHENSIVE VOLATILITY ANALYSIS SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        # Current metrics
        summary += "CURRENT METRICS:\n"
        summary += f"  Implied Vol:     {self.current_implied_vol*100:.2f}%\n"
        if self.current_vix:
            summary += f"  VIX Level:       {self.current_vix:.2f}\n"
        if self.current_price:
            summary += f"  Underlying:      ${self.current_price:.2f}\n"
        summary += "\n"
        
        # Historical statistics
        summary += "HISTORICAL STATISTICS:\n"
        iv_stats = df['implied_vol'].describe()
        summary += f"  IV Mean:         {iv_stats['mean']*100:.2f}%\n"
        summary += f"  IV Std Dev:      {iv_stats['std']*100:.2f}%\n"
        summary += f"  IV Min:          {iv_stats['min']*100:.2f}%\n"
        summary += f"  IV 25th %ile:    {iv_stats['25%']*100:.2f}%\n"
        summary += f"  IV Median:       {iv_stats['50%']*100:.2f}%\n"
        summary += f"  IV 75th %ile:    {iv_stats['75%']*100:.2f}%\n"
        summary += f"  IV Max:          {iv_stats['max']*100:.2f}%\n"
        summary += "\n"
        
        # VIX comparison
        if 'vix' in df.columns:
            summary += "VIX RELATIONSHIP:\n"
            corr = df['iv_vix_correlation'].iloc[-1]
            spread = df['iv_vix_spread'].iloc[-1]
            ratio = df['iv_vix_ratio'].iloc[-1]
            
            summary += f"  Current Correlation: {corr:.3f}\n"
            summary += f"  IV-VIX Spread:       {spread*100:.2f}pp\n"
            summary += f"  IV/VIX Ratio:        {ratio:.3f}\n"
            
            avg_spread = df['iv_vix_spread'].mean()
            summary += f"  Avg Spread:          {avg_spread*100:.2f}pp\n"
            
            if spread > avg_spread + 0.05:
                summary += "  → IV RICH relative to VIX\n"
            elif spread < avg_spread - 0.05:
                summary += "  → IV CHEAP relative to VIX\n"
            else:
                summary += "  → IV FAIR relative to VIX\n"
            summary += "\n"
        
        # Realized vol comparison
        if 'realized_vol' in df.columns:
            summary += "REALIZED VOLATILITY:\n"
            rv = df['realized_vol'].iloc[-1]
            summary += f"  Current RV (30d):    {rv*100:.2f}%\n"
            iv_rv_spread = self.current_implied_vol - rv
            summary += f"  IV-RV Spread:        {iv_rv_spread*100:.2f}pp\n"
            
            if iv_rv_spread > 0.1:
                summary += "  → Implied vol EXPENSIVE vs realized\n"
            elif iv_rv_spread < -0.05:
                summary += "  → Implied vol CHEAP vs realized\n"
            else:
                summary += "  → Implied vol FAIR vs realized\n"
            summary += "\n"
        
        # Regime analysis
        pct = df['iv_percentile'].iloc[-1]
        if pd.notna(pct):
            summary += "REGIME ANALYSIS:\n"
            summary += f"  Current Percentile:  {pct*100:.1f}%\n"
            
            if pct > 0.8:
                summary += "  Regime:              EXTREME HIGH\n"
                summary += "  Signal:              Strong sell volatility\n"
            elif pct > 0.6:
                summary += "  Regime:              ELEVATED\n"
                summary += "  Signal:              Modest sell volatility\n"
            elif pct > 0.4:
                summary += "  Regime:              NORMAL\n"
                summary += "  Signal:              Neutral\n"
            elif pct > 0.2:
                summary += "  Regime:              DEPRESSED\n"
                summary += "  Signal:              Modest buy volatility\n"
            else:
                summary += "  Regime:              EXTREME LOW\n"
                summary += "  Signal:              Strong buy volatility\n"
            summary += "\n"
        
        # Greeks summary
        if self.current_price and self.current_implied_vol:
            summary += "ATM GREEKS (30-DAY):\n"
            S = self.current_price
            K = S
            T = 30 / 365
            r = self.risk_free_rate
            sigma = self.current_implied_vol
            
            delta = BlackScholesGreeks.call_delta(S, K, T, r, sigma)
            gamma = BlackScholesGreeks.gamma(S, K, T, r, sigma)
            vega = BlackScholesGreeks.vega(S, K, T, r, sigma)
            theta = BlackScholesGreeks.call_theta(S, K, T, r, sigma)
            
            summary += f"  Delta:               {delta:.4f}\n"
            summary += f"  Gamma:               {gamma:.6f}\n"
            summary += f"  Vega (per 1% IV):    ${vega:.2f}\n"
            summary += f"  Theta (per day):     ${theta:.2f}\n"
            summary += "\n"
        
        # Data quality
        summary += "DATA QUALITY:\n"
        summary += f"  Total observations:  {len(df)}\n"
        summary += f"  Date range:          {df.index.min().date()} to {df.index.max().date()}\n"
        summary += f"  Missing IV data:     {df['implied_vol'].isna().sum()}\n"
        
        if 'vix' in df.columns:
            summary += f"  Missing VIX data:    {df['vix'].isna().sum()}\n"
        
        summary += "\n" + "=" * 50 + "\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)

    # ==================== Export Methods ====================
    
    def export_data(self):
        """Export comprehensive dataset"""
        if self.volatility_data is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        # Ask for format
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Data")
        export_window.geometry("400x250")
        export_window.transient(self.root)
        export_window.grab_set()
        
        ttk.Label(export_window, text="Select Export Format:", 
                 font=("Arial", 11, "bold")).pack(pady=10)
        
        format_var = tk.StringVar(value="csv")
        
        ttk.Radiobutton(export_window, text="CSV (Comma-Separated Values)", 
                       variable=format_var, value="csv").pack(pady=5)
        ttk.Radiobutton(export_window, text="JSON (JavaScript Object Notation)", 
                       variable=format_var, value="json").pack(pady=5)
        
        ttk.Label(export_window, text="\nInclude:", 
                 font=("Arial", 10, "bold")).pack(pady=5)
        
        include_summary = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_window, text="Summary statistics", 
                       variable=include_summary).pack()
        
        include_greeks = tk.BooleanVar(value=True)
        ttk.Checkbutton(export_window, text="Greeks calculations", 
                       variable=include_greeks).pack()
        
        def do_export():
            fmt = format_var.get()
            
            if fmt == "csv":
                self.export_csv(include_summary.get(), include_greeks.get())
            else:
                self.export_json(include_summary.get(), include_greeks.get())
            
            export_window.destroy()
        
        ttk.Button(export_window, text="Export", 
                  command=do_export).pack(pady=15)
        ttk.Button(export_window, text="Cancel", 
                  command=export_window.destroy).pack()

    def export_csv(self, include_summary=True, include_greeks=True):
        """Export data as CSV"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.symbol_var.get()}_volatility_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filepath:
            return
        
        try:
            df = self.volatility_data.copy()
            
            # Add price data if available
            if self.price_data is not None:
                df['underlying_close'] = self.price_data['close']
            
            # Calculate greeks if requested
            if include_greeks and self.current_price and self.current_implied_vol:
                S = self.current_price
                r = self.risk_free_rate
                T = 30 / 365
                
                df['atm_delta_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.call_delta(S, S, T, r, sigma) 
                    if pd.notna(sigma) else np.nan
                )
                df['atm_gamma_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.gamma(S, S, T, r, sigma) 
                    if pd.notna(sigma) else np.nan
                )
                df['atm_vega_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.vega(S, S, T, r, sigma) 
                    if pd.notna(sigma) else np.nan
                )
                df['atm_theta_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.call_theta(S, S, T, r, sigma) 
                    if pd.notna(sigma) else np.nan
                )
            
            # Export main data
            df.to_csv(filepath)
            
            # Add summary as comments at the end if requested
            if include_summary:
                with open(filepath, 'a') as f:
                    f.write("\n\n# SUMMARY STATISTICS\n")
                    f.write(f"# Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Symbol: {self.symbol_var.get()}\n")
                    f.write(f"# Current IV: {self.current_implied_vol*100:.2f}%\n")
                    
                    if self.current_vix:
                        f.write(f"# Current VIX: {self.current_vix:.2f}\n")
                    
                    if self.current_price:
                        f.write(f"# Current Price: ${self.current_price:.2f}\n")
                    
                    stats = df['implied_vol'].describe()
                    f.write(f"# IV Mean: {stats['mean']*100:.2f}%\n")
                    f.write(f"# IV Std: {stats['std']*100:.2f}%\n")
                    f.write(f"# IV Min: {stats['min']*100:.2f}%\n")
                    f.write(f"# IV Max: {stats['max']*100:.2f}%\n")
            
            self.log_message(f"✓ Data exported successfully to {filepath}")
            messagebox.showinfo("Export Successful", 
                              f"Data exported to:\n{filepath}")
            
        except Exception as e:
            self.log_message(f"Export error: {e}")
            messagebox.showerror("Export Error", str(e))

    def export_json(self, include_summary=True, include_greeks=True):
        """Export data as JSON"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"{self.symbol_var.get()}_volatility_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not filepath:
            return
        
        try:
            df = self.volatility_data.copy()
            
            # Add price data if available
            if self.price_data is not None:
                df['underlying_close'] = self.price_data['close']
            
            # Calculate greeks if requested
            if include_greeks and self.current_price and self.current_implied_vol:
                S = self.current_price
                r = self.risk_free_rate
                T = 30 / 365
                
                df['atm_delta_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.call_delta(S, S, T, r, sigma) 
                    if pd.notna(sigma) else None
                )
                df['atm_gamma_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.gamma(S, S, T, r, sigma) 
                    if pd.notna(sigma) else None
                )
                df['atm_vega_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.vega(S, S, T, r, sigma) 
                    if pd.notna(sigma) else None
                )
                df['atm_theta_30d'] = df['implied_vol'].apply(
                    lambda sigma: BlackScholesGreeks.call_theta(S, S, T, r, sigma) 
                    if pd.notna(sigma) else None
                )
            
            # Convert to JSON-serializable format
            df_reset = df.reset_index()
            df_reset['date'] = df_reset['date'].astype(str)
            
            export_data = {
                "metadata": {
                    "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "symbol": self.symbol_var.get(),
                    "duration": self.duration_var.get(),
                    "total_observations": len(df)
                },
                "current_metrics": {},
                "data": df_reset.to_dict(orient='records')
            }
            
            # Add current metrics
            if self.current_implied_vol:
                export_data["current_metrics"]["implied_vol"] = float(self.current_implied_vol)
            
            if self.current_vix:
                export_data["current_metrics"]["vix"] = float(self.current_vix)
            
            if self.current_price:
                export_data["current_metrics"]["underlying_price"] = float(self.current_price)
            
            # Add summary statistics if requested
            if include_summary:
                stats = df['implied_vol'].describe()
                export_data["statistics"] = {
                    "mean": float(stats['mean']),
                    "std": float(stats['std']),
                    "min": float(stats['min']),
                    "25th_percentile": float(stats['25%']),
                    "median": float(stats['50%']),
                    "75th_percentile": float(stats['75%']),
                    "max": float(stats['max'])
                }
                
                if 'vix' in df.columns and self.current_vix:
                    vix_corr = df['iv_vix_correlation'].iloc[-1]
                    if pd.notna(vix_corr):
                        export_data["statistics"]["iv_vix_correlation"] = float(vix_corr)
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.log_message(f"✓ Data exported successfully to {filepath}")
            messagebox.showinfo("Export Successful", 
                              f"Data exported to:\n{filepath}")
            
        except Exception as e:
            self.log_message(f"Export error: {e}")
            messagebox.showerror("Export Error", str(e))


# ==================== Main Application ====================

def main():
    root = tk.Tk()
    app = AdvancedVolatilityDashboard(root)
    
    # Handle window close
    def on_closing():
        if app.connected:
            if messagebox.askokcancel("Quit", "Disconnect from IBKR and quit?"):
                try:
                    app.ib_app.disconnect()
                except:
                    pass
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED IMPLIED VOLATILITY & GREEKS TRADING DASHBOARD")
    print("=" * 60)
    print()
    print("Features:")
    print("  + Full implied volatility analysis")
    print("  + VIX correlation & comparison")
    print("  + Black-Scholes Greeks calculations")
    print("  + 9-panel professional chart grid")
    print("  + Regime-based trading strategies")
    print("  + CSV & JSON export capabilities")
    print()
    print("Instructions:")
    print("  1. Ensure IBKR TWS or Gateway is running")
    print("  2. Enable API connections in TWS settings")
    print("  3. Connect using the dashboard")
    print("  4. Load data for your symbol")
    print("  5. Run full analysis")
    print()
    print("Starting application...")
    print("=" * 60)
    print()
    
    main()