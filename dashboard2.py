"""Advanced Implied Volatility & VIX Intelligence Suite.

A comprehensive dashboard for analyzing implied volatility data,
VIX correlation, and providing regime-based trading insights using
Interactive Brokers API.
"""

import itertools
import threading
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ibapi.client import EClient
from ibapi.common import BarData, TickerId
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.style.use("seaborn-v0_8-darkgrid")


# ---------------------------------------------------------------------------
# IB API Wrapper
# ---------------------------------------------------------------------------

class IBApp(EWrapper, EClient):
    """Thread-safe IB interface with convenient storage and
    signalling primitives.
    """

    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self._lock = threading.Lock()

        self.historical_data: Dict[int, List[Dict[str, Any]]] = (
            defaultdict(list)
        )
        self.hist_end_events: Dict[int, threading.Event] = {}

        self.contract_details: Dict[int, List[Any]] = defaultdict(list)
        self.contract_events: Dict[int, threading.Event] = {}

        self.option_params: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.option_events: Dict[int, threading.Event] = {}

        self.market_data: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.market_events: Dict[int, threading.Event] = {}

        self.error_messages: deque[str] = deque(maxlen=300)

    def _signal(self, container: Dict[int, threading.Event], req_id: int):
        """Signal waiting event for the given request ID."""
        event = container.get(req_id)
        if isinstance(event, threading.Event):
            event.set()

    def reset_hist_request(self, req_id: int):
        """Reset historical data request."""
        with self._lock:
            if req_id in self.historical_data:
                del self.historical_data[req_id]

    def reset_contract_request(self, req_id: int):
        """Reset contract details request."""
        with self._lock:
            if req_id in self.contract_details:
                del self.contract_details[req_id]

    def reset_option_params(self, req_id: int):
        """Reset option parameters request."""
        with self._lock:
            if req_id in self.option_params:
                del self.option_params[req_id]

    def reset_market_data(self, req_id: int):
        """Reset market data request."""
        with self._lock:
            if req_id in self.market_data:
                del self.market_data[req_id]

    def get_hist_records(self, req_id: int) -> List[Dict[str, Any]]:
        """Get historical data records."""
        with self._lock:
            return list(self.historical_data.get(req_id, []))

    def get_contract_details(self, req_id: int) -> List[Any]:
        """Get contract details."""
        with self._lock:
            return list(self.contract_details.get(req_id, []))

    def get_option_params(self, req_id: int) -> List[Dict[str, Any]]:
        """Get option parameters."""
        with self._lock:
            return list(self.option_params.get(req_id, []))

    def get_market_data(self, req_id: int) -> Dict[str, Any]:
        """Get market data."""
        with self._lock:
            return dict(self.market_data.get(req_id, {}))

    # -------------------- API Callbacks -------------------- #
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Handle IB API errors."""
        msg = f"Error {reqId} | Code {errorCode}: {errorString}"
        print(msg)
        self.error_messages.append(msg)
        self._signal(self.hist_end_events, reqId)
        self._signal(self.contract_events, reqId)
        self._signal(self.option_events, reqId)
        self._signal(self.market_events, reqId)

    def historicalData(self, reqId: TickerId, bar: BarData):
        """Handle historical data callback."""
        record = {
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        with self._lock:
            self.historical_data[reqId].append(record)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Handle end of historical data."""
        self._signal(self.hist_end_events, reqId)

    def contractDetails(self, reqId: int, contractDetails):
        """Handle contract details callback."""
        with self._lock:
            self.contract_details[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId: int):
        """Handle end of contract details."""
        self._signal(self.contract_events, reqId)

    def securityDefinitionOptionParameter(
        self, reqId: int, exchange: str, underlyingConId: int,
        tradingClass: str, multiplier: str, expirations: set, strikes: set
    ):
        """Handle security definition option parameter callback."""
        params = {
            "exchange": exchange,
            "tradingClass": tradingClass,
            "multiplier": multiplier,
            "expirations": sorted(expirations),
            "strikes": sorted(strikes),
        }
        with self._lock:
            self.option_params[reqId].append(params)

    def securityDefinitionOptionParameterEnd(self, reqId: int):
        """Handle end of option parameter definition."""
        self._signal(self.option_events, reqId)

    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Handle tick price callback."""
        if price > 0:
            with self._lock:
                self.market_data[reqId]['price'] = price
                self.market_data[reqId]['tickType'] = tickType
            self._signal(self.market_events, reqId)

    def tickSize(self, reqId: TickerId, tickType: int, size: float):
        """Handle tick size callback."""
        with self._lock:
            self.market_data[reqId]['size'] = size

    def tickString(self, reqId: int, tickType: int, value: str):
        """Handle tick string callback."""
        with self._lock:
            self.market_data[reqId][f"tickString_{tickType}"] = value


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def black_scholes_call_greeks(
    spot: float,
    strike: float,
    vol: float,
    r: float,
    t: float
) -> Tuple[float, float, float, float]:
    """Calculate Black-Scholes Greeks for an ATM call option.

    Args:
        spot: Current stock price
        strike: Option strike price
        vol: Volatility (annualized)
        r: Risk-free rate
        t: Time to expiry (in years)

    Returns:
        Tuple of (Delta, Gamma, Vega per 1%, Theta per day)
    """
    if spot <= 0 or strike <= 0 or vol <= 0 or t <= 0:
        return np.nan, np.nan, np.nan, np.nan

    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (spot * vol * sqrt_t)
    vega = spot * norm.pdf(d1) * sqrt_t
    theta = -(spot * norm.pdf(d1) * vol) / (2 * sqrt_t) - \
        r * strike * np.exp(-r * t) * norm.cdf(d2)
    theta = theta / 365.0  # per day

    return delta, gamma, vega / 100.0, theta


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class ImpliedVolatilityDashboard:
    """Advanced volatility and VIX intelligence dashboard."""

    def __init__(self, root: tk.Tk):
        """Initialize the dashboard.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(
            "Advanced Implied Volatility & VIX Intelligence Suite"
        )
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

        # Core data attributes
        self.equity_iv: Optional[pd.DataFrame] = None
        self.price_data: Optional[pd.DataFrame] = None
        self.vix_data: Optional[pd.DataFrame] = None
        self.combined_data: Optional[pd.DataFrame] = None
        self.analysis_data: Optional[pd.DataFrame] = None
        self.greeks_profile: Optional[pd.DataFrame] = None

        # Current market data
        self.current_implied_vol: Optional[float] = None
        self.current_realized_vol: Optional[float] = None
        self.current_atm_iv: Optional[float] = None
        self.current_price: Optional[float] = None
        self.current_vix: Optional[float] = None

        # IB API components
        self.ib_app = IBApp()
        self.connected = False
        self.req_id_iter = itertools.count(1)

        # Configuration parameters
        self.request_timeout = 40
        self.vol_annualization = 252
        self.percentile_window = 252
        self.mc_paths = 1500
        self.risk_free_rate = 0.02

        self._build_ui()

    # -------------------- UI Construction -------------------- #
    def _build_ui(self):
        """Build the user interface."""
        # Paned layout (modern resize experience)
        outer_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        outer_pane.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(outer_pane, padding=10)
        outer_pane.add(control_frame, weight=1)

        right_pane = ttk.PanedWindow(outer_pane, orient=tk.VERTICAL)
        outer_pane.add(right_pane, weight=3)

        summary_frame = ttk.Frame(right_pane, padding=10)
        right_pane.add(summary_frame, weight=1)

        plot_frame = ttk.Frame(right_pane, padding=10)
        right_pane.add(plot_frame, weight=4)

        # ---------------- Controls (left pane) ---------------- #
        self._build_connection_controls(control_frame)
        self._build_data_controls(control_frame)
        self._build_export_controls(control_frame)
        self._build_metrics_display(control_frame)
        self._build_status_log(control_frame)

        # ---------------- Summary & plots (right pane) ---------------- #
        self._build_summary_and_plots(summary_frame, plot_frame)

    def _build_connection_controls(self, parent):
        """Build connection controls section."""
        conn_frame = ttk.LabelFrame(
            parent, text="Interactive Brokers Connection", padding=10
        )
        conn_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conn_frame, text="Host:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.host_var = tk.StringVar(value="127.0.0.1")
        ttk.Entry(conn_frame, textvariable=self.host_var, width=15).grid(
            row=0, column=1, padx=5
        )

        ttk.Label(conn_frame, text="Port:").grid(
            row=0, column=2, sticky=tk.W, pady=2
        )
        self.port_var = tk.StringVar(value="7497")
        ttk.Entry(conn_frame, textvariable=self.port_var, width=8).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(conn_frame, text="Client ID:").grid(
            row=0, column=4, sticky=tk.W, pady=2
        )
        self.client_id_var = tk.IntVar(value=0)
        ttk.Entry(conn_frame, textvariable=self.client_id_var, width=6).grid(
            row=0, column=5, padx=5
        )

        self.connect_btn = ttk.Button(
            conn_frame, text="Connect", command=self.connect_ib
        )
        self.connect_btn.grid(row=0, column=6, padx=5)

        self.disconnect_btn = ttk.Button(
            conn_frame, text="Disconnect", command=self.disconnect_ib,
            state=tk.DISABLED
        )
        self.disconnect_btn.grid(row=0, column=7, padx=5)

    def _build_data_controls(self, parent):
        """Build data controls section."""
        query_frame = ttk.LabelFrame(
            parent, text="Market Data & Analysis Controls", padding=10
        )
        query_frame.pack(fill=tk.X, pady=5)

        ttk.Label(query_frame, text="Symbol:").grid(
            row=0, column=0, sticky=tk.W
        )
        self.symbol_var = tk.StringVar(value="SPY")
        ttk.Entry(query_frame, textvariable=self.symbol_var, width=12).grid(
            row=0, column=1, padx=5
        )

        ttk.Label(query_frame, text="History Duration:").grid(
            row=0, column=2, sticky=tk.W
        )
        self.duration_var = tk.StringVar(value="3 Y")
        ttk.Entry(query_frame, textvariable=self.duration_var, width=10).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(query_frame, text="Bar Size:").grid(
            row=0, column=4, sticky=tk.W
        )
        self.bar_size_var = tk.StringVar(value="1 day")
        ttk.Entry(query_frame, textvariable=self.bar_size_var, width=10).grid(
            row=0, column=5, padx=5
        )

        self.load_btn = ttk.Button(
            query_frame, text="Load Symbol + VIX History",
            command=self.load_historical_data, state=tk.DISABLED
        )
        self.load_btn.grid(row=0, column=6, padx=5)

        self.live_btn = ttk.Button(
            query_frame, text="Refresh Live Quotes",
            command=self.refresh_live_quotes, state=tk.DISABLED
        )
        self.live_btn.grid(row=0, column=7, padx=5)

        self.iv_btn = ttk.Button(
            query_frame, text="Fetch ATM IV Snapshot",
            command=self.fetch_current_atm_iv, state=tk.DISABLED
        )
        self.iv_btn.grid(row=0, column=8, padx=5)

        self.analyze_btn = ttk.Button(
            query_frame, text="Run Advanced Analytics",
            command=self.run_full_analysis, state=tk.DISABLED
        )
        self.analyze_btn.grid(row=0, column=9, padx=5)

    def _build_export_controls(self, parent):
        """Build export controls section."""
        export_frame = ttk.LabelFrame(parent, text="Data Export", padding=10)
        export_frame.pack(fill=tk.X, pady=5)

        self.export_csv_btn = ttk.Button(
            export_frame, text="Export Dataset (CSV)",
            command=lambda: self.export_data("csv"), state=tk.DISABLED
        )
        self.export_csv_btn.grid(row=0, column=0, padx=5)

        self.export_json_btn = ttk.Button(
            export_frame, text="Export Dataset (JSON)",
            command=lambda: self.export_data("json"), state=tk.DISABLED
        )
        self.export_json_btn.grid(row=0, column=1, padx=5)

    def _build_metrics_display(self, parent):
        """Build metrics display section."""
        metrics_frame = ttk.LabelFrame(
            parent, text="Real-Time Metrics & Regime Intelligence",
            padding=10
        )
        metrics_frame.pack(fill=tk.X, pady=5)

        rows = [
            ("Underlying Spot:", "underlying_price_lbl"),
            ("Live VIX:", "live_vix_lbl"),
            ("Hist IV (close):", "hist_iv_lbl"),
            ("ATM IV Snapshot:", "atm_iv_lbl"),
            ("Realized Vol (30d):", "hv_lbl"),
            ("IV / HV Ratio:", "iv_hv_ratio_lbl"),
            ("IV Percentile (252d):", "iv_percentile_lbl"),
            ("IV Z-Score:", "iv_zscore_lbl"),
            ("VIX Correlation (252d):", "vix_corr_lbl"),
            ("IV – VIX Premium:", "iv_vix_premium_lbl"),
        ]
        for idx, (label, attr) in enumerate(rows):
            ttk.Label(metrics_frame, text=label).grid(
                row=idx, column=0, sticky=tk.W, pady=2
            )
            setattr(
                self, attr,
                ttk.Label(metrics_frame, text="N/A", font=("Arial", 11, "bold"))
            )
            getattr(self, attr).grid(
                row=idx, column=1, sticky=tk.W, padx=5, pady=2
            )

        ttk.Label(
            metrics_frame, text="Volatility Regime:"
        ).grid(row=len(rows), column=0, sticky=tk.W, pady=(10, 2))
        self.regime_lbl = ttk.Label(
            metrics_frame, text="N/A", font=("Arial", 12, "bold")
        )
        self.regime_lbl.grid(
            row=len(rows), column=1, sticky=tk.W, pady=(10, 2)
        )

        ttk.Label(
            metrics_frame, text="Strategy Lens:"
        ).grid(row=len(rows) + 1, column=0, sticky=tk.W, pady=2)
        self.strategy_lbl = ttk.Label(
            metrics_frame, text="N/A", wraplength=320, justify=tk.LEFT
        )
        self.strategy_lbl.grid(row=len(rows) + 1, column=1, sticky=tk.W, pady=2)

    def _build_status_log(self, parent):
        """Build status log section."""
        status_frame = ttk.LabelFrame(parent, text="Event Log", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_text = scrolledtext.ScrolledText(
            status_frame, height=12, font=("Consolas", 10)
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def _build_summary_and_plots(self, summary_frame, plot_frame):
        """Build summary and plots section."""
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        self.summary_text = scrolledtext.ScrolledText(
            summary_frame, font=("Consolas", 10)
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig, axes = plt.subplots(3, 3, figsize=(22, 13))
        self.ax_grid = axes.flatten()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # -------------------- Utility methods -------------------- #
    def log(self, msg: str):
        """Log a message to the status text area.

        Args:
            msg: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def next_req_id(self) -> int:
        """Get next request ID."""
        return next(self.req_id_iter)

    def create_stock_contract(self, symbol: str) -> Contract:
        """Create a stock contract.

        Args:
            symbol: Stock symbol

        Returns:
            IB Contract object
        """
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def create_vix_contract(self) -> Contract:
        """Create a VIX contract.

        Returns:
            IB Contract object for VIX
        """
        contract = Contract()
        contract.symbol = "VIX"
        contract.secType = "IND"
        contract.exchange = "CBOE"
        contract.currency = "USD"
        return contract

    def flush_errors(self):
        """Flush error messages to log."""
        while self.ib_app.error_messages:
            self.log(self.ib_app.error_messages.popleft())

    # -------------------- IB Connection -------------------- #
    def connect_ib(self):
        """Connect to Interactive Brokers."""
        if self.connected:
            self.log("Already connected.")
            return

        host = self.host_var.get().strip()
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer.")
            return

        client_id = self.client_id_var.get()
        self.log(
            f"Connecting to IBKR Gateway/TWS at {host}:{port} "
            f"(Client {client_id})..."
        )

        def api_loop():
            """API connection loop."""
            try:
                self.ib_app.run()
            except Exception as exc:
                self.log(f"API loop stopped: {exc}")

        try:
            # Start API loop thread
            self.api_thread = threading.Thread(target=api_loop, daemon=True)
            self.api_thread.start()

            connect_result = self.ib_app.connect(host, port, clientId=client_id)
            if connect_result is not None and connect_result < 0:
                self.log("Connect call returned negative status.")

            for _ in range(200):
                if self.ib_app.isConnected():
                    self.connected = True
                    break
                time.sleep(0.1)

            if not self.connected:
                self.log("Connection timed out.")
                return

            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.load_btn.config(state=tk.NORMAL)
            self.live_btn.config(state=tk.NORMAL)
            self.iv_btn.config(state=tk.NORMAL)
            self.log("Connected successfully.")

        except Exception as exc:
            self.log(f"Connection failed: {exc}")

    def disconnect_ib(self):
        """Disconnect from Interactive Brokers."""
        if not self.connected:
            self.log("Not currently connected.")
            return
        try:
            self.log("Disconnecting from IBKR...")
            self.ib_app.disconnect()
            for _ in range(50):
                if not self.ib_app.isConnected():
                    break
                time.sleep(0.1)

            self.connected = False
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.load_btn.config(state=tk.DISABLED)
            self.live_btn.config(state=tk.DISABLED)
            self.iv_btn.config(state=tk.DISABLED)
            self.analyze_btn.config(state=tk.DISABLED)
            self.export_csv_btn.config(state=tk.DISABLED)
            self.export_json_btn.config(state=tk.DISABLED)

            self.clear_data()
            self.log("Disconnected. Safe trading!")
        except Exception as exc:
            self.log(f"Disconnect error: {exc}")

    # -------------------- Data lifecycle -------------------- #
    def clear_data(self):
        """Clear all stored data and reset UI."""
        self.equity_iv = None
        self.price_data = None
        self.vix_data = None
        self.combined_data = None
        self.analysis_data = None
        self.greeks_profile = None
        self.current_implied_vol = None
        self.current_realized_vol = None
        self.current_atm_iv = None
        self.current_price = None
        self.current_vix = None

        label_names = (
            "underlying_price_lbl", "live_vix_lbl", "hist_iv_lbl",
            "atm_iv_lbl", "hv_lbl", "iv_hv_ratio_lbl", "iv_percentile_lbl",
            "iv_zscore_lbl", "vix_corr_lbl", "iv_vix_premium_lbl"
        )
        for label_name in label_names:
            getattr(self, label_name).config(text="N/A", foreground="black")

        self.regime_lbl.config(text="N/A", foreground="black")
        self.strategy_lbl.config(text="N/A")

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Load data to see analytics.\n")

        for ax in self.ax_grid:
            ax.clear()
        self.fig.suptitle(
            "Volatility Intelligence Dashboard", fontsize=16, fontweight="bold"
        )
        self.canvas.draw_idle()

    # -------------------- Request wrappers -------------------- #
    def request_historical(
        self, contract: Contract, duration: str, bar_size: str, what: str
    ) -> Optional[pd.DataFrame]:
        """Request historical data from IB.

        Args:
            contract: IB contract
            duration: Duration string
            bar_size: Bar size string
            what: What to show parameter

        Returns:
            DataFrame with historical data or None if failed
        """
        req_id = self.next_req_id()
        event = threading.Event()
        self.ib_app.hist_end_events[req_id] = event
        self.ib_app.reset_hist_request(req_id)

        try:
            self.ib_app.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what,
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[]
            )

            if not event.wait(self.request_timeout):
                self.log(
                    f"Historical request {req_id} timed out ({what}). Cancelling."
                )
                self.ib_app.cancelHistoricalData(req_id)
                return None

            rows = self.ib_app.get_hist_records(req_id)
            if not rows:
                self.log(f"No data delivered for req {req_id} ({what}).")
                return None

            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df

        except Exception as exc:
            self.log(f"Historical request failure ({what}): {exc}")
            return None
        finally:
            self.ib_app.hist_end_events.pop(req_id, None)

    def request_market_snapshot(self, contract: Contract) -> Optional[float]:
        """Request market data snapshot.

        Args:
            contract: IB contract

        Returns:
            Price or None if failed
        """
        req_id = self.next_req_id()
        event = threading.Event()
        self.ib_app.market_events[req_id] = event
        self.ib_app.reset_market_data(req_id)

        try:
            self.ib_app.reqMktData(req_id, contract, "", True, False, [])
            if not event.wait(10):
                self.log("Market data snapshot timeout.")
                self.ib_app.cancelMktData(req_id)
                return None

            data = self.ib_app.get_market_data(req_id)
            price = data.get("price")
            return price
        except Exception as exc:
            self.log(f"Snapshot error: {exc}")
            return None
        finally:
            self.ib_app.cancelMktData(req_id)
            self.ib_app.market_events.pop(req_id, None)

    # -------------------- Data loading -------------------- #
    def load_historical_data(self):
        """Load historical data for analysis."""
        if not self.connected:
            messagebox.showwarning(
                "Connection required", "Please connect to IBKR first."
            )
            return

        symbol = self.symbol_var.get().strip().upper()
        duration = self.duration_var.get().strip()
        bar_size = self.bar_size_var.get().strip()

        if not symbol:
            messagebox.showerror("Missing symbol", "Please enter a symbol.")
            return

        self.log(
            f"Loading {duration} of {bar_size} data for {symbol} and VIX..."
        )

        stock_contract = self.create_stock_contract(symbol)
        vix_contract = self.create_vix_contract()

        iv_df = self.request_historical(
            stock_contract, duration, bar_size, "OPTION_IMPLIED_VOLATILITY"
        )
        price_df = self.request_historical(
            stock_contract, duration, bar_size, "TRADES"
        )
        vix_df = self.request_historical(
            vix_contract, duration, bar_size, "TRADES"
        )

        self.flush_errors()

        if iv_df is None or price_df is None or vix_df is None:
            self.log("Failed to download all required series.")
            self.analyze_btn.config(state=tk.DISABLED)
            return

        iv_df.rename(columns={"close": "implied_vol"}, inplace=True)
        price_df.rename(columns={"close": "close_price"}, inplace=True)
        vix_df.rename(columns={"close": "vix_close"}, inplace=True)

        # Align data
        common_idx = (iv_df.index
                     .intersection(price_df.index)
                     .intersection(vix_df.index))
        if common_idx.empty:
            self.log("No overlapping dates between symbol and VIX.")
            self.analyze_btn.config(state=tk.DISABLED)
            return

        iv_df = iv_df.loc[common_idx]
        price_df = price_df.loc[common_idx]
        vix_df = vix_df.loc[common_idx]

        # Calculate realized volatility
        price_df['log_ret'] = np.log(price_df['close_price']).diff()
        hv_windows = [5, 10, 21, 30, 60]
        hv_df = pd.DataFrame(index=price_df.index)
        for window in hv_windows:
            hv_df[f'hv_{window}'] = (
                price_df['log_ret']
                .rolling(window, min_periods=max(5, window // 2))
                .std() * np.sqrt(self.vol_annualization)
            )

        hv_df['hv_ewma'] = (
            price_df['log_ret']
            .ewm(alpha=1 - 0.94, adjust=False)
            .std() * np.sqrt(self.vol_annualization)
        )

        # Combine datasets
        combined = iv_df[['implied_vol']].join(
            price_df[['close_price', 'log_ret']], how='left'
        )
        combined = combined.join(vix_df[['vix_close']], how='left')
        combined = combined.join(hv_df, how='left')

        # Calculate analytics
        combined['iv_percentile'] = (
            combined['implied_vol']
            .rolling(self.percentile_window, min_periods=60)
            .apply(
                lambda s: stats.percentileofscore(s, s.iloc[-1], kind="mean")
                / 100, raw=False
            )
        )
        rolling_mean = (
            combined['implied_vol']
            .rolling(self.percentile_window, min_periods=60)
            .mean()
        )
        rolling_std = (
            combined['implied_vol']
            .rolling(self.percentile_window, min_periods=60)
            .std()
        )
        combined['iv_zscore'] = (
            (combined['implied_vol'] - rolling_mean) /
            rolling_std.replace(0, np.nan)
        )

        combined['iv_vs_vix_premium'] = (
            combined['implied_vol'] - combined['vix_close']
        )
        combined['iv_vix_ratio'] = (
            combined['implied_vol'] / combined['vix_close']
        )
        combined['rolling_corr_252'] = (
            combined['implied_vol'].rolling(252).corr(combined['vix_close'])
        )

        # Store data
        self.equity_iv = iv_df
        self.price_data = price_df
        self.vix_data = vix_df
        self.combined_data = combined

        # Update current values
        self.current_implied_vol = (
            combined['implied_vol'].iloc[-1]
            if combined['implied_vol'].notna().any() else None
        )
        self.current_realized_vol = (
            combined['hv_30'].dropna().iloc[-1]
            if combined['hv_30'].notna().any() else None
        )
        self.current_vix = (
            combined['vix_close'].dropna().iloc[-1]
            if combined['vix_close'].notna().any() else None
        )
        self.current_price = (
            combined['close_price'].dropna().iloc[-1]
            if combined['close_price'].notna().any() else None
        )

        # Update UI
        self.update_metric_labels()
        self.refresh_live_quotes()
        self.analyze_btn.config(state=tk.NORMAL)
        self.export_csv_btn.config(state=tk.NORMAL)
        self.export_json_btn.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(
            tk.END,
            f"Historical sample loaded: {len(combined):,} observations "
            f"spanning {combined.index.min().date()} to "
            f"{combined.index.max().date()}.\n"
        )
        self.log("Historical data ready. Run analytics to see full diagnostics.")

    def refresh_live_quotes(self):
        """Refresh live market quotes."""
        if not self.connected:
            self.log("Connect first to refresh live data.")
            return

        symbol = self.symbol_var.get().strip().upper()
        stock_contract = self.create_stock_contract(symbol)
        vix_contract = self.create_vix_contract()

        spot = self.request_market_snapshot(stock_contract)
        vix = self.request_market_snapshot(vix_contract)

        if spot:
            self.current_price = spot
        if vix:
            self.current_vix = vix

        self.update_metric_labels()
        self.log("Live quotes refreshed.")

    def fetch_current_atm_iv(self):
        """Fetch current at-the-money implied volatility."""
        if not self.connected or self.combined_data is None:
            messagebox.showwarning(
                "Requirement", "Connect and load historical data first."
            )
            return

        symbol = self.symbol_var.get().strip().upper()

        self.log(f"Fetching ATM IV snapshot for {symbol}...")
        stock_contract = self.create_stock_contract(symbol)

        # Get contract details
        req_id_contract = self.next_req_id()
        event_contract = threading.Event()
        self.ib_app.contract_events[req_id_contract] = event_contract
        self.ib_app.reset_contract_request(req_id_contract)

        self.ib_app.reqContractDetails(req_id_contract, stock_contract)
        if not event_contract.wait(10):
            self.log("Contract details timeout.")
            self.ib_app.contract_events.pop(req_id_contract, None)
            return

        details = self.ib_app.get_contract_details(req_id_contract)
        self.ib_app.contract_events.pop(req_id_contract, None)
        if not details:
            self.log("No contract details returned.")
            return
        con_id = details[0].contract.conId

        # Get option parameters
        req_id_opt = self.next_req_id()
        event_opt = threading.Event()
        self.ib_app.option_events[req_id_opt] = event_opt
        self.ib_app.reset_option_params(req_id_opt)

        self.ib_app.reqSecDefOptParams(req_id_opt, symbol, "", "STK", con_id)
        if not event_opt.wait(10):
            self.log("Option parameter request timeout.")
            self.ib_app.option_events.pop(req_id_opt, None)
            return

        params = self.ib_app.get_option_params(req_id_opt)
        self.ib_app.option_events.pop(req_id_opt, None)

        if not params:
            self.log("Failed to obtain option parameters.")
            return

        smart = next(
            (p for p in params if p['exchange'] == 'SMART'), params[0]
        )
        expirations = smart['expirations']
        strikes = smart['strikes']
        multiplier = smart['multiplier']
        trading_class = smart['tradingClass']

        # Find suitable expiration
        today = datetime.today()
        future_exps = [
            exp for exp in expirations
            if datetime.strptime(exp, "%Y%m%d") > today + timedelta(days=7)
        ]
        if not future_exps:
            self.log("No suitable expirations >7 days.")
            return
        expiry = min(future_exps)

        # Find ATM strike
        spot = self.current_price or self.request_market_snapshot(stock_contract)
        if spot is None:
            self.log("Cannot determine spot price.")
            return

        atm_strike = min(strikes, key=lambda x: abs(x - spot))

        # Create option contract
        option_contract = Contract()
        option_contract.symbol = symbol
        option_contract.secType = "OPT"
        option_contract.exchange = "SMART"
        option_contract.currency = "USD"
        option_contract.lastTradeDateOrContractMonth = expiry
        option_contract.strike = atm_strike
        option_contract.right = "C"
        option_contract.multiplier = multiplier
        option_contract.tradingClass = trading_class

        # Request IV
        req_id_iv = self.next_req_id()
        event_iv = threading.Event()
        self.ib_app.market_events[req_id_iv] = event_iv
        self.ib_app.reset_market_data(req_id_iv)

        self.ib_app.reqMktData(req_id_iv, option_contract, "106", True, False, [])

        if not event_iv.wait(10):
            self.log("ATM IV snapshot timeout.")
            self.ib_app.cancelMktData(req_id_iv)
            self.ib_app.market_events.pop(req_id_iv, None)
            return

        data = self.ib_app.get_market_data(req_id_iv)
        self.ib_app.cancelMktData(req_id_iv)
        self.ib_app.market_events.pop(req_id_iv, None)

        implied = data.get("price")
        if implied is None or implied <= 0:
            self.log("No model IV returned.")
            return

        self.current_atm_iv = implied
        self.update_metric_labels()
        self.log(f"ATM IV snapshot: {implied:.4f} ({implied*100:.2f}%)")

    # -------------------- Analytics -------------------- #
    def run_full_analysis(self):
        """Run comprehensive volatility analysis."""
        if self.combined_data is None or self.price_data is None:
            messagebox.showwarning("Load data", "Please load historical data first.")
            return

        combined = self.combined_data.copy()
        price = self.price_data.copy()

        # Forward metrics
        combined['forward_iv_30'] = (
            combined['implied_vol'].rolling(30).mean().shift(-29)
        )
        combined['forward_hv_30'] = (
            price['log_ret'].rolling(30).std().shift(-29) *
            np.sqrt(self.vol_annualization)
        )
        combined['forward_return_30'] = np.log(
            price['close_price'].shift(-30) / price['close_price']
        )
        combined['iv_forward_gap'] = (
            combined['forward_iv_30'] - combined['implied_vol']
        )
        combined['forward_return_pct'] = np.exp(
            combined['forward_return_30']
        ) - 1

        analysis_df = combined.dropna(
            subset=['forward_iv_30', 'forward_hv_30', 'forward_return_30']
        ).copy()
        if analysis_df.empty or len(analysis_df) < 40:
            self.log("Insufficient overlapping history for forward analysis.")
            return

        # Regressions
        reg_iv_fwd_iv = stats.linregress(
            analysis_df['implied_vol'], analysis_df['forward_iv_30']
        )
        reg_iv_fwd_hv = stats.linregress(
            analysis_df['implied_vol'], analysis_df['forward_hv_30']
        )

        slope = reg_iv_fwd_iv.slope
        intercept = reg_iv_fwd_iv.intercept
        split_threshold = (
            intercept / (1 - slope) if abs(1 - slope) > 1e-6
            else analysis_df['implied_vol'].median()
        )

        high_mask = analysis_df['implied_vol'] > split_threshold
        low_mask = ~high_mask

        reg_high = reg_low = None
        if high_mask.sum() >= 15:
            reg_high = stats.linregress(
                analysis_df.loc[high_mask, 'implied_vol'],
                (analysis_df.loc[high_mask, 'forward_hv_30'] -
                 analysis_df.loc[high_mask, 'implied_vol'])
            )
        if low_mask.sum() >= 15:
            reg_low = stats.linregress(
                analysis_df.loc[low_mask, 'implied_vol'],
                (analysis_df.loc[low_mask, 'forward_hv_30'] -
                 analysis_df.loc[low_mask, 'implied_vol'])
            )

        # VIX correlation
        vix_corr = (
            combined['implied_vol'].rolling(252).corr(combined['vix_close'])
        )
        current_corr = (
            vix_corr.dropna().iloc[-1] if vix_corr.notna().any() else None
        )

        # ARIMA forecast
        forecast = None
        forecast_dates = None
        try:
            arima = ARIMA(combined['implied_vol'].dropna(), order=(1, 1, 1))
            fit = arima.fit()
            forecast_steps = 30
            forecast = fit.forecast(forecast_steps)
            last = combined.index[-1]
            forecast_dates = pd.bdate_range(
                last + timedelta(days=1), periods=forecast_steps
            )
        except Exception as exc:
            self.log(f"ARIMA warning: {exc}")

        # Greeks simulation
        self.greeks_profile = self.simulate_greeks()

        # Monte Carlo
        mc_summary = self.run_monte_carlo(price)

        # Regime summary table
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ['Very Low', 'Low', 'Normal', 'Elevated', 'High']
        analysis_df['iv_regime_bucket'] = pd.cut(
            analysis_df['iv_percentile'], bins=bins, labels=labels,
            include_lowest=True
        )
        regime_stats = analysis_df.groupby('iv_regime_bucket').agg({
            'forward_return_pct': ['mean', 'std', 'count'],
            'iv_forward_gap': 'mean',
            'iv_vix_ratio': 'mean'
        }).round(4)

        # Information coefficients
        ic_iv_ret = analysis_df['implied_vol'].corr(
            analysis_df['forward_return_30']
        )
        ic_premium_ret = (
            analysis_df['iv_vs_vix_premium'].corr(analysis_df['forward_return_30'])
        )

        # Save analysis data
        self.analysis_data = analysis_df

        # Update summary
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(
            tk.END, "=== Regime-Conditioned Forward Outcomes ===\n"
        )
        self.summary_text.insert(tk.END, regime_stats.to_string())
        self.summary_text.insert(tk.END, "\n\n=== Predictive Regressions ===\n")
        self.summary_text.insert(
            tk.END,
            f"Current IV → Forward IV: slope={reg_iv_fwd_iv.slope:.3f}, "
            f"intercept={reg_iv_fwd_iv.intercept:.3f}, "
            f"R²={reg_iv_fwd_iv.rvalue ** 2:.3f}\n"
        )
        self.summary_text.insert(
            tk.END,
            f"Current IV → Forward Realized Vol: slope={reg_iv_fwd_hv.slope:.3f}, "
            f"intercept={reg_iv_fwd_hv.intercept:.3f}, "
            f"R²={reg_iv_fwd_hv.rvalue ** 2:.3f}\n"
        )
        self.summary_text.insert(
            tk.END, f"Regime split threshold (IV): {split_threshold:.4f}\n"
        )
        if reg_high:
            self.summary_text.insert(
                tk.END,
                f"High regime premium regression: slope={reg_high.slope:.3f}, "
                f"R²={reg_high.rvalue ** 2:.3f}\n"
            )
        if reg_low:
            self.summary_text.insert(
                tk.END,
                f"Low regime premium regression: slope={reg_low.slope:.3f}, "
                f"R²={reg_low.rvalue ** 2:.3f}\n"
            )
        self.summary_text.insert(tk.END, "\n=== Correlation & Predictive Power ===\n")
        if current_corr is not None:
            self.summary_text.insert(
                tk.END,
                f"Rolling 252d IV-VIX correlation (latest): {current_corr:.3f}\n"
            )
        self.summary_text.insert(
            tk.END, f"IC (IV vs 30d return): {ic_iv_ret:.3f}\n"
        )
        self.summary_text.insert(
            tk.END, f"IC (IV-VIX premium vs 30d return): {ic_premium_ret:.3f}\n"
        )

        if mc_summary:
            self.summary_text.insert(
                tk.END, "\n=== Monte Carlo 30-Day Return Scenario ===\n"
            )
            self.summary_text.insert(
                tk.END, f"5th percentile: {mc_summary['p5']:.2f}%\n"
            )
            self.summary_text.insert(
                tk.END, f"Median: {mc_summary['p50']:.2f}%\n"
            )
            self.summary_text.insert(
                tk.END, f"95th percentile: {mc_summary['p95']:.2f}%\n"
            )

        # Render charts
        self.update_charts(
            combined, analysis_df, reg_iv_fwd_iv, reg_iv_fwd_hv,
            forecast, forecast_dates, vix_corr
        )

        self.update_regime_intelligence()
        self.log("Advanced analytics completed.")
        self.analyze_btn.config(state=tk.NORMAL)

    def simulate_greeks(self) -> Optional[pd.DataFrame]:
        """Simulate option Greeks over time.

        Returns:
            DataFrame with Greek values or None if insufficient data
        """
        if self.current_price is None:
            return None

        iv = self.current_atm_iv or self.current_implied_vol
        if iv is None or not np.isfinite(iv):
            return None

        strike = self.current_price
        t_days = np.arange(1, 91)
        greeks = []

        for days in t_days:
            t = days / self.vol_annualization
            delta, gamma, vega, theta = black_scholes_call_greeks(
                self.current_price, strike, iv, self.risk_free_rate, t
            )
            greeks.append((days, delta, gamma, vega, theta))

        df = pd.DataFrame(greeks, columns=['days', 'delta', 'gamma', 'vega', 'theta'])
        df.set_index('days', inplace=True)
        return df

    def run_monte_carlo(
        self, price_df: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Run Monte Carlo simulation for return scenarios.

        Args:
            price_df: Price data DataFrame

        Returns:
            Dictionary with percentile returns or None if failed
        """
        if price_df is None or price_df['log_ret'].dropna().empty:
            return None

        mu = price_df['log_ret'].mean() * self.vol_annualization
        sigma = price_df['log_ret'].std() * np.sqrt(self.vol_annualization)

        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return None

        rng = np.random.default_rng(42)
        dt = 30 / self.vol_annualization
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        returns = rng.normal(drift, vol, size=self.mc_paths)
        pct_returns = (np.exp(returns) - 1) * 100

        return {
            "p5": np.percentile(pct_returns, 5),
            "p50": np.percentile(pct_returns, 50),
            "p95": np.percentile(pct_returns, 95)
        }

    # -------------------- Visuals -------------------- #
    def update_charts(
        self, combined: pd.DataFrame, analysis_df: pd.DataFrame,
        reg_iv_fwd_iv, reg_iv_fwd_hv, forecast, forecast_dates, vix_corr
    ):
        """Update all charts with current data.

        Args:
            combined: Combined dataset
            analysis_df: Analysis dataset
            reg_iv_fwd_iv: IV forward regression
            reg_iv_fwd_hv: HV forward regression
            forecast: ARIMA forecast
            forecast_dates: Forecast dates
            vix_corr: VIX correlation series
        """
        axes = self.ax_grid

        # 1) IV, HV
        axes[0].clear()
        axes[0].plot(combined.index, combined['implied_vol'], label='IV', lw=1.6)
        if 'hv_30' in combined:
            axes[0].plot(combined.index, combined['hv_30'], label='HV 30d', lw=1.1)
        if 'hv_ewma' in combined:
            axes[0].plot(combined.index, combined['hv_ewma'], label='HV EWMA', lw=1)
        axes[0].set_title("Implied vs Realized Volatility")
        axes[0].set_ylabel("Vol")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # 2) IV & VIX time series
        axes[1].clear()
        axes[1].plot(combined.index, combined['implied_vol'], label='IV', lw=1.4)
        axes[1].plot(combined.index, combined['vix_close'], label='VIX', lw=1.4, alpha=0.7)
        axes[1].set_title("Symbol IV vs VIX")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # 3) Percentile & Z-score
        axes[2].clear()
        axes[2].plot(combined.index, combined['iv_percentile'] * 100, label='Percentile', lw=1.4)
        axes[2].axhline(80, color='red', linestyle='--', alpha=0.5)
        axes[2].axhline(20, color='green', linestyle='--', alpha=0.5)
        axes[2].set_title("IV Percentile (252d)")
        axes[2].set_ylabel("Percentile (%)")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        # 4) IV vs VIX scatter
        axes[3].clear()
        axes[3].scatter(combined['vix_close'], combined['implied_vol'], alpha=0.5, s=15)
        if combined['vix_close'].notna().any():
            slope, intercept, r, _, _ = stats.linregress(
                combined['vix_close'].dropna(), combined['implied_vol'].dropna()
            )
            x_vals = np.linspace(combined['vix_close'].min(), combined['vix_close'].max(), 100)
            axes[3].plot(x_vals, slope * x_vals + intercept, color='red', lw=2,
                        label=f"Fit (R²={r ** 2:.2f})")
        axes[3].set_title("IV vs VIX Scatter")
        axes[3].set_xlabel("VIX")
        axes[3].set_ylabel("IV")
        axes[3].grid(alpha=0.3)
        axes[3].legend()

        # 5) Rolling correlation
        axes[4].clear()
        axes[4].plot(vix_corr.index, vix_corr, lw=1.4)
        axes[4].set_title("Rolling 252-Day IV/VIX Correlation")
        axes[4].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[4].set_ylabel("Correlation")
        axes[4].grid(alpha=0.3)

        # 6) IV–VIX premium
        axes[5].clear()
        axes[5].plot(combined.index, combined['iv_vs_vix_premium'], lw=1.4)
        axes[5].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[5].set_title("IV – VIX Premium (Rich/Cheap)")
        axes[5].set_ylabel("Premium")
        axes[5].grid(alpha=0.3)

        # 7) IV → forward IV scatter
        axes[6].clear()
        axes[6].scatter(analysis_df['implied_vol'], analysis_df['forward_iv_30'], alpha=0.5, s=16)
        x_vals = np.linspace(analysis_df['implied_vol'].min(), analysis_df['implied_vol'].max(), 100)
        axes[6].plot(x_vals, reg_iv_fwd_iv.slope * x_vals + reg_iv_fwd_iv.intercept, color='red', lw=2,
                    label=f"Fit (R²={reg_iv_fwd_iv.rvalue ** 2:.2f})")
        axes[6].plot([analysis_df['implied_vol'].min(), analysis_df['implied_vol'].max()],
                    [analysis_df['implied_vol'].min(), analysis_df['implied_vol'].max()],
                    linestyle='--', color='gray', alpha=0.6)
        axes[6].set_title("Current IV vs Forward 30d IV")
        axes[6].set_xlabel("Current IV")
        axes[6].set_ylabel("Forward IV")
        axes[6].legend()
        axes[6].grid(alpha=0.3)

        # 8) Greeks profile
        axes[7].clear()
        if self.greeks_profile is not None:
            axes[7].plot(self.greeks_profile.index, self.greeks_profile['delta'], label='Delta')
            axes[7].plot(self.greeks_profile.index, self.greeks_profile['gamma'], label='Gamma')
            axes[7].plot(self.greeks_profile.index, self.greeks_profile['vega'], label='Vega (per 1%)')
            axes[7].plot(self.greeks_profile.index, self.greeks_profile['theta'], label='Theta (per day)')
            axes[7].set_title("Theoretical ATM Call Greeks (90-Day Horizon)")
            axes[7].set_xlabel("Days to Expiry")
            axes[7].legend()
            axes[7].grid(alpha=0.3)
        else:
            axes[7].text(0.5, 0.5, "Insufficient data for Greeks", ha='center', va='center')
            axes[7].set_axis_off()

        # 9) IV time series with forecast
        axes[8].clear()
        axes[8].plot(combined.index, combined['implied_vol'], label='IV', lw=1.6)
        if forecast is not None and forecast_dates is not None:
            axes[8].plot(forecast_dates, forecast, 'r--', lw=1.4, label='ARIMA Forecast')
        axes[8].set_title("Implied Volatility with ARIMA Projection")
        axes[8].set_xlabel("Date")
        axes[8].grid(alpha=0.3)
        axes[8].legend()

        self.fig.suptitle(
            f"Volatility & VIX Intelligence – {self.symbol_var.get().upper()}",
            fontsize=16, fontweight="bold"
        )
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw_idle()

    # -------------------- Metrics & Regime intelligence -------------------- #
    def update_metric_labels(self):
        """Update metric display labels with current values."""
        def format_pct(value: Optional[float]) -> str:
            if value is not None and np.isfinite(value):
                return f"{value:.4f} ({value * 100:.2f}%)"
            return "N/A"

        def colorize(
            label_widget: ttk.Label, value: Optional[float],
            high_threshold: float, low_threshold: float
        ):
            if value is None or not np.isfinite(value):
                label_widget.config(text="N/A", foreground="black")
            elif value > high_threshold:
                label_widget.config(text=format_pct(value), foreground="red")
            elif value < low_threshold:
                label_widget.config(text=format_pct(value), foreground="green")
            else:
                label_widget.config(text=format_pct(value), foreground="black")

        if self.current_price is not None:
            self.underlying_price_lbl.config(text=f"{self.current_price:.2f}")
        else:
            self.underlying_price_lbl.config(text="N/A")

        if self.current_vix is not None:
            self.live_vix_lbl.config(text=f"{self.current_vix:.2f}")
        else:
            self.live_vix_lbl.config(text="N/A")

        colorize(self.hist_iv_lbl, self.current_implied_vol, 0.4, 0.15)
        colorize(self.atm_iv_lbl, self.current_atm_iv, 0.45, 0.12)
        colorize(self.hv_lbl, self.current_realized_vol, 0.4, 0.12)

        if self.current_implied_vol and self.current_realized_vol and self.current_realized_vol > 0:
            ratio = self.current_implied_vol / self.current_realized_vol
            color = "red" if ratio > 1.5 else "green" if ratio < 0.8 else "black"
            self.iv_hv_ratio_lbl.config(text=f"{ratio:.2f}", foreground=color)
        else:
            self.iv_hv_ratio_lbl.config(text="N/A", foreground="black")

        if self.combined_data is not None:
            latest = self.combined_data.iloc[-1]
            if pd.notna(latest['iv_percentile']):
                self.iv_percentile_lbl.config(
                    text=f"{latest['iv_percentile'] * 100:.1f}%"
                )
            else:
                self.iv_percentile_lbl.config(text="N/A")

            if pd.notna(latest['iv_zscore']):
                self.iv_zscore_lbl.config(text=f"{latest['iv_zscore']:.2f}")
            else:
                self.iv_zscore_lbl.config(text="N/A")

            if pd.notna(latest['rolling_corr_252']):
                self.vix_corr_lbl.config(text=f"{latest['rolling_corr_252']:.2f}")
            else:
                self.vix_corr_lbl.config(text="N/A")

            if pd.notna(latest['iv_vs_vix_premium']):
                premium = latest['iv_vs_vix_premium']
                color = "red" if premium > 0 else "green" if premium < 0 else "black"
                self.iv_vix_premium_lbl.config(text=f"{premium:.4f}", foreground=color)
            else:
                self.iv_vix_premium_lbl.config(text="N/A", foreground="black")
        else:
            self.iv_percentile_lbl.config(text="N/A")
            self.iv_zscore_lbl.config(text="N/A")
            self.vix_corr_lbl.config(text="N/A")
            self.iv_vix_premium_lbl.config(text="N/A", foreground="black")

        self.update_regime_intelligence()

    def update_regime_intelligence(self):
        """Update regime intelligence display."""
        if self.combined_data is None:
            return

        latest = self.combined_data.iloc[-1]
        pct = latest['iv_percentile']
        premium = latest['iv_vs_vix_premium']
        corr = latest['rolling_corr_252']
        ratio = latest['iv_vix_ratio']

        if pd.isna(pct):
            self.regime_lbl.config(text="N/A", foreground="black")
            self.strategy_lbl.config(text="Load more data for regime insights.")
            return

        if pct >= 0.85:
            regime = "EXTREMELY HIGH VOL"
            color = "darkred"
            strategy = (
                "Favor defensive short-vol strategies (iron condors, ratio spreads) "
                "with dynamic hedges. Elevated VIX confirms stress – consider "
                "shorter tenors."
            )
        elif pct >= 0.70:
            regime = "ELEVATED VOLATILITY"
            color = "orangered"
            strategy = (
                "Bias toward net short volatility but hedge tail risk. "
                "Calendar spreads or covered calls shine when IV trades rich to VIX."
            )
        elif pct >= 0.40:
            regime = "NEUTRAL / FAIR VOL"
            color = "black"
            strategy = (
                "Blend delta plays with selective premium selling. "
                "Monitor IV-VIX premium: "
            )
        elif pct >= 0.20:
            regime = "SUPPRESSED VOL"
            color = "steelblue"
            strategy = (
                "Consider initiating long-vol structures (debit spreads, diagonals). "
                "Low VIX-tilt reduces option carry costs."
            )
        else:
            regime = "ULTRA-LOW VOL"
            color = "green"
            strategy = (
                "Aggressively explore long optionality (straddles/strangles). "
                "Cheap IV versus VIX can presage explosive moves."
            )

        if pd.notna(premium) and pd.notna(ratio):
            if premium > 0.05:
                strategy += " IV materially richer than VIX—size shorts with caution."
            elif premium < -0.05:
                strategy += " IV discounted relative to VIX—optionalities on sale."
        if pd.notna(corr):
            if corr > 0.7:
                strategy += (
                    " Correlation with VIX is strong: volatility moves likely "
                    "mirror macro shocks."
                )
            elif corr < 0.2:
                strategy += (
                    " Low IV/VIX correlation hints at idiosyncratic drivers; "
                    "lean into stock-specific catalysts."
                )

        self.regime_lbl.config(text=regime, foreground=color)
        self.strategy_lbl.config(text=strategy)

    # -------------------- Export -------------------- #
    def export_data(self, fmt: str):
        """Export data to file.

        Args:
            fmt: Export format ('csv' or 'json')
        """
        if self.combined_data is None:
            messagebox.showwarning("No data", "Load data before exporting.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}")],
            title=f"Export combined dataset as {fmt.upper()}"
        )
        if not filepath:
            return

        export_df = self.combined_data.copy()
        if self.analysis_data is not None:
            export_df = export_df.join(
                self.analysis_data[['forward_iv_30', 'forward_hv_30', 'forward_return_30']],
                how='left'
            )

        if self.greeks_profile is not None:
            greeks_export = self.greeks_profile.copy()
            greeks_export.index.name = "days"
            export_df = export_df.assign(
                greeks_delta=self.greeks_profile['delta'].reindex(export_df.index, method='ffill'),
                greeks_gamma=self.greeks_profile['gamma'].reindex(export_df.index, method='ffill'),
                greeks_vega=self.greeks_profile['vega'].reindex(export_df.index, method='ffill'),
                greeks_theta=self.greeks_profile['theta'].reindex(export_df.index, method='ffill'),
            )

        try:
            if fmt == "csv":
                export_df.to_csv(filepath, float_format="%.6f")
            else:
                export_df.to_json(filepath, orient="table", index=True)
            self.log(f"Dataset exported to {filepath}")
        except Exception as exc:
            self.log(f"Export failed: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the application."""
    root = tk.Tk()
    ImpliedVolatilityDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
