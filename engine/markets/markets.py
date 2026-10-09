"""Equity linkage, past-investment lookups and what-if scenario simulation.

Daily returns give a correlation structure and regression betas, and a shock to one name
propagates through them:  expected move in i = beta(i|j) * shock to j.  The R-squared of each
regression is carried alongside, because a beta from an unrelated pair is noise.

For what-ifs that are not about one stock (a war, a product launch, a company failing) the
local model only proposes which tickers are hit and how hard. That is a premise, not an
answer: the propagation itself is arithmetic on the measured betas.

Not investment advice. Prices come from yfinance (`pip install yfinance`), which is imported
only when data is fetched, so everything else works on plain dicts of returns.
"""

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BENCH = "SPY"
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "XOM", "LMT", "TSM", "PFE"]
SECTOR_PROXIES = {
    "technology": "XLK", "energy": "XLE", "financials": "XLF", "healthcare": "XLV",
    "industrials": "XLI", "defense": "ITA", "semiconductors": "SOXX", "consumer": "XLY",
    "staples": "XLP", "utilities": "XLU", "materials": "XLB", "real estate": "XLRE",
    "gold": "GLD", "oil": "USO", "treasuries": "TLT", "volatility": "^VIX",
}
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

MIN_R2 = 0.10  # below this a beta is noise dressed as a relationship
TRADING_DAYS = 252
MAX_SHOCK = 90.0

DISCLAIMER = (
    "Betas are measured from recent sessions and describe how these names moved together in "
    "normal conditions. Correlations rise sharply in a crisis, so treat spillover as a lower "
    "bound. Past performance is not a forecast. Not investment advice."
)


# --------------------------------------------------------------------------- prices


@dataclass
class PriceData:
    tickers: list
    dates: list  # datetime.date, shared by every series
    close: dict  # ticker -> list of closes aligned with `dates`
    returns: dict  # ticker -> list of log returns, one shorter than `dates`
    dropped: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.dates) - 1


def fetch_prices(tickers, period="1y", start=None, adjust=True, min_rows=20):
    """Download daily closes with yfinance and compute log returns.

    `start` (YYYY-MM-DD) overrides `period`. `adjust=True` folds dividends and splits into the
    price (total return); False gives the nominal price printed on the day. Tickers that mostly
    failed are dropped before rows are aligned, so one bad symbol cannot empty the table.
    """
    import yfinance as yf

    wanted = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not wanted:
        raise ValueError("no tickers given")
    span = {"start": start} if start else {"period": period}
    raw = yf.download(wanted, interval="1d", auto_adjust=adjust, progress=False,
                      group_by="column", threads=False, **span)
    if raw is None or len(raw) == 0:
        raise ValueError("no price data came back; check the tickers and your connection")

    frame = raw["Close"] if hasattr(raw.columns, "levels") else raw
    if getattr(frame, "ndim", 2) == 1:
        frame = frame.to_frame(name=wanted[0])
    dropped = [c for c in frame.columns if frame[c].notna().sum() < max(20, len(frame) // 4)]
    frame = frame.drop(columns=dropped).dropna(how="all").ffill().dropna()
    if frame.shape[1] == 0:
        raise ValueError(f"every ticker failed to download: {', '.join(wanted)}")
    if len(frame) < min_rows:
        raise ValueError(f"only {len(frame)} usable trading days; need at least {min_rows}")

    close = {c: [float(v) for v in frame[c]] for c in frame.columns}
    return price_data(close, [d.date() for d in frame.index], dropped)


def price_data(close, dates, dropped=()):
    """Build a PriceData from aligned closes; also the entry point for offline data."""
    returns = {
        t: [math.log(b / a) for a, b in zip(series, series[1:])] for t, series in close.items()
    }
    return PriceData(list(close), list(dates), close, returns, list(dropped))


# --------------------------------------------------------------------------- linkage


def correlation(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    var_a = sum((x - ma) ** 2 for x in a)
    var_b = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else float("nan")


def beta_matrix(returns):
    """Pairwise beta(i|j) (slope of i's returns on j's) and R-squared, as nested dicts [i][j]."""
    tickers = list(returns)
    betas = {i: {} for i in tickers}
    r2 = {i: {} for i in tickers}
    for i in tickers:
        for j in tickers:
            corr = correlation(returns[i], returns[j])
            std_i, std_j = statistics.stdev(returns[i]), statistics.stdev(returns[j])
            betas[i][j] = corr * std_i / std_j if std_j > 0 else float("nan")
            r2[i][j] = corr * corr
    return betas, r2


def propagate(shock, betas, r2, min_r2=MIN_R2):
    """Push a shock vector {ticker: pct} through the betas -> {ticker: (implied pct, best R2)}.

    Several shocked names driving one target are combined as an R2-weighted average, not summed:
    correlated sources carry largely the same information, so summing would double count.
    """
    out = {}
    for target in betas:
        if target in shock:
            continue
        num = den = best = 0.0
        for source, size in shock.items():
            if source not in betas[target]:
                continue
            b, w = betas[target][source], r2[target][source]
            if not (math.isfinite(b) and math.isfinite(w)) or w < min_r2:
                continue
            num += w * b * size
            den += w
            best = max(best, w)
        if den > 0:
            out[target] = (num / den, best)
    return out


def strongest_links(returns, limit=10):
    """The most correlated pairs as (a, b, correlation), strongest first."""
    tickers = list(returns)
    pairs = [
        (a, b, correlation(returns[a], returns[b]))
        for n, a in enumerate(tickers)
        for b in tickers[n + 1:]
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))[:limit]


def shock(data, source, pct):
    """Propagate an explicit move in one ticker through the measured structure."""
    source = source.upper()
    if source not in data.returns:
        raise ValueError(f"no price data for {source}")
    betas, r2 = beta_matrix(data.returns)
    return propagate({source: pct}, betas, r2)


def beta_to(data, bench=DEFAULT_BENCH):
    """Each ticker's (beta, R2, annualised volatility %) against the benchmark."""
    bench = bench.upper()
    if bench not in data.returns:
        raise ValueError(f"no data for the benchmark {bench}")
    betas, r2 = beta_matrix(data.returns)
    return {
        t: (betas[t][bench], r2[t][bench], statistics.stdev(data.returns[t]) * math.sqrt(TRADING_DAYS) * 100)
        for t in data.tickers if t != bench
    }


def quote(data):
    """Last price and 1-day, 5-day and whole-window moves (%) plus annualised volatility (%)."""
    rows = {}
    for t in data.tickers:
        c = data.close[t]
        rows[t] = {
            "last": c[-1],
            "1d": (c[-1] / c[-2] - 1) * 100,
            "5d": (c[-1] / c[-6] - 1) * 100 if len(c) > 6 else float("nan"),
            "window": (c[-1] / c[0] - 1) * 100,
            "vol": statistics.stdev(data.returns[t]) * math.sqrt(TRADING_DAYS) * 100,
        }
    return rows


# --------------------------------------------------------------------------- past investment


def hold(ticker, amount, start, bench=DEFAULT_BENCH):
    """What `amount` put into `ticker` on `start` (YYYY-MM-DD or YYYY-MM) would be worth now.

    Two pulls: dividend-adjusted for total return, nominal for the price actually printed on the
    day (and so the shares you could buy). Ignores tax, commission and spread.
    """
    ticker, bench = ticker.upper(), (bench or "").upper()
    if amount <= 0:
        raise ValueError("amount must be positive")
    if not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", start):
        raise ValueError("date must be YYYY-MM-DD (or YYYY-MM)")
    if len(start) == 7:
        start += "-01"

    symbols = [ticker] + ([bench] if bench and bench != ticker else [])
    total = fetch_prices(symbols, start=start, adjust=True, min_rows=2)
    nominal = fetch_prices([ticker], start=start, adjust=False, min_rows=2)
    if ticker not in total.close:
        raise ValueError(f"no price history for {ticker} from {start}")
    return hold_result(ticker, amount, total, nominal, bench)


def hold_result(ticker, amount, total, nominal, bench=None):
    adj, nom = total.close[ticker], nominal.close[ticker]
    years = max((total.dates[-1] - total.dates[0]).days / 365.25, 1e-9)
    tr_multiple, px_multiple = adj[-1] / adj[0], nom[-1] / nom[0]

    curve = [amount * p / adj[0] for p in adj]
    peak, worst, worst_at, peak_at, top_at = curve[0], 0.0, 0, 0, 0
    for i, value in enumerate(curve):
        if value > peak:
            peak, top_at = value, i
        drop = (value - peak) / peak  # a percentage, so a growing position does not skew it
        if drop < worst:
            worst, worst_at, peak_at = drop, i, top_at

    result = {
        "ticker": ticker, "amount": amount, "start": total.dates[0], "end": total.dates[-1],
        "years": years, "shares": amount / nom[0], "buy_price": nom[0], "last_price": nom[-1],
        "value_price_only": amount * px_multiple, "value_total_return": amount * tr_multiple,
        "cagr": tr_multiple ** (1 / years) - 1, "max_drawdown": worst,
        "drawdown_peak": total.dates[peak_at], "drawdown_trough": total.dates[worst_at],
        "curve": curve, "bench": None,
    }
    if bench and bench in total.close and bench != ticker:
        b = total.close[bench]
        multiple = b[-1] / b[0]
        result["bench"] = {
            "ticker": bench, "value": amount * multiple, "cagr": multiple ** (1 / years) - 1,
        }
    return result


# --------------------------------------------------------------------------- what-if


SCENARIO_SCHEMA = {
    "type": "object",
    "properties": {
        "shocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "move_pct": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["ticker", "move_pct", "reason"],
            },
        },
        "mechanism": {"type": "string"},
        "confidence": {"type": "string"},
        "second_order": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["shocks", "mechanism", "confidence", "second_order"],
}

SCENARIO_SYSTEM = (
    "You are an equity analyst translating a scenario into a first-order shock vector, answering "
    "with JSON only. Name only liquid, real, currently listed US tickers or well-known ADRs. "
    "move_pct is the immediate percentage move you would expect in that name on the news, "
    "positive or negative; keep it within plus or minus 40 unless the scenario is an outright "
    "bankruptcy. Pick the three to six names where the effect is most direct and defensible, not "
    "a long list of loosely related ones. Say honestly in `confidence` how speculative this is. "
    "You are supplying a premise for a calculation, not a prediction, and you must not give "
    "investment advice."
)


def load_model(model_path=MODEL_PATH, n_ctx=4096, **kwargs):
    """Load a GGUF instruct model with llama-cpp-python. Extra kwargs go to `Llama`."""
    from llama_cpp import Llama  # imported here so the rest works without it

    return Llama(model_path=str(model_path), n_ctx=n_ctx, verbose=False, **kwargs)


def parse_scenario(llm, scenario, universe):
    """Ask the model for a shock vector. Returns {"shocks": {ticker: (pct, reason)}, ...}."""
    body = (
        f"Scenario: {scenario}\n\nTickers already under consideration: {', '.join(universe)}\n\n"
        "Give the first-order shock vector."
    )
    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": SCENARIO_SYSTEM}, {"role": "user", "content": body}],
        response_format={"type": "json_object", "schema": SCENARIO_SCHEMA},
        temperature=0.0,
        max_tokens=900,
    )
    out = json.loads(response["choices"][0]["message"]["content"])

    shocks = {}
    for item in out.get("shocks") or []:
        ticker = str(item.get("ticker", "")).upper().strip()
        if not re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.\-]{0,9}", ticker):
            continue
        try:
            move = float(item.get("move_pct") or 0)
        except (TypeError, ValueError):
            continue
        shocks[ticker] = (max(-MAX_SHOCK, min(MAX_SHOCK, move)), str(item.get("reason", ""))[:90])
    if not shocks:
        raise ValueError("the model proposed no usable tickers for that scenario")
    return {
        "shocks": shocks,
        "mechanism": str(out.get("mechanism", "")),
        "confidence": str(out.get("confidence", "")),
        "second_order": [str(s) for s in (out.get("second_order") or [])][:6],
    }


def apply_scenario(proposal, data):
    """Propagate a `parse_scenario` proposal through the measured betas of `data`."""
    live = {t: pct for t, (pct, _) in proposal["shocks"].items() if t in data.returns}
    missing = [t for t in proposal["shocks"] if t not in data.returns]
    if not live:
        raise ValueError("none of the proposed tickers had usable price data")
    betas, r2 = beta_matrix(data.returns)
    return {"shocked": live, "missing": missing, "moves": propagate(live, betas, r2)}


def whatif(llm, scenario, tickers=None, period="1y", attempts=2):
    """Read a plain-English scenario with the model, fetch prices, and propagate its shocks."""
    universe = [t.upper() for t in tickers or DEFAULT_TICKERS]
    last_error = None
    for _ in range(attempts):
        try:
            proposal = parse_scenario(llm, scenario, universe)
            break
        except ValueError as error:  # includes json.JSONDecodeError from a truncated reply
            last_error = error
    else:
        raise ValueError(f"could not read that scenario: {last_error}")
    data = fetch_prices(sorted(set(universe) | set(proposal["shocks"])), period=period)
    return {"proposal": proposal, **apply_scenario(proposal, data), "sessions": data.n}


# --------------------------------------------------------------------------- formatting


def _table(header, rows):
    rows = [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    def line(row):
        return "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
    return "\n".join([line(header), "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


def format_propagation(shocked, moves, sessions, reasons=None):
    reasons = reasons or {}
    lines = ["Shock"]
    lines.append(_table(["ticker", "assumed move", "reason"],
                        [(t, f"{v:+.1f}%", reasons.get(t, "")) for t, v in shocked.items()]))
    lines += ["", f"Propagated through measured betas ({sessions} sessions)"]
    if not moves:
        lines.append(f"  nothing is linked strongly enough (R2 below {MIN_R2}); add related tickers")
    else:
        ordered = sorted(moves.items(), key=lambda kv: -abs(kv[1][0]))
        lines.append(_table(
            ["ticker", "implied move", "R2", "linkage"],
            [(t, f"{mv:+.2f}%", f"{q:.2f}", "weak" if q < 0.25 else "moderate" if q < 0.5 else "strong")
             for t, (mv, q) in ordered]))
    return "\n".join(lines + ["", DISCLAIMER])


def format_whatif(result):
    proposal = result["proposal"]
    reasons = {t: reason for t, (_, reason) in proposal["shocks"].items()}
    lines = ["The shock vector below is the model's judgement, not a measurement."]
    if proposal["mechanism"]:
        lines.append(proposal["mechanism"])
    if proposal["confidence"]:
        lines.append(f"Confidence: {proposal['confidence']}")
    if result["missing"]:
        lines.append(f"No price history for {', '.join(result['missing'])}; excluded.")
    lines += ["", format_propagation(result["shocked"], result["moves"], result["sessions"], reasons)]
    if proposal["second_order"]:
        lines += ["", "Second-order effects the model flags:"] + [f"  - {s}" for s in proposal["second_order"]]
    return "\n".join(lines)


def format_hold(r):
    lines = [
        f"{r['ticker']}: ${r['amount']:,.0f} invested on {r['start']} "
        f"({r['shares']:,.3f} shares at ${r['buy_price']:.2f}, now ${r['last_price']:.2f}, held {r['years']:.1f} years)",
        "",
        f"  price only:                ${r['value_price_only']:,.2f} ({(r['value_price_only'] / r['amount'] - 1) * 100:+.1f}%)",
        f"  with dividends reinvested: ${r['value_total_return']:,.2f} ({(r['value_total_return'] / r['amount'] - 1) * 100:+.1f}%)",
        f"  annualised (CAGR):         {r['cagr'] * 100:+.2f}%",
        f"  worst drawdown:            {r['max_drawdown'] * 100:.1f}% (peak {r['drawdown_peak']} to trough {r['drawdown_trough']})",
    ]
    if r["bench"]:
        b = r["bench"]
        diff = r["value_total_return"] - b["value"]
        lines.append(f"  {b['ticker']} over the same window: ${b['value']:,.2f} (CAGR {b['cagr'] * 100:+.2f}%); "
                     f"${abs(diff):,.2f} {'ahead of' if diff >= 0 else 'behind'} it")
    return "\n".join(lines + ["", DISCLAIMER])


def format_quote(rows):
    return _table(["ticker", "last", "1d", "5d", "window", "ann.vol"], [
        (t, f"{q['last']:.2f}", f"{q['1d']:+.2f}%",
         f"{q['5d']:+.1f}%" if math.isfinite(q["5d"]) else "-", f"{q['window']:+.1f}%", f"{q['vol']:.0f}%")
        for t, q in rows.items()])


def format_links(data):
    lines = [f"Strongest linkages ({data.n} sessions)", _table(
        ["pair", "correlation", "shared variance"],
        [(f"{a} - {b}", f"{c:+.2f}", f"{c * c:.0%}") for a, b, c in strongest_links(data.returns)])]
    return "\n".join(lines)


def format_beta(rows, bench):
    ordered = sorted(rows.items(), key=lambda kv: -kv[1][0])
    return _table(["ticker", f"beta vs {bench}", "R2", "ann.vol", ""], [
        (t, f"{b:.2f}", f"{q:.2f}", f"{v:.0f}%", "weak fit" if q < MIN_R2 else "") for t, (b, q, v) in ordered])


# --------------------------------------------------------------------------- interactive


def _ask(prompt, parse=str, default=None):
    while True:
        answer = input(prompt + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not answer and default is not None:
            return default
        try:
            return parse(answer)
        except ValueError as error:
            print(f"  {error}")


def _tickers(answer):
    found = answer.replace(",", " ").upper().split()
    if not found:
        raise ValueError("enter at least one ticker")
    return found


def _command(commands):
    def parse(answer):
        if answer not in commands:
            raise ValueError("pick one of the listed commands")
        return answer
    return parse


def main():
    watchlist = list(DEFAULT_TICKERS)
    llm = None
    commands = ["quote", "invest", "link", "beta", "shock", "whatif", "sectors", "watch", "quit"]
    while True:
        print()
        action = _ask(f"What now? {commands}", _command(commands))
        try:
            if action == "quit":
                return
            if action == "watch":
                chosen = _ask("Tickers", _tickers, default=" ".join(watchlist))
                watchlist = chosen.split() if isinstance(chosen, str) else chosen
                print("Watchlist:", ", ".join(watchlist))
            elif action == "quote":
                print(format_quote(quote(fetch_prices(watchlist, "3mo"))))
            elif action == "sectors":
                data = fetch_prices(list(SECTOR_PROXIES.values()), "6mo")
                names = {v: k for k, v in SECTOR_PROXIES.items()}
                print(_table(["sector", "proxy", "change", "ann.vol"], [
                    (names.get(t, t), t, f"{q['window']:+.1f}%", f"{q['vol']:.0f}%")
                    for t, q in sorted(quote(data).items(), key=lambda kv: -kv[1]["window"])]))
            elif action == "link":
                print(format_links(fetch_prices(watchlist)))
            elif action == "beta":
                bench = _ask("Benchmark", str.upper, default=DEFAULT_BENCH)
                print(format_beta(beta_to(fetch_prices(list(dict.fromkeys(watchlist + [bench]))), bench), bench))
            elif action == "invest":
                ticker = _ask("Ticker", str.upper)
                amount = _ask("Amount in dollars", lambda a: float(a.lstrip("$").replace(",", "")))
                start = _ask("Start date YYYY-MM-DD")
                print(format_hold(hold(ticker, amount, start)))
            elif action == "shock":
                source = _ask("Ticker that moves", str.upper)
                pct = _ask("Move in percent, e.g. -15", lambda a: float(a.rstrip("%")))
                data = fetch_prices(list(dict.fromkeys([source] + watchlist)))
                print(format_propagation({source: pct}, shock(data, source, pct), data.n))
            elif action == "whatif":
                scenario = _ask("Scenario in plain English")
                llm = llm or load_model()
                print(format_whatif(whatif(llm, scenario, watchlist)))
        except (ValueError, ImportError) as error:
            print(f"  {error}")


if __name__ == "__main__":
    main()
