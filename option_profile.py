"""P/L profile of a multi-leg option position (PyQt5).

Black-Scholes with r = 4.2%. Every leg is priced at its own implied vol,
solved from the entry premium at the spot, so the "now" curve passes
through zero at the spot; the IV slider scales it. Stock legs count
shares (delta 1 per share, no other greeks).

    python option_profile.py
    python option_profile.py --portfolio portfolios/msft-collar.json
"""

import bisect
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta

from PyQt5.QtCore import QDate, QLocale, QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDateEdit, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QTableWidget, QVBoxLayout,
    QWidget,
)

CONTRACT_SIZE = 100.0      # shares per option contract

# Legs table columns. Bid/Ask are quoted market data; IV* and the greek
# columns are model output.
COL_ENTRY = 6
COL_BID = 7
COL_ASK = 8
COL_IV = 9
COL_DELTA = 10
COL_GAMMA = 11
COL_THETA = 12
COL_VEGA = 13

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass
class Leg:
    """side=+1 bought, side=-1 sold; iv None means the fallback vol.

    The fields after iv are the quoted snapshot, kept as it was received.
    They never feed the model: the curve and the greeks are recomputed
    from entry / model IV, and the market numbers are only shown.
    """
    ticker: str
    kind: str          # "C", "P" or "S" (stock)
    strike: float
    expiry: date
    qty: int           # contracts, or shares for stock
    side: int
    entry: float       # premium or price per share
    iv: float = None
    bid: float = None
    ask: float = None
    last: float = None
    market_iv: float = None
    volume: int = None
    oi: int = None
    market_delta: float = None


def portfolio_from_data(data):
    """Parse the JSON portfolio format into (meta, legs).

    Option prices are per share; "iv" and "delta" come in percent and in
    plain units. A stock leg has no strike and no expiry.
    """
    legs = []
    for item in data.get("legs", []):
        kind = str(item.get("kind", "C")).upper()
        if kind == "S":
            expiry = date(1900, 1, 1)
            strike = 0.0
        else:
            expiry = date.fromisoformat(item["expiry"])
            strike = float(item.get("strike", 0.0))
        iv_pct = item.get("iv")
        volume = item.get("volume")
        oi = item.get("oi")
        delta = item.get("delta")
        legs.append(Leg(
            ticker=str(item.get("ticker", "")).upper(),
            kind=kind,
            strike=strike,
            expiry=expiry,
            qty=int(item.get("qty", 1)),
            side=1 if float(item.get("side", 1)) > 0 else -1,
            entry=float(item["entry"]),
            bid=float(item["bid"]) if "bid" in item else None,
            ask=float(item["ask"]) if "ask" in item else None,
            last=float(item["last"]) if "last" in item else None,
            market_iv=float(iv_pct) / 100.0 if iv_pct is not None else None,
            volume=int(volume) if volume is not None else None,
            oi=int(oi) if oi is not None else None,
            market_delta=float(delta) if delta is not None else None,
        ))

    meta = {
        "name": data.get("name", ""),
        "valuation_date": date.fromisoformat(data["valuation_date"]),
        "spot": float(data["spot"]),
        "source": data.get("source", ""),
        "view": data.get("view") or {},
    }
    return meta, legs


def read_portfolio(path):
    """(meta, legs) from a saved portfolio file."""
    # utf-8-sig: a BOM left by Windows editors or PowerShell is harmless.
    with open(path, "r", encoding="utf-8-sig") as handle:
        return portfolio_from_data(json.load(handle))


def portfolio_to_data(name, valuation_date, spot, source, legs, view=None):
    """The JSON portfolio format; the inverse of portfolio_from_data."""
    items = []
    for leg in legs:
        item = {
            "ticker": leg.ticker,
            "kind": leg.kind,
            "qty": leg.qty,
            "side": leg.side,
            "entry": round(leg.entry, 4),
        }
        if leg.kind != "S":
            item["strike"] = leg.strike
            item["expiry"] = leg.expiry.isoformat()
        for key, value in (("bid", leg.bid), ("ask", leg.ask),
                           ("last", leg.last), ("volume", leg.volume),
                           ("oi", leg.oi)):
            if value is not None:
                item[key] = value
        if leg.market_iv is not None:
            item["iv"] = round(leg.market_iv * 100.0, 2)
        if leg.market_delta is not None:
            item["delta"] = round(leg.market_delta, 4)
        items.append(item)

    return {
        "name": name,
        "valuation_date": valuation_date.isoformat(),
        "spot": spot,
        "source": source,
        "currency": "USD",
        "contract_size": int(CONTRACT_SIZE),
        "legs": items,
        "view": view or {},
    }


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# The rate from the options DB (rfr_history).
RISK_FREE = 0.042


def black_scholes(kind, spot, strike, years, sigma, rate=None):
    """Option price by Black-Scholes."""
    if rate is None:
        rate = RISK_FREE
    if years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        if kind == "C":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    growth = math.exp(-rate * years)
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike)
          + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t

    if kind == "C":
        return spot * _norm_cdf(d1) - strike * growth * _norm_cdf(d2)
    return strike * growth * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(kind, spot, strike, years, price, rate=None):
    """Volatility matching the premium (bisection); None below intrinsic."""
    if rate is None:
        rate = RISK_FREE
    if years <= 0 or spot <= 0 or strike <= 0 or price <= 0:
        return None
    intrinsic = max(0.0, spot - strike) if kind == "C" else max(0.0, strike - spot)
    if price <= intrinsic + 1e-9:
        return None

    low, high = 1e-4, 5.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if black_scholes(kind, spot, strike, years, middle, rate=rate) < price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def black_scholes_greeks(kind, spot, strike, years, sigma, rate=None):
    """(delta, gamma, theta per day, vega per vol point); None if dead."""
    if rate is None:
        rate = RISK_FREE
    if years <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return None

    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike)
          + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    growth = math.exp(-rate * years)

    if kind == "C":
        delta = _norm_cdf(d1)
        theta_year = (-spot * density * sigma / (2.0 * root_t)
                      - rate * strike * growth * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_year = (-spot * density * sigma / (2.0 * root_t)
                      + rate * strike * growth * _norm_cdf(-d2))

    gamma = density / (spot * sigma * root_t)
    vega = spot * density * root_t / 100.0
    return delta, gamma, theta_year / 365.0, vega


def leg_value(leg, spot, on_date):
    """Stock is worth the spot; an option drops to intrinsic after expiry."""
    if leg.kind == "S":
        return spot
    if on_date >= leg.expiry:
        if leg.kind == "C":
            return max(0.0, spot - leg.strike)
        return max(0.0, leg.strike - spot)

    years = (leg.expiry - on_date).days / 365.0
    sigma = leg.iv if leg.iv else SIGMA["value"]
    return black_scholes(leg.kind, spot, leg.strike, years, sigma)


# Fallback vol when the implied vol cannot be solved.
SIGMA = {"value": 0.25}


def position_pl(legs, spot, on_date):
    """P/L of the whole position in dollars."""
    total = 0.0
    for leg in legs:
        if leg.kind == "S":
            # Stock quantity is shares: no contract multiplier.
            total += leg.side * leg.qty * (spot - leg.entry)
            continue
        value = leg_value(leg, spot, on_date)
        total += leg.side * leg.qty * CONTRACT_SIZE * (value - leg.entry)
    return total


def money(value):
    """$1,185 / -$500."""
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def price_label(value):
    return f"${value:,.2f}"


def _interp(xs, ys, target):
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]
    index = bisect.bisect_left(xs, target)
    x0, x1 = xs[index - 1], xs[index]
    y0, y1 = ys[index - 1], ys[index]
    if x1 == x0:
        return y1
    share = (target - x0) / (x1 - x0)
    return y0 + share * (y1 - y0)


def _ordinal(day):
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def date_label(value, today):
    days = (value - today).days
    if days == 0:
        suffix = "Now"
    else:
        suffix = f"{days}d"
    return (f"{WEEKDAYS[value.weekday()]} {MONTHS[value.month - 1]} "
            f"{value.day}{_ordinal(value.day)} ({suffix})")


def _nice_step(span, target_ticks=6):
    """Round grid step for the requested number of ticks."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


class ProfitChart(QWidget):
    MARGIN_LEFT = 74
    MARGIN_TOP = 28
    MARGIN_RIGHT = 22
    MARGIN_BOTTOM = 42

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series = []
        self._spot = None
        self._center = None
        self._breakevens = []
        self._cursor_x = None
        self._x_min = 0.0
        self._x_max = 1.0
        self.setMinimumHeight(380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        if x != self._cursor_x:
            self._cursor_x = x
            self.update()

    def leaveEvent(self, _event):
        if self._cursor_x is not None:
            self._cursor_x = None
            self.update()

    def set_cursor_price(self, price):
        width = self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT
        if width <= 0 or not (self._x_min <= price <= self._x_max):
            self._cursor_x = None
        else:
            self._cursor_x = (self.MARGIN_LEFT + (price - self._x_min)
                              / (self._x_max - self._x_min) * width)
        self.update()

    def set_data(self, series, spot, x_min, x_max, center=None, breakevens=None):
        self._series = series
        self._spot = spot
        self._center = center
        self._breakevens = breakevens or []
        self._x_min = x_min
        self._x_max = x_max
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#0b0f16"))

        plot = QRectF(
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM,
        )

        if not self._series:
            painter.setPen(QColor("#8b93a3"))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Add at least one leg to see the profile")
            return

        values = [y for series in self._series for y in series["ys"]]
        y_lo, y_hi = min(values), max(values)
        padding = (y_hi - y_lo) * 0.10 or 500.0
        y_lo -= padding
        y_hi += padding
        y_lo = min(y_lo, 0.0)
        y_hi = max(y_hi, 0.0)

        def map_x(value):
            return plot.left() + (value - self._x_min) / (self._x_max - self._x_min) * plot.width()

        def map_y(value):
            return plot.bottom() - (value - y_lo) / (y_hi - y_lo) * plot.height()

        painter.setClipRect(plot)
        self._draw_grid(painter, plot, y_lo, y_hi, map_x, map_y)
        self._draw_fills(painter, plot, map_x, map_y)
        self._draw_curves(painter, plot, map_x, map_y)
        self._draw_center(painter, plot, map_x)
        self._draw_breakevens(painter, plot, map_x)
        self._draw_spot(painter, plot, map_x)
        self._draw_crosshair(painter, plot, map_x, map_y)
        painter.setClipping(False)

        self._draw_axes_labels(painter, plot, y_lo, y_hi, map_x, map_y)
        self._draw_legend(painter, plot)

    def _draw_grid(self, painter, plot, y_lo, y_hi, map_x, map_y):
        grid_pen = QPen(QColor("#1b2230"), 1)
        painter.setPen(grid_pen)

        step_y = _nice_step(y_hi - y_lo, 7)
        tick = math.ceil(y_lo / step_y) * step_y
        while tick <= y_hi:
            y = map_y(tick)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            tick += step_y

        step_x = _nice_step(self._x_max - self._x_min, 28)
        tick = math.ceil(self._x_min / step_x) * step_x
        while tick <= self._x_max:
            x = map_x(tick)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            tick += step_x

        painter.setPen(QPen(QColor("#c9d1dd"), 1.2))
        zero_y = map_y(0.0)
        painter.drawLine(QPointF(plot.left(), zero_y), QPointF(plot.right(), zero_y))

        painter.setPen(QPen(QColor("#1a2130"), 1))
        painter.drawRect(plot)

    def _solid_series(self):
        return [series for series in self._series if series["style"] == "solid"]

    def _draw_fills(self, painter, plot, map_x, map_y):
        zero_y = map_y(0.0)
        painter.save()
        painter.setClipRect(plot)
        for series in self._solid_series():
            for points, sign in self._sign_runs(series["xs"], series["ys"]):
                if len(points) < 2:
                    continue
                path = QPainterPath(QPointF(map_x(points[0][0]), zero_y))
                for x, y in points:
                    path.lineTo(QPointF(map_x(x), map_y(y)))
                path.lineTo(QPointF(map_x(points[-1][0]), zero_y))
                path.closeSubpath()

                if sign > 0:
                    edge = max(plot.top(),
                               min(map_y(y) for _x, y in points))
                    if edge >= zero_y:
                        continue
                    gradient = QLinearGradient(0.0, edge, 0.0, zero_y)
                    gradient.setColorAt(0.0, QColor(35, 194, 107, 64))
                    gradient.setColorAt(1.0, QColor(35, 194, 107, 4))
                else:
                    edge = min(plot.bottom(),
                               max(map_y(y) for _x, y in points))
                    if edge <= zero_y:
                        continue
                    gradient = QLinearGradient(0.0, zero_y, 0.0, edge)
                    gradient.setColorAt(0.0, QColor(255, 85, 102, 4))
                    gradient.setColorAt(1.0, QColor(255, 85, 102, 58))
                painter.fillPath(path, QBrush(gradient))
        painter.restore()

    @staticmethod
    def _sign_runs(xs, ys):
        """(points, sign) runs of constant P/L sign, split at zero crossings."""
        runs = []
        points = [(xs[0], ys[0])]
        current = 1 if ys[0] >= 0 else -1
        for i in range(1, len(xs)):
            x0, y0 = xs[i - 1], ys[i - 1]
            x1, y1 = xs[i], ys[i]
            if (y0 >= 0) != (y1 >= 0) and y0 != y1:
                share = y0 / (y0 - y1)
                crossing = (x0 + share * (x1 - x0), 0.0)
                runs.append((points + [crossing], current))
                points = [crossing]
                current = 1 if y1 >= 0 else -1
            points.append((xs[i], ys[i]))
        runs.append((points, current))
        return runs

    def _draw_curves(self, painter, plot, map_x, map_y):
        order = {"dot": 0, "dash": 1, "solid": 2}
        for series in sorted(self._series, key=lambda s: order[s["style"]]):
            xs, ys = series["xs"], series["ys"]

            if series["style"] == "solid":
                # Green above zero, red below.
                for i in range(len(xs) - 1):
                    middle = (ys[i] + ys[i + 1]) / 2.0
                    color = QColor("#23c26b") if middle >= 0 else QColor("#ff5566")
                    painter.setPen(QPen(color, 2.4))
                    painter.drawLine(QPointF(map_x(xs[i]), map_y(ys[i])),
                                     QPointF(map_x(xs[i + 1]), map_y(ys[i + 1])))
                continue

            pen = QPen(QColor(series["color"]), 1.7)
            pen.setStyle(Qt.DashLine if series["style"] == "dash" else Qt.DotLine)
            painter.setPen(pen)

            path = QPainterPath()
            path.moveTo(map_x(xs[0]), map_y(ys[0]))
            for x, y in zip(xs[1:], ys[1:]):
                path.lineTo(map_x(x), map_y(y))
            painter.drawPath(path)

    def _draw_center(self, painter, plot, map_x):
        if self._center is None:
            return
        x = map_x(self._center)
        pen = QPen(QColor("#8b93a3"), 1)
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

    def _draw_breakevens(self, painter, plot, map_x):
        painter.setFont(QFont("Segoe UI", 8))
        for value in self._breakevens:
            if not (self._x_min <= value <= self._x_max):
                continue
            x = map_x(value)
            painter.setPen(QPen(QColor("#3d7bfd"), 1.2))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor("#6ea0ff"))
            painter.drawText(QRectF(x - 60, plot.top() + 2, 120, 16),
                             Qt.AlignCenter, price_label(value))

    def _draw_crosshair(self, painter, plot, map_x, map_y):
        if self._cursor_x is None or not self._series:
            return
        if not (plot.left() <= self._cursor_x <= plot.right()):
            return

        price = self._x_min + ((self._cursor_x - plot.left())
                               / plot.width() * (self._x_max - self._x_min))

        painter.setPen(QPen(QColor("#e6e9ef"), 1))
        painter.drawLine(QPointF(self._cursor_x, plot.top()),
                         QPointF(self._cursor_x, plot.bottom()))

        base = self._center if self._center else price
        percent = (price / base - 1.0) * 100.0 if base else 0.0
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(self._cursor_x - 85, plot.top() + 2, 170, 18),
                         Qt.AlignCenter, f"{price_label(price)} ({percent:+.0f}%)")

        # A value on every curve, labels stacked so they never overlap.
        marks = []
        for series in self._series:
            value = _interp(series["xs"], series["ys"], price)
            marks.append({
                "true_y": map_y(value),
                "label_y": map_y(value),
                "text": money(value),
                "style": series["style"],
                "color": series["color"],
            })
        if not marks:
            return

        marks.sort(key=lambda mark: mark["true_y"])
        gap = 18.0
        for index in range(1, len(marks)):
            limits = marks[index - 1]["label_y"] + gap
            if marks[index]["label_y"] < limits:
                marks[index]["label_y"] = limits
        overflow = marks[-1]["label_y"] - (plot.bottom() - 10.0)
        if overflow > 0:
            for mark in marks:
                mark["label_y"] -= overflow

        for mark in marks:
            solid = mark["style"] == "solid"
            painter.setPen(QPen(QColor("#0b0f16"), 2))
            painter.setBrush(QColor("#ffffff") if solid else QColor(mark["color"]))
            radius = 4.5 if solid else 3.5
            painter.drawEllipse(QPointF(self._cursor_x, mark["true_y"]),
                                radius, radius)
            painter.setBrush(Qt.NoBrush)

        for mark in marks:
            solid = mark["style"] == "solid"
            font = QFont("Segoe UI", 10 if solid else 9)
            font.setBold(solid)
            painter.setFont(font)
            painter.setPen(QColor("#ffffff") if solid else QColor(mark["color"]))
            y = mark["label_y"]
            if self._cursor_x + 130 < plot.right():
                text_rect = QRectF(self._cursor_x + 11, y - 10, 115, 20)
                align = Qt.AlignLeft | Qt.AlignVCenter
            else:
                text_rect = QRectF(self._cursor_x - 126, y - 10, 115, 20)
                align = Qt.AlignRight | Qt.AlignVCenter
            painter.drawText(text_rect, align, mark["text"])

    def _draw_spot(self, painter, plot, map_x):
        if not self._spot or not (self._x_min <= self._spot <= self._x_max):
            return
        x = map_x(self._spot)
        painter.setPen(QPen(QColor("#3d7bfd"), 1.2))
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        # The spot label sits in the top margin, outside the clip region.
        painter.save()
        painter.setClipping(False)
        painter.setPen(QColor("#6ea0ff"))
        painter.drawText(QRectF(x - 55, plot.top() - 20, 110, 16),
                         Qt.AlignCenter, price_label(self._spot))
        painter.restore()

    def _draw_axes_labels(self, painter, plot, y_lo, y_hi, map_x, map_y):
        painter.setPen(QColor("#8b93a3"))
        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        step_y = _nice_step(y_hi - y_lo, 7)
        tick = math.ceil(y_lo / step_y) * step_y
        while tick <= y_hi:
            text = money(tick)
            painter.drawText(QRectF(0, map_y(tick) - 8, self.MARGIN_LEFT - 10, 16),
                             Qt.AlignRight | Qt.AlignVCenter, text)
            tick += step_y

        step_x = _nice_step(self._x_max - self._x_min, 28)
        tick = math.ceil(self._x_min / step_x) * step_x
        while tick <= self._x_max:
            painter.drawText(QRectF(map_x(tick) - 30, plot.bottom() + 8, 60, 16),
                             Qt.AlignCenter, f"${tick:,.0f}")
            tick += step_x

    def _draw_legend(self, painter, plot):
        color_by_style = {"solid": "#dfe3ea", "dash": "#ff8f8f", "dot": "#b08a8a"}
        painter.setFont(QFont("Segoe UI", 8))
        for i, series in enumerate(self._series):
            y = plot.top() + 12 + i * 18
            pen = QPen(QColor(color_by_style[series["style"]]), 1.8)
            if series["style"] == "dash":
                pen.setStyle(Qt.DashLine)
            elif series["style"] == "dot":
                pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(plot.left() + 12, y),
                             QPointF(plot.left() + 40, y))
            painter.setPen(QColor("#9aa3b5"))
            painter.drawText(QPointF(plot.left() + 48, y + 4), series["label"])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Option Position P/L Profile")
        self.resize(1440, 880)

        # The date the position is valued at; a loaded portfolio may carry
        # its own historical valuation date.
        self.valuation_date = date.today()

        # None: middle curve at the midpoint to the nearest expiry, else
        # that many days before it.
        self.mid_curve_days_before = None

        # Center the X axis on the spot instead of the strike midpoint.
        self.center_on_spot = False

        # Caption of a loaded portfolio: name, quote source and the date
        # it was valued at.
        self.portfolio_name = ""
        self.source_text = ""

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        layout.addLayout(self._build_metrics_row())

        self.chart = ProfitChart()
        layout.addWidget(self.chart, stretch=1)

        layout.addWidget(self._build_slider_panel())

        table_row = QHBoxLayout()
        table_row.addWidget(self._build_table_panel(), stretch=1)
        layout.addLayout(table_row)

        self.setCentralWidget(central)

        self._add_default_legs()
        self._recompute()
        # 28 days out: the 24-Dec-2026 expiry of the default legs.
        self.days_slider.setValue(28)

    def _build_metrics_row(self):
        row = QHBoxLayout()
        row.setSpacing(26)

        self.badge = QLabel("AAPL ▲")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedHeight(26)
        self.badge.setMinimumWidth(86)
        self.badge.setStyleSheet(
            "background:#26a269; color:#08130c; border-radius:4px;"
            "font-weight:700; font-size:12px;")
        row.addWidget(self.badge)

        self.metric_debit_title, self.metric_debit = self._metric(
            row, "NET DEBIT", "#dfe6f2")
        _, self.metric_loss = self._metric(row, "MAX LOSS", "#ff5566")
        _, self.metric_profit = self._metric(row, "MAX PROFIT", "#23c26b")
        _, self.metric_chance = self._metric(row, "CHANCE OF PROFIT", "#8b93a3")
        self.metric_break_title, self.metric_break = self._metric(
            row, "BREAKEVEN", "#dfe6f2")
        self.metric_chance.setText("—% 🔒")

        row.addStretch(1)
        return row

    @staticmethod
    def _metric(row, caption, color):
        box = QVBoxLayout()
        box.setSpacing(1)
        title = QLabel(caption)
        title.setStyleSheet("color:#8b93a3; font-size:10px; letter-spacing:1px;")
        value = QLabel("—")
        value.setStyleSheet(f"color:{color}; font-size:13px; font-weight:600;")
        box.addWidget(title)
        box.addWidget(value)
        row.addLayout(box)
        return title, value

    def _build_slider_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background:#0e131c; border-radius:6px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        days_row = QHBoxLayout()
        self.days_caption = QLabel("DATE:")
        self.days_caption.setStyleSheet("color:#dfe6f2; font-weight:600;")
        self.days_slider = QSlider(Qt.Horizontal)
        self.days_slider.setRange(0, 60)
        self.days_slider.valueChanged.connect(self._recompute)
        self.days_remaining = QLabel("")
        self.days_remaining.setStyleSheet("color:#8b93a3;")
        days_row.addWidget(self.days_caption)
        days_row.addWidget(self.days_slider, stretch=1)
        days_row.addWidget(self.days_remaining)
        layout.addLayout(days_row)

        self.range_caption = QLabel("RANGE: ±22%")
        self.range_caption.setStyleSheet("color:#dfe6f2; font-weight:600;")
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setRange(1, 100)
        self.range_slider.setValue(22)
        self.range_slider.valueChanged.connect(self._recompute)
        range_box = QHBoxLayout()
        range_box.addWidget(self.range_caption)
        range_box.addWidget(self.range_slider, stretch=1)

        # Multiplier over every leg's base vol (100 = x1).
        self.iv_caption = QLabel("IMPLIED VOLATILITY: x1.00")
        self.iv_caption.setStyleSheet("color:#dfe6f2; font-weight:600;")
        self.iv_slider = QSlider(Qt.Horizontal)
        self.iv_slider.setRange(10, 300)
        self.iv_slider.setValue(100)
        self.iv_slider.valueChanged.connect(self._recompute)
        self.scale_ticks = QLabel("x1          x2          x3")
        self.scale_ticks.setStyleSheet("color:#4a5468; font-size:9px;")
        iv_box = QHBoxLayout()
        iv_box.addWidget(self.iv_caption)
        iv_box.addWidget(self.iv_slider, stretch=1)
        iv_box.addWidget(self.scale_ticks)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(24)
        controls_row.addLayout(range_box, stretch=1)
        controls_row.addLayout(iv_box, stretch=1)
        layout.addLayout(controls_row)

        return panel

    def _build_table_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        label_spot = QLabel("Spot:")
        label_spot.setStyleSheet("color:#8b93a3;")
        self.spot_spin = QDoubleSpinBox()
        self.spot_spin.setRange(0.01, 1_000_000.0)
        self.spot_spin.setDecimals(2)
        self.spot_spin.setSingleStep(1.0)
        self.spot_spin.setValue(355.58)
        self.spot_spin.setPrefix("$")
        self.spot_spin.valueChanged.connect(self._recompute)

        self.update_button = QPushButton("Update")
        self.update_button.setToolTip(
            "Refresh the spot and every leg's IV from the live Bid/Ask "
            "(Yahoo Finance via yfinance); IV is solved from the midline.")
        self.update_button.clicked.connect(self._on_update_quotes)

        invert_button = QPushButton("Inversion")
        invert_button.setToolTip("Swap Long and Short on every leg.")
        invert_button.clicked.connect(self._on_invert_sides)

        self.quote_status = QLabel("")
        self.quote_status.setStyleSheet("color:#8b93a3; font-size:10px;")
        # A long caption must not widen the window: let it shrink and clip.
        self.quote_status.setSizePolicy(QSizePolicy.Ignored,
                                        QSizePolicy.Preferred)

        add_button = QPushButton("+ Leg")
        add_button.clicked.connect(self._on_add_leg)
        remove_button = QPushButton("− Leg")
        remove_button.clicked.connect(self._on_remove_leg)

        save_button = QPushButton("Save")
        save_button.setToolTip("Save the position as a portfolio file (JSON).")
        save_button.clicked.connect(self._on_save_portfolio)
        load_button = QPushButton("Load")
        load_button.setToolTip(
            "Open a portfolio file: legs, spot and the valuation date.")
        load_button.clicked.connect(self._on_load_portfolio)

        controls.addWidget(label_spot)
        controls.addWidget(self.spot_spin)
        controls.addSpacing(10)
        controls.addWidget(self.quote_status)
        controls.addStretch(1)
        controls.addWidget(self.update_button)
        controls.addWidget(invert_button)
        controls.addWidget(add_button)
        controls.addWidget(remove_button)
        controls.addWidget(save_button)
        controls.addWidget(load_button)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 14)
        self.table.setHorizontalHeaderLabels(
            ["Ticker", "Type", "Strike", "Expiration",
             "Qty", "Position", "Entry", "Bid", "Ask", "IV*",
             "Δ", "Γ", "Θ", "ν"])
        model_iv_header = self.table.horizontalHeaderItem(COL_IV)
        model_iv_header.setToolTip(
            "IV solved from the entry premium (the mid): model value.")
        # A greek header shows the position total for the full quantity.
        greek_tips = {
            COL_DELTA: "Delta per share; total is the position delta, shares.",
            COL_GAMMA: "Gamma per share; total is Δ per $1, full quantity.",
            COL_THETA: "Theta per share per day; total is $ per day.",
            COL_VEGA: "Vega per share per vol point; total is $ per vol pt.",
        }
        for column, tip in greek_tips.items():
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        # Fixed widths keep the columns inside the window: left to their
        # size hint, the cell widgets push everything past 1900 px.
        widths = {0: 58, 1: 80, 2: 84, 3: 92, 4: 58, 5: 98, 6: 84,
                  7: 72, 8: 72, 9: 76}
        for column, width in widths.items():
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.Interactive)
            self.table.setColumnWidth(column, width)
        # The greek columns take what is left and split it evenly.
        for column in (COL_DELTA, COL_GAMMA, COL_THETA, COL_VEGA):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMinimumWidth(1000)
        self.table.setFixedHeight(172)
        layout.addWidget(self.table)

        # Position-level greeks, in dollars: model output.
        self.position_greeks = QLabel("")
        self.position_greeks.setStyleSheet(
            "color:#c9d1dd; font-size:11px; font-weight:600;")
        layout.addWidget(self.position_greeks)

        # Which columns come from the quote and which from the model.
        self.legend = QLabel(
            '<span style="color:#8b93a3;">MARKET &nbsp;Bid · Ask · Last · '
            'OI · Vol · Mkt IV · Δ quoted</span> &nbsp;&nbsp;'
            '<span style="color:#c9d1dd;">MODEL &nbsp;IV* from the mid · '
            'Δ Γ Θ ν by Black-Scholes, r = 4.2%</span>')
        self.legend.setStyleSheet("font-size:10px;")
        layout.addWidget(self.legend)

        return panel

    def _on_add_leg(self):
        self._add_leg_row(Leg("AAPL", "C", 355.0,
                              self.valuation_date + timedelta(days=28),
                              1, +1, 4.20))
        self._recompute()

    def _on_remove_leg(self):
        row = self.table.currentRow()
        if row >= 0 and self.table.rowCount() > 1:
            self.table.removeRow(row)
            self._recompute()

    def _on_invert_sides(self):
        """Flip every leg Long <-> Short."""
        for row in range(self.table.rowCount()):
            side = self.table.cellWidget(row, 5)
            side.setCurrentIndex(1 - side.currentIndex())
        self._recompute()

    def _on_update_quotes(self):
        """Refresh the spot and every leg's IV from live Yahoo quotes."""
        try:
            import yfinance as yf
        except ImportError:
            self.quote_status.setText("yfinance is not installed")
            return

        legs = self._read_legs()
        if not legs:
            self.quote_status.setText("no legs to update")
            return

        # The Yahoo calls block the event loop for a few seconds: say so
        # before they start, not after.
        self.update_button.setText("Getting data…")
        self.update_button.setEnabled(False)
        self.quote_status.setText("getting quotes from Yahoo…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            spots = {}
            for ticker in {leg.ticker for leg in legs}:
                spots[ticker] = self._fetch_last_price(yf, ticker)

            chains = {}
            for leg in legs:
                if leg.kind == "S":
                    continue
                key = (leg.ticker, leg.expiry)
                if key in chains:
                    continue
                try:
                    chains[key] = yf.Ticker(leg.ticker).option_chain(
                        leg.expiry.isoformat())
                except Exception:
                    chains[key] = None

            stock_quotes = {}
            for ticker in {leg.ticker for leg in legs if leg.kind == "S"}:
                stock_quotes[ticker] = self._stock_quote(
                    yf, ticker, spots.get(ticker), chains)

            updated = 0
            skipped = 0
            for row, leg in enumerate(legs):
                if leg.kind == "S":
                    quote = stock_quotes.get(leg.ticker)
                else:
                    quote = self._market_quote(
                        leg, chains.get((leg.ticker, leg.expiry)),
                        spots.get(leg.ticker))
                if quote is None:
                    skipped += 1
                    continue

                bid, ask, iv = quote
                self.table.cellWidget(row, COL_BID).setValue(bid)
                self.table.cellWidget(row, COL_ASK).setValue(ask)
                if iv is not None:
                    iv_spin = self.table.cellWidget(row, COL_IV)
                    iv_spin.blockSignals(True)
                    # Pin the leg to the market IV; the slider still scales.
                    iv_spin.setProperty("manual", True)
                    iv_spin.setValue(iv * 100.0)
                    iv_spin.blockSignals(False)
                updated += 1

            main_spot = spots.get(legs[0].ticker)
            if main_spot:
                self.spot_spin.blockSignals(True)
                self.spot_spin.setValue(main_spot)
                self.spot_spin.blockSignals(False)

            self._recompute()

            if main_spot:
                text = f"yfinance: spot ${main_spot:.2f}, {updated} leg(s)"
            else:
                text = f"yfinance: {updated} leg(s) updated"
            if skipped:
                text += f", {skipped} skipped"
            self.quote_status.setText(text)
        except Exception as exc:
            self.quote_status.setText(f"yfinance error: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self.update_button.setText("Update")
            self.update_button.setEnabled(True)

    def _market_quote(self, leg, chain, spot):
        """(bid, ask, iv) for one contract, or None; iv may be None."""
        if chain is None:
            return None
        days = (leg.expiry - self.valuation_date).days

        quotes = chain.calls if leg.kind == "C" else chain.puts
        matches = quotes[(quotes["strike"] - leg.strike).abs() < 0.005]
        if matches.empty:
            return None

        quote = matches.iloc[0]
        bid = float(quote.get("bid", 0.0) or 0.0)
        ask = float(quote.get("ask", 0.0) or 0.0)
        # Rejects missing quotes and NaN: any NaN comparison is False.
        if not (bid > 0 and ask >= bid):
            return None

        iv = None
        if days > 0 and spot is not None:
            mid = (bid + ask) / 2.0
            iv = implied_vol(leg.kind, spot, leg.strike, days / 365.0, mid)
        return bid, ask, iv

    @staticmethod
    def _fetch_last_price(yf, ticker):
        try:
            price = float(yf.Ticker(ticker).fast_info["last_price"])
            if price > 0:
                return price
        except Exception:
            pass
        try:
            closes = yf.Ticker(ticker).history(period="1d")["Close"]
            price = float(closes.iloc[-1])
            if price > 0:
                return price
        except Exception:
            pass
        return None

    def _stock_quote(self, yf, ticker, spot, chains):
        """(bid, ask, atm_iv) of the underlying itself, or None."""
        bid = ask = None
        try:
            info = yf.Ticker(ticker).info
            bid = float(info.get("bid") or 0.0)
            ask = float(info.get("ask") or 0.0)
        except Exception:
            pass
        if not (bid and ask and bid > 0 and ask >= bid):
            # No quote outside trading hours: fall back to the last price.
            if not spot:
                return None
            bid = ask = spot

        expiry = None
        for (chain_ticker, chain_expiry), chain in chains.items():
            if chain_ticker != ticker or chain is None:
                continue
            if expiry is None or chain_expiry < expiry:
                expiry = chain_expiry
        if expiry is None:
            try:
                listed = yf.Ticker(ticker).options
            except Exception:
                listed = []
            if listed:
                expiry = date.fromisoformat(listed[0])

        iv = None
        if expiry is not None and spot:
            chain = chains.get((ticker, expiry))
            if chain is None:
                try:
                    chain = yf.Ticker(ticker).option_chain(expiry.isoformat())
                except Exception:
                    chain = None
            if chain is not None:
                days = (expiry - self.valuation_date).days
                iv = self._atm_cross_iv(chain, spot, days)
        return bid, ask, iv

    @staticmethod
    def _atm_cross_iv(chain, spot, days):
        """Average IV of the calls and puts on the two strikes around the spot."""
        if days <= 0 or not spot:
            return None

        strikes = set()
        for quotes in (getattr(chain, "calls", None),
                       getattr(chain, "puts", None)):
            if quotes is not None and not quotes.empty:
                strikes.update(float(strike) for strike in quotes["strike"])
        if not strikes:
            return None

        below = [strike for strike in strikes if strike <= spot]
        above = [strike for strike in strikes if strike >= spot]
        neighbours = {max(below) if below else min(strikes),
                      min(above) if above else max(strikes)}

        values = []
        for strike in neighbours:
            for quotes, kind in ((getattr(chain, "calls", None), "C"),
                                 (getattr(chain, "puts", None), "P")):
                if quotes is None or quotes.empty:
                    continue
                matches = quotes[(quotes["strike"] - strike).abs() < 0.005]
                if matches.empty:
                    continue
                quote = matches.iloc[0]
                bid = float(quote.get("bid", 0.0) or 0.0)
                ask = float(quote.get("ask", 0.0) or 0.0)
                # Also rejects NaN: any NaN comparison is False.
                if not (bid > 0 and ask >= bid):
                    continue
                iv = implied_vol(kind, spot, strike, days / 365.0,
                                 (bid + ask) / 2.0)
                if iv is not None:
                    values.append(iv)
        if not values:
            return None
        return sum(values) / len(values)

    def _add_default_legs(self):
        """Bear call ladder 330/355/380C, 24-Dec-2026: $1,763 credit."""
        expiry = date(2026, 12, 24)
        self._add_leg_row(Leg("AAPL", "C", 330.0, expiry, 1, -1, 30.00))
        self._add_leg_row(Leg("AAPL", "C", 355.0, expiry, 1, +1, 10.95))
        self._add_leg_row(Leg("AAPL", "C", 380.0, expiry, 1, +1, 1.42))

    def load_diagonal(self):
        """Double diagonal: 300P/390C (24-Dec-2026), 285P/395C (15-Jan-2027)."""
        while self.table.rowCount() > 0:
            self.table.removeRow(0)
        near = date(2026, 12, 24)
        far = date(2027, 1, 15)
        self._add_leg_row(Leg("AAPL", "P", 300.0, near, 1, -1, 0.82))
        self._add_leg_row(Leg("AAPL", "C", 390.0, near, 1, -1, 0.41))
        self._add_leg_row(Leg("AAPL", "P", 285.0, far, 1, +1, 1.07))
        self._add_leg_row(Leg("AAPL", "C", 395.0, far, 1, +1, 1.23))
        self.spot_spin.setValue(353.23)
        self._recompute()

    def load_portfolio(self, path):
        """Load a portfolio: legs with their quotes, spot, valuation date."""
        meta, legs = read_portfolio(path)
        while self.table.rowCount() > 0:
            self.table.removeRow(0)

        self.portfolio_name = meta["name"]
        self.source_text = meta["source"]
        self.valuation_date = meta["valuation_date"]
        self.spot_spin.blockSignals(True)
        self.spot_spin.setValue(meta["spot"])
        self.spot_spin.blockSignals(False)
        for leg in legs:
            self._add_leg_row(leg)

        view = meta["view"]
        self.center_on_spot = bool(view.get("center_spot", False))
        self.mid_curve_days_before = int(view.get("mid_days", 0)) or None
        if view.get("range"):
            self.range_slider.setValue(int(view["range"]))
        if view.get("iv"):
            self.iv_slider.setValue(int(view["iv"]))
        self._recompute()

        # The day slider is bounded by the expiries, so it moves after
        # the first pass.
        if "days" in view:
            self.days_slider.setValue(int(view["days"]))
            self._recompute()

        source = self.source_text or "saved file"
        self.quote_status.setText(
            f"{self.portfolio_name} · {source} · "
            f"valued {self.valuation_date.isoformat()}")
        if view.get("cursor"):
            self.chart.set_cursor_price(float(view["cursor"]))

    def _on_load_portfolio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open portfolio", "portfolios", "Portfolio (*.json)")
        if path:
            self.load_portfolio(path)

    def _on_save_portfolio(self):
        """Write the current table in the portfolio JSON format."""
        suggested = self.portfolio_name or "portfolio"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save portfolio", f"{suggested}.json",
            "Portfolio (*.json)")
        if not path:
            return

        view = {
            "range": self.range_slider.value(),
            "iv": self.iv_slider.value(),
            "days": self.days_slider.value(),
        }
        if self.mid_curve_days_before:
            view["mid_days"] = self.mid_curve_days_before
        if self.center_on_spot:
            view["center_spot"] = True
        data = portfolio_to_data(
            self.portfolio_name, self.valuation_date,
            self.spot_spin.value(), self.source_text,
            self._read_legs(), view)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        self.quote_status.setText(f"saved {path}")

    def _add_leg_row(self, leg):
        row = self.table.rowCount()
        self.table.insertRow(row)

        ticker = QLineEdit(leg.ticker)
        ticker.setMaximumWidth(64)
        ticker.textChanged.connect(self._recompute)

        kind = QComboBox()
        kind.addItems(["Call", "Put", "Stock"])
        kind.setCurrentIndex({"C": 0, "P": 1, "S": 2}.get(leg.kind, 0))
        kind.currentIndexChanged.connect(self._on_kind_changed)

        strike = QDoubleSpinBox()
        # Zero renders as "—" (a stock has no strike).
        strike.setRange(0.0, 1_000_000.0)
        strike.setDecimals(2)
        strike.setSingleStep(5.0)
        strike.setSpecialValueText("—")
        strike.setButtonSymbols(QAbstractSpinBox.NoButtons)
        strike.setAlignment(Qt.AlignRight)
        strike.setValue(leg.strike)
        strike.valueChanged.connect(self._recompute)

        expiry = QDateEdit()
        expiry.setCalendarPopup(True)
        expiry.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        expiry.setDisplayFormat("dd-MMM-yy")
        expiry.setButtonSymbols(QAbstractSpinBox.NoButtons)
        expiry.setMaximumWidth(96)
        # The minimum date renders as "—" (stock row).
        expiry.setMinimumDate(QDate(1900, 1, 1))
        expiry.setSpecialValueText("—")
        expiry.setDate(QDate(leg.expiry.year, leg.expiry.month, leg.expiry.day))
        expiry.dateChanged.connect(self._recompute)

        qty = QSpinBox()
        qty.setRange(1, 1_000_000)
        qty.setButtonSymbols(QAbstractSpinBox.NoButtons)
        qty.setAlignment(Qt.AlignRight)
        qty.setValue(leg.qty)
        qty.valueChanged.connect(self._recompute)

        side = QComboBox()
        side.addItems(["Long (+)", "Short (−)"])
        side.setCurrentIndex(0 if leg.side > 0 else 1)
        side.currentIndexChanged.connect(self._recompute)

        entry = QDoubleSpinBox()
        entry.setRange(0.0, 1_000_000.0)
        entry.setDecimals(3)
        entry.setButtonSymbols(QAbstractSpinBox.NoButtons)
        entry.setAlignment(Qt.AlignRight)
        entry.setValue(leg.entry)
        entry.valueChanged.connect(self._recompute)

        # The quoted snapshot: shown as received, never fed to the model.
        bid = self._quote_cell()
        ask = self._quote_cell()
        for cell, value in ((bid, leg.bid), (ask, leg.ask)):
            if value is not None:
                cell.setValue(value)
            cell.setToolTip("Quoted with the snapshot (market data).")

        # Percent IV, solved from the entry premium; a typed value is
        # pinned by the "manual" flag.
        iv_spin = QDoubleSpinBox()
        iv_spin.setRange(0.0, 300.0)
        iv_spin.setDecimals(2)
        iv_spin.setSingleStep(0.5)
        iv_spin.setSuffix(" %")
        iv_spin.setSpecialValueText("—")
        iv_spin.setValue(leg.iv * 100.0 if leg.iv else 0.0)
        iv_spin.setProperty("manual", False)
        iv_spin.valueChanged.connect(self._on_iv_edited)

        greek_cells = []
        greek_tips = ("Delta per $1 move of the underlying",
                      "Gamma per $1 move of the underlying",
                      "Theta per calendar day",
                      "Vega per volatility point")
        for tip in greek_tips:
            cell = QLabel("—")
            cell.setAlignment(Qt.AlignCenter)
            cell.setStyleSheet("color:#c9d1dd; font-size:11px;")
            cell.setToolTip(f"Black-Scholes at the selected date: {tip}")
            greek_cells.append(cell)

        for column, widget in enumerate(
                (ticker, kind, strike, expiry, qty, side, entry,
                 bid, ask, iv_spin, *greek_cells)):
            self.table.setCellWidget(row, column, widget)

        # The quoted delta is kept on the cell; the greek column itself
        # shows the model value at the selected date.
        if leg.market_delta is not None:
            greek_cells[0].setProperty("market_delta", leg.market_delta)
        self._apply_stock_state(row)

    @staticmethod
    def _quote_cell():
        cell = QDoubleSpinBox()
        cell.setRange(0.0, 1_000_000.0)
        cell.setDecimals(2)
        cell.setButtonSymbols(QAbstractSpinBox.NoButtons)
        cell.setReadOnly(True)
        cell.setAlignment(Qt.AlignRight)
        cell.setSpecialValueText("—")
        return cell

    def _apply_stock_state(self, row):
        """Blank strike, expiration and IV on a stock row."""
        is_stock = self.table.cellWidget(row, 1).currentIndex() == 2
        strike = self.table.cellWidget(row, 2)
        expiry = self.table.cellWidget(row, 3)
        qty = self.table.cellWidget(row, 4)
        entry = self.table.cellWidget(row, COL_ENTRY)

        for cell in (strike, expiry):
            cell.setEnabled(not is_stock)
        self.table.cellWidget(row, COL_IV).setEnabled(not is_stock)
        qty.setToolTip("Shares (not contracts)" if is_stock else "Contracts")

        # Park the option's values while the row is stock.
        for cell in (strike, expiry, qty, entry):
            cell.blockSignals(True)
        if is_stock:
            if strike.value() > 0:
                strike.setProperty("parked_strike", strike.value())
            if expiry.date() != expiry.minimumDate():
                expiry.setProperty("parked_date", expiry.date())
            qty.setProperty("parked_qty", qty.value())
            entry.setProperty("parked_entry", entry.value())
            strike.setValue(0.0)                     # blank
            expiry.setDate(expiry.minimumDate())
        else:
            if strike.value() == 0.0:
                parked_strike = strike.property("parked_strike")
                strike.setValue(parked_strike if parked_strike
                                else round(self.spot_spin.value()))
            if expiry.date() == expiry.minimumDate():
                parked_date = expiry.property("parked_date")
                if not isinstance(parked_date, QDate) or not parked_date.isValid():
                    parked_date = QDate.currentDate().addDays(28)
                expiry.setDate(parked_date)
            parked_qty = qty.property("parked_qty")
            if parked_qty:
                qty.setValue(parked_qty)
            parked_entry = entry.property("parked_entry")
            if parked_entry is not None:
                entry.setValue(parked_entry)
        for cell in (strike, expiry, qty, entry):
            cell.blockSignals(False)

    def _on_kind_changed(self, _index=0):
        combo = self.sender()
        for row in range(self.table.rowCount()):
            if self.table.cellWidget(row, 1) is not combo:
                continue
            self._apply_stock_state(row)
            if combo.currentIndex() == 2:
                # A stock leg starts as a 100-share lot bought at the spot.
                qty = self.table.cellWidget(row, 4)
                entry = self.table.cellWidget(row, COL_ENTRY)
                for spin, value in ((qty, 100),
                                    (entry, self.spot_spin.value())):
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)
            break
        self._recompute()

    def _on_iv_edited(self):
        spin = self.sender()
        if spin is not None:
            spin.setProperty("manual", True)
        self._recompute()

    def _read_legs(self):
        legs = []
        for row in range(self.table.rowCount()):
            ticker = self.table.cellWidget(row, 0).text().strip().upper() or "AAPL"
            kind_index = self.table.cellWidget(row, 1).currentIndex()
            if kind_index == 2:
                kind = "S"
            else:
                kind = "C" if kind_index == 0 else "P"
            strike = self.table.cellWidget(row, 2).value()
            expiry = self.table.cellWidget(row, 3).date().toPyDate()
            qty = self.table.cellWidget(row, 4).value()
            side = 1 if self.table.cellWidget(row, 5).currentIndex() == 0 else -1
            entry = self.table.cellWidget(row, COL_ENTRY).value()
            iv_spin = self.table.cellWidget(row, COL_IV)
            iv = iv_spin.value() / 100.0 if iv_spin.property("manual") else None

            market = {}
            for name, column in (("bid", COL_BID), ("ask", COL_ASK)):
                value = self.table.cellWidget(row, column).value()
                market[name] = value if value > 0 else None
            market_delta = self.table.cellWidget(row, COL_DELTA).property(
                "market_delta")
            market["market_delta"] = (float(market_delta)
                                      if market_delta is not None else None)
            legs.append(Leg(ticker, kind, strike, expiry, qty, side, entry,
                            iv, **market))
        return legs

    def _recompute(self, *_args):
        legs = self._read_legs()
        option_legs = [leg for leg in legs if leg.kind != "S"]
        spot = self.spot_spin.value()
        today = self.valuation_date
        range_pct = self.range_slider.value()
        scale = self.iv_slider.value() / 100.0
        SIGMA["value"] = 0.25 * scale

        # Per-leg vol from the entry premium at the spot; the slider scales
        # it, and a manual IV cell is taken as is.
        atm_used = None
        atm_gap = None
        for row, leg in enumerate(legs):
            if leg.kind == "S":
                continue
            days_left = (leg.expiry - today).days
            if leg.iv is not None:
                base = leg.iv
            elif days_left <= 0:
                base = None
            else:
                base = implied_vol(leg.kind, spot, leg.strike,
                                   days_left / 365.0, leg.entry)
                if base is None:
                    base = 0.25
                spin = self.table.cellWidget(row, COL_IV)
                spin.blockSignals(True)
                spin.setValue(base * 100.0)
                spin.blockSignals(False)

            if base is None:
                leg.iv = None
                continue
            leg.iv = base * scale
            gap = abs(leg.strike - spot)
            if atm_gap is None or gap < atm_gap:
                atm_gap = gap
                atm_used = leg.iv

        if atm_used is None:
            self.iv_caption.setText(f"IMPLIED VOLATILITY: x{scale:.2f}")
        else:
            self.iv_caption.setText(
                f"IMPLIED VOLATILITY: {atm_used * 100:.1f}% (x{scale:.2f})")

        if legs:
            self.badge.setText(f"{legs[0].ticker} ▲")
            # Today to the latest option expiry; stock legs never stretch it.
            furthest = max(0, max((leg.expiry - today).days for leg in option_legs)) \
                if option_legs else 0
            self.days_slider.setMaximum(furthest)
        else:
            self.badge.setText("—")

        days = self.days_slider.value()
        eval_date = today + timedelta(days=days)
        self.days_caption.setText(f"DATE: {date_label(eval_date, today)}")

        upcoming = [(leg.expiry - eval_date).days
                    for leg in option_legs if leg.expiry > eval_date]
        if not legs:
            self.days_remaining.setText("")
        elif upcoming:
            self.days_remaining.setText(f"({min(upcoming)}d remaining)")
        else:
            self.days_remaining.setText("(At expiration)")

        # Strike midpoint, or the spot; a stock-only position has no strikes.
        if self.center_on_spot or not option_legs:
            center = spot
        else:
            center = (min(leg.strike for leg in option_legs)
                      + max(leg.strike for leg in option_legs)) / 2.0

        x_min = center * (1.0 - range_pct / 100.0)
        x_max = center * (1.0 + range_pct / 100.0)
        self.range_caption.setText(
            f"RANGE: ±{range_pct}% ({x_min:,.0f}..{x_max:,.0f})")
        xs = self._price_grid(x_min, x_max, legs)

        # Blue markers follow the selected-date curve, not the expiry.
        self._update_metrics(legs, spot, center)
        self._update_greeks(legs, spot, eval_date)
        self._update_position_greeks(legs, spot, eval_date)
        series = self._build_series(legs, xs, eval_date, today)
        chart_breakevens = []
        if series:
            chart_breakevens = self._zero_crossings(xs, series[0]["ys"])
        self.chart.set_data(series, spot, x_min, x_max, center, chart_breakevens)

    @staticmethod
    def _zero_crossings(xs, ys):
        crossings = []
        for i in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            if y0 == 0:
                crossings.append(xs[i])
            elif y0 * y1 < 0:
                share = y0 / (y0 - y1)
                crossings.append(xs[i] + share * (xs[i + 1] - xs[i]))
        return crossings

    def _price_grid(self, low, high, legs, steps=400):
        """Uniform grid plus the strikes: a tent peak sits right on a strike."""
        prices = {low + i * (high - low) / steps for i in range(steps + 1)}
        for leg in legs:
            if leg.kind != "S" and low <= leg.strike <= high:
                prices.add(leg.strike)
        return sorted(prices)

    def _build_series(self, legs, xs, eval_date, today):
        if not legs:
            return []

        def curve(on_date):
            return [position_pl(legs, price, on_date) for price in xs]

        series = [{
            "style": "solid",
            "color": "#dfe3ea",
            "label": date_label(eval_date, today),
            "xs": xs,
            "ys": curve(eval_date),
        }]

        # Selected date, an intermediate curve and the nearest expiry; the
        # middle one is the midpoint, or N days before when asked.
        later = sorted({leg.expiry for leg in legs
                        if leg.kind != "S" and leg.expiry > eval_date})
        if later:
            nearest = later[0]
            if self.mid_curve_days_before:
                middle = nearest - timedelta(days=self.mid_curve_days_before)
                if middle <= eval_date:
                    middle = eval_date + timedelta(
                        days=max(1, (nearest - eval_date).days // 2))
            else:
                middle = eval_date + timedelta(
                    days=max(1, (nearest - eval_date).days // 2))
            if middle < nearest:
                series.append({
                    "style": "dash",
                    "color": "#ff8f8f",
                    "label": date_label(middle, today),
                    "xs": xs,
                    "ys": curve(middle),
                })
            series.append({
                "style": "dot",
                "color": "#b08a8a",
                "label": f"Expiration ({(nearest - today).days}d)",
                "xs": xs,
                "ys": curve(nearest),
            })

        return series

    def _update_metrics(self, legs, spot, center):
        if not legs:
            self.metric_debit_title.setText("NET DEBIT")
            self.metric_debit.setText("—")
            self.metric_loss.setText("—")
            self.metric_profit.setText("—")
            self.metric_break_title.setText("BREAKEVEN")
            self.metric_break.setText("—")
            return []

        debit = 0.0
        for leg in legs:
            basis = leg.qty * leg.entry
            if leg.kind != "S":
                basis *= CONTRACT_SIZE
            debit += leg.side * basis
        if debit > 0.5:
            self.metric_debit_title.setText("NET DEBIT")
            self.metric_debit.setText(f"💰 {money(debit)}")
        elif debit < -0.5:
            self.metric_debit_title.setText("NET CREDIT")
            self.metric_debit.setText(f"💰 {money(-debit)}")
        else:
            self.metric_debit_title.setText("NET DEBIT")
            self.metric_debit.setText("💰 $0")

        # Metrics use the nearest option expiry over a range wider than the
        # chart, so the asymptotes are captured. The lower edge is zero:
        # with a long stock the worst case sits at the far downside.
        first_expiry = min((leg.expiry for leg in legs if leg.kind != "S"),
                           default=self.valuation_date)
        wide_min = 0.01   # a cent, not zero: BS is the model below it
        wide_max = spot * 3.0
        wide_xs = self._price_grid(wide_min, wide_max, legs)
        payoff = [position_pl(legs, price, first_expiry) for price in wide_xs]

        # Edge slope in $ per $ of spot: about ±100 means unbounded.
        step_right = wide_xs[-1] - wide_xs[-2]
        step_left = wide_xs[1] - wide_xs[0]
        rise_rate = (payoff[-1] - payoff[-2]) / step_right
        fall_rate = (payoff[1] - payoff[0]) / step_left
        unlimited = 0.2 * CONTRACT_SIZE

        rises_right = rise_rate > unlimited
        falls_right = rise_rate < -unlimited
        falls_left = fall_rate < -unlimited

        if falls_right or falls_left:
            self.metric_loss.setText("↘ ∞")
        else:
            self.metric_loss.setText(f"↘ {money(max(0.0, -min(payoff)))}")

        if rises_right:
            self.metric_profit.setText("↗ ∞")
        else:
            self.metric_profit.setText(f"↗ {money(max(0.0, max(payoff)))}")

        breakevens = []
        for i in range(len(wide_xs) - 1):
            y0, y1 = payoff[i], payoff[i + 1]
            if y0 == 0:
                breakevens.append(wide_xs[i])
            elif y0 * y1 < 0:
                share = y0 / (y0 - y1)
                breakevens.append(wide_xs[i] + share * (wide_xs[i + 1] - wide_xs[i]))

        if not breakevens:
            self.metric_break_title.setText("BREAKEVEN")
            self.metric_break.setText("—")
        elif len(breakevens) == 1:
            value = breakevens[0]
            direction = "Above" if value > spot else "Below"
            # Measured from the range midpoint, not from the spot.
            percent = (value / center - 1.0) * 100.0
            self.metric_break_title.setText("BREAKEVEN")
            self.metric_break.setText(
                f"→ {direction} {price_label(value)} ({percent:+.1f}%)")
        elif len(breakevens) == 2:
            self.metric_break_title.setText("BREAKEVENS")
            self.metric_break.setText(
                f"→ Outside of {price_label(breakevens[0])} - "
                f"{price_label(breakevens[-1])}")
        else:
            self.metric_break_title.setText("BREAKEVENS")
            zones = [f"Between {price_label(low)} - {price_label(high)}"
                     for low, high in zip(breakevens[0::2], breakevens[1::2])]
            first, *rest = zones
            self.metric_break.setText(
                "→ " + first + "".join("\n" + zone for zone in rest))

        return breakevens

    def _update_greeks(self, legs, spot, eval_date):
        """Fill the greek columns at the selected date; stock shows delta only."""
        normal = "color:#c9d1dd; font-size:11px;"
        muted = "color:#4a5468; font-size:11px;"
        for row, leg in enumerate(legs):
            cells = [self.table.cellWidget(row, column)
                     for column in (COL_DELTA, COL_GAMMA,
                                    COL_THETA, COL_VEGA)]
            if None in cells:
                continue
            delta_cell, gamma_cell, theta_cell, vega_cell = cells
            for cell in cells:
                cell.setStyleSheet(normal)
            tip = ("Black-Scholes at the selected date: Delta per $1 "
                   "move of the underlying")
            if leg.market_delta is not None:
                tip += f"\nQuoted Δ {leg.market_delta:+.4f} (market data)."
            delta_cell.setToolTip(tip)

            def blank_rest():
                for cell in (gamma_cell, theta_cell, vega_cell):
                    cell.setText("—")
                    cell.setStyleSheet(muted)

            if leg.kind == "S":
                # A share is delta one; the position size is in POSITION (MODEL).
                delta_cell.setText(f"{leg.side:+.2f}")
                delta_cell.setToolTip(
                    f"Stock: Delta 1 per share ({leg.qty:,} shares, "
                    f"position Δ {leg.side * leg.qty:+,} sh).")
                blank_rest()
                continue

            years = (leg.expiry - eval_date).days / 365.0
            if years <= 0:
                # At expiration only the delta step is left.
                if spot > leg.strike:
                    delta = 1.0 if leg.kind == "C" else -1.0
                elif spot < leg.strike:
                    delta = 0.0
                else:
                    delta = 0.5 if leg.kind == "C" else -0.5
                delta *= leg.side
                if delta == 0:
                    delta = 0.0
                delta_cell.setText(f"{delta:+.2f}")
                blank_rest()
                continue
            if not leg.iv:
                delta_cell.setText("—")
                delta_cell.setStyleSheet(muted)
                blank_rest()
                continue

            greeks = black_scholes_greeks(
                leg.kind, spot, leg.strike, years, leg.iv)
            if greeks is None:
                delta_cell.setText("—")
                delta_cell.setStyleSheet(muted)
                blank_rest()
                continue
            delta, gamma, theta, vega = greeks
            # Long-only values above; a sold leg flips all of them.
            delta *= leg.side
            gamma *= leg.side
            theta *= leg.side
            vega *= leg.side
            if abs(delta) < 0.005:
                delta = 0.0
            if abs(theta) < 0.0005:
                theta = 0.0
            delta_cell.setText(f"{delta:+.2f}")
            gamma_cell.setText(f"{gamma:.3f}")
            theta_cell.setText(f"{theta:+.3f}")
            vega_cell.setText(f"{vega:.3f}")

    def _set_greek_headers(self, delta_shares=None, gamma=None,
                           theta_day=None, vega_point=None):
        """Greek headers carry the position total next to the letter."""
        labels = (("Δ", delta_shares, "{:+,.0f}"),
                  ("Γ", gamma, "{:+,.2f}"),
                  ("Θ", theta_day, "{:+,.0f}"),
                  ("ν", vega_point, "{:+,.0f}"))
        for column, (name, value, fmt) in zip(
                (COL_DELTA, COL_GAMMA, COL_THETA, COL_VEGA), labels):
            # "Γ: +0.02%": the letter, the total, the unit on one line.
            text = name
            if value is not None and (name == "Δ" or abs(value) >= 0.005):
                text = f"{name}: {fmt.format(value)}%"
            self.table.horizontalHeaderItem(column).setText(text)

    def _update_position_greeks(self, legs, spot, eval_date):
        """Dollar greeks of the whole position at the selected date."""
        if not legs:
            self.position_greeks.setText("")
            self._set_greek_headers()
            return
        self.position_greeks.setToolTip(
            "Model values at the selected date, per the full quantity: "
            "ΣΔ shares, ΣΓ Δ per $1, ΣΘ $ per day, Σν $ per vol point.")

        delta_shares = 0.0
        gamma = 0.0
        theta_day = 0.0
        vega_point = 0.0
        for leg in legs:
            contracts = leg.side * leg.qty * CONTRACT_SIZE
            if leg.kind == "S":
                delta_shares += leg.side * leg.qty
                continue

            years = (leg.expiry - eval_date).days / 365.0
            if years <= 0:
                if spot == leg.strike:
                    step = 0.5 * (1.0 if leg.kind == "C" else -1.0)
                elif spot > leg.strike:
                    step = 1.0 if leg.kind == "C" else 0.0
                else:
                    step = 0.0 if leg.kind == "C" else -1.0
                delta_shares += contracts * step
                continue
            if not leg.iv:
                continue
            greeks = black_scholes_greeks(leg.kind, spot, leg.strike,
                                          years, leg.iv)
            if greeks is None:
                continue
            leg_delta, leg_gamma, leg_theta, leg_vega = greeks
            delta_shares += contracts * leg_delta
            gamma += contracts * leg_gamma
            theta_day += contracts * leg_theta
            vega_point += contracts * leg_vega

        self.position_greeks.setText(
            f"POSITION (MODEL)   ΣΔ {delta_shares:+,.0f} sh · "
            f"ΣΓ {gamma:+,.2f} · ΣΘ {theta_day:+,.0f} $/day · "
            f"Σν {vega_point:+,.0f} $/vol pt")
        self._set_greek_headers(delta_shares, gamma, theta_day, vega_point)


def run_selftest():
    """Model self-check: no GUI, no network."""
    SIGMA["value"] = 0.22
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    call0 = black_scholes("C", 100, 100, 1.0, 0.2, rate=0.0)
    put0 = black_scholes("P", 100, 100, 1.0, 0.2, rate=0.0)
    check("Black-Scholes call ATM at r = 0 (100, 1y, 20%) = 7.9656",
          abs(call0 - 7.9656) < 0.001, f"{call0:.6f}")
    check("call = put at r = 0", abs(call0 - put0) < 1e-9, f"{put0:.6f}")

    call_parity = black_scholes("C", 120, 100, 0.5, 0.3)
    put_parity = black_scholes("P", 120, 100, 0.5, 0.3)
    expected_parity = 120.0 - 100.0 * math.exp(-RISK_FREE * 0.5)
    check("put-call parity C - P = S - K·e^(-rT) at r = 4.2%",
          abs((call_parity - put_parity) - expected_parity) < 1e-9,
          f"{call_parity - put_parity:.9f}")

    call_atm = black_scholes("C", 100, 100, 1.0, 0.2)
    check("Black-Scholes call ATM at r = 4.2% = 10.0291",
          abs(call_atm - 10.029063) < 0.001, f"{call_atm:.6f}")

    check("expiration: call intrinsic = 30",
          abs(black_scholes("C", 130, 100, 0.0, 0.3) - 30.0) < 1e-9)
    check("expiration: out-of-the-money put = 0",
          abs(black_scholes("P", 130, 100, 0.0, 0.3)) < 1e-9)

    # Calendar: -355C 24-Dec-26 / +355C 15-Jan-27.
    legs = [
        Leg("AAPL", "C", 355.0, date(2026, 12, 24), 1, -1, 4.20),
        Leg("AAPL", "C", 355.0, date(2027, 1, 15), 1, +1, 11.05),
    ]
    debit = sum(leg.side * leg.qty * leg.entry * CONTRACT_SIZE for leg in legs)
    check("test bench: net debit $685", abs(debit - 685.0) < 1e-9, f"{debit:.2f}")

    near_expiry = date(2026, 12, 24)
    low_pl = position_pl(legs, 10.0, near_expiry)
    check("calendar: P/L at S = 10 equals -$685",
          abs(low_pl + 685.0) < 0.6, f"{low_pl:.2f}")

    # The rate lifts a deep in-the-money far call slightly above -$685.
    years_left = (legs[1].expiry - near_expiry).days / 365.0
    expected_high = (-685.0 + CONTRACT_SIZE * legs[1].strike
                     * (1.0 - math.exp(-RISK_FREE * years_left)))
    high_pl = position_pl(legs, 3000.0, near_expiry)
    check("calendar: P/L at S = 3000 = -$685 + rate correction",
          abs(high_pl - expected_high) < 0.6, f"{high_pl:.2f}")

    peak = max(position_pl(legs, 300.0 + i * 0.5, near_expiry)
               for i in range(161))
    check("calendar: there is profit inside the range (tent)",
          peak > 0.0, f"{peak:.2f}")

    # 323C/328C/333C: debit $685, max loss $1,185.
    photo = [
        Leg("AAPL", "C", 323.0, date(2026, 12, 25), 1, -1, 21.35),
        Leg("AAPL", "C", 328.0, date(2026, 12, 25), 1, +1, 13.70),
        Leg("AAPL", "C", 333.0, date(2026, 12, 25), 1, +1, 14.50),
    ]
    photo_expiry = date(2026, 12, 25)
    photo_debit = sum(leg.side * leg.qty * leg.entry * CONTRACT_SIZE
                      for leg in photo)
    check("photo: net debit $685", abs(photo_debit - 685.0) < 1e-9,
          f"{photo_debit:.2f}")
    check("photo: P/L at S = 300 equals -$685",
          abs(position_pl(photo, 300.0, photo_expiry) + 685.0) < 0.6)
    check("photo: P/L at S = 330 equals -$1,185",
          abs(position_pl(photo, 330.0, photo_expiry) + 1185.0) < 0.6)
    check("photo: P/L at S = 355 equals +$1,015",
          abs(position_pl(photo, 355.0, photo_expiry) - 1015.0) < 0.6)
    check("photo: max loss $1,185",
          abs(min(position_pl(photo, 60.0 + i * 0.5, photo_expiry)
                  for i in range(3000)) + 1185.0) < 0.6)

    # -1185 + 100·(S - 333) = 0 at $344.85.
    prev = None
    photo_breakeven = None
    price = 280.0
    while price <= 400.0:
        value = position_pl(photo, price, photo_expiry)
        if prev is not None and prev < 0 <= value:
            photo_breakeven = price
        prev = value
        price += 0.01
    check("photo: breakeven $344.85",
          photo_breakeven is not None and abs(photo_breakeven - 344.85) < 0.02,
          f"{photo_breakeven:.2f}" if photo_breakeven else "not found")

    # Debit put spread 350/340: -$200 debit, +$800 below 340.
    put_spread = [
        Leg("X", "P", 350.0, date(2027, 1, 15), 1, +1, 5.00),
        Leg("X", "P", 340.0, date(2027, 1, 15), 1, -1, 3.00),
    ]
    expiry = date(2027, 1, 15)
    low = position_pl(put_spread, 0.5, expiry)
    high = position_pl(put_spread, 400.0, expiry)
    check("put spread: profit $800 as S→0", abs(low - 800.0) < 0.6, f"{low:.2f}")
    check("put spread: loss -$200 as S→∞", abs(high + 200.0) < 0.6, f"{high:.2f}")

    # The ATM 355C at $10.95 with 28 days left implies about 25.7%.
    solved = implied_vol("C", 355.58, 355.0, 28.0 / 365.0, 10.95)
    check("implied vol: ATM 355C from $10.95 ≈ 25.7%",
          solved is not None and abs(solved - 0.257) < 0.003,
          f"{solved * 100:.2f}%" if solved else "None")

    # With per-leg vols the "now" curve is zero at the spot and its trough
    # is a small loss, not the +$266 a single-IV model gives.
    reference_today = date(2026, 11, 26)
    ladder = [
        Leg("AAPL", "C", 330.0, date(2026, 12, 24), 1, -1, 30.00),
        Leg("AAPL", "C", 355.0, date(2026, 12, 24), 1, +1, 10.95),
        Leg("AAPL", "C", 380.0, date(2026, 12, 24), 1, +1, 1.42),
    ]
    years_now = (ladder[0].expiry - reference_today).days / 365.0
    for leg in ladder:
        leg.iv = implied_vol(leg.kind, 355.58, leg.strike, years_now, leg.entry)
    anchor = position_pl(ladder, 355.58, reference_today)
    check("per-leg vol: today curve is zero at the spot",
          abs(anchor) < 0.5, f"{anchor:.2f}")
    trough = min(position_pl(ladder, 340.0 + i * 0.05, reference_today)
                 for i in range(800))
    check("per-leg vol: today trough is a small loss, not +$266",
          -150.0 < trough < -10.0, f"{trough:.2f}")

    # Screen readings at ~26.5% ATM: 330 → +713, 355 → +26.5,
    # 359.05 → -11.4, 380 → +386.
    for leg in ladder:
        leg.iv = implied_vol(leg.kind, 355.58, leg.strike, years_now, leg.entry)
    atm_base = ladder[1].iv
    for leg in ladder:
        leg.iv *= 0.265 / atm_base
    site_points = {330.0: 713.0, 355.0: 26.5, 359.05: -11.4, 380.0: 386.0}
    worst = max(abs(position_pl(ladder, price, reference_today) - expected)
                for price, expected in site_points.items())
    check("site Now readings at ~26.5% ATM match within $15",
          worst < 15.0, f"worst {worst:.1f}")

    # Double diagonal, $107 debit:
    #   STO 300P 24-Dec-26 @0.82   STO 390C 24-Dec-26 @0.41
    #   BTO 285P 15-Jan-27 @1.07   BTO 395C 15-Jan-27 @1.23
    diagonal = [
        Leg("AAPL", "P", 300.0, date(2026, 12, 24), 1, -1, 0.82),
        Leg("AAPL", "C", 390.0, date(2026, 12, 24), 1, -1, 0.41),
        Leg("AAPL", "P", 285.0, date(2027, 1, 15), 1, +1, 1.07),
        Leg("AAPL", "C", 395.0, date(2027, 1, 15), 1, +1, 1.23),
    ]
    diagonal_debit = sum(leg.side * leg.qty * leg.entry * CONTRACT_SIZE
                         for leg in diagonal)
    check("double diagonal: net debit $107",
          abs(diagonal_debit - 107.0) < 1e-9, f"{diagonal_debit:.2f}")

    for leg in diagonal:
        years_left = (leg.expiry - reference_today).days / 365.0
        leg.iv = implied_vol(leg.kind, 355.58, leg.strike, years_left,
                             leg.entry)
    first_expiry = date(2026, 12, 24)
    floor_pl = position_pl(diagonal, 10.0, first_expiry)
    check("double diagonal: downside bounded at about -$1,679",
          abs(floor_pl + 1679.0) < 2.0, f"{floor_pl:.1f}")
    ceiling_pl = position_pl(diagonal, 3000.0, first_expiry)
    check("double diagonal: upside bounded at about -$507",
          abs(ceiling_pl + 507.0) < 2.0, f"{ceiling_pl:.1f}")
    pocket = position_pl(diagonal, 390.0, first_expiry)
    check("double diagonal: profit pocket near $390",
          pocket > 400.0, f"{pocket:.1f}")
    anchor_now = position_pl(diagonal, 355.58, reference_today)
    check("double diagonal: today curve is near zero at the spot",
          abs(anchor_now) < 5.0, f"{anchor_now:.1f}")

    # Stock: quantity in shares, linear P/L, no contract multiplier.
    stock = [Leg("AAPL", "S", 0.0, reference_today, 50, +1, 100.0)]
    up = position_pl(stock, 110.0, reference_today)
    check("stock: 50 shares, +$10 = +$500 (no ×100 contract)",
          abs(up - 500.0) < 1e-9, f"{up:.0f}")
    down = position_pl(stock, 90.0, reference_today)
    check("stock: 50 shares, -$10 = -$500", abs(down + 500.0) < 1e-9)
    later = reference_today + timedelta(days=30)
    check("stock: P/L does not depend on the date",
          abs(position_pl(stock, 110.0, later) - 500.0) < 1e-9)
    stock[0].side = -1
    check("stock: short 50 shares, +$10 = -$500",
          abs(position_pl(stock, 110.0, reference_today) + 500.0) < 1e-9)

    hedge_expiry = date(2027, 1, 15)
    hedge = [
        Leg("X", "S", 0.0, hedge_expiry, 100, +1, 100.0),
        Leg("X", "C", 100.0, hedge_expiry, 1, -1, 10.0),
    ]
    hedge_pl = position_pl(hedge, 150.0, hedge_expiry)
    check("stock + short call at S = 150 gives +$1,000",
          abs(hedge_pl - 1000.0) < 0.6, f"{hedge_pl:.0f}")

    call_greeks = black_scholes_greeks("C", 100.0, 100.0, 28.0 / 365.0, 0.30)
    put_greeks = black_scholes_greeks("P", 100.0, 100.0, 28.0 / 365.0, 0.30)
    check("greeks: call delta - put delta = 1",
          abs((call_greeks[0] - put_greeks[0]) - 1.0) < 1e-12)
    check("greeks: ATM call delta above 0.5 (the rate pulls it up)",
          call_greeks[0] > 0.5, f"{call_greeks[0]:.3f}")
    check("greeks: gamma and vega positive, theta negative",
          call_greeks[1] > 0 and call_greeks[3] > 0 and call_greeks[2] < 0)
    check("greeks: call and put vega agree",
          abs(call_greeks[3] - put_greeks[3]) < 1e-12)
    check("greeks: an expired option has none",
          black_scholes_greeks("C", 100.0, 100.0, 0.0, 0.3) is None)

    # The portfolio format keeps the quoted snapshot as it was received.
    sample = {
        "name": "check",
        "valuation_date": "2026-11-25",
        "spot": 100.0,
        "source": "test",
        "legs": [
            {"ticker": "msft", "kind": "C", "strike": 105.0,
             "expiry": "2027-01-15", "qty": 2, "side": -1, "entry": 3.5,
             "bid": 3.4, "ask": 3.6, "last": 3.55, "iv": 28.5,
             "volume": 40, "oi": 900, "delta": 0.41},
            {"ticker": "MSFT", "kind": "S", "qty": 100, "side": 1,
             "entry": 100.0},
        ],
    }
    meta, sample_legs = portfolio_from_data(sample)
    call = sample_legs[0]
    check("portfolio: bid/ask/last/IV/volume/OI/Δ are kept as quoted",
          call.bid == 3.4 and call.ask == 3.6 and call.last == 3.55
          and abs(call.market_iv - 0.285) < 1e-12 and call.volume == 40
          and call.oi == 900 and abs(call.market_delta - 0.41) < 1e-12)
    check("portfolio: the valuation date is kept",
          meta["valuation_date"] == date(2026, 11, 25))
    check("portfolio: a stock leg has no strike and no expiry",
          sample_legs[1].kind == "S" and sample_legs[1].strike == 0.0)
    written = portfolio_to_data(meta["name"], meta["valuation_date"],
                                meta["spot"], meta["source"], sample_legs)
    _, round_trip = portfolio_from_data(written)
    check("portfolio: JSON round trip keeps the legs and quotes",
          portfolio_to_data("x", meta["valuation_date"], 100.0, "t",
                            round_trip)["legs"] == written["legs"])

    print("\noption_profile.py self-check (no GUI, no network)")
    print("-" * 62)
    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        status = "OK  " if ok else "FAIL"
        suffix = f"  [{detail}]" if detail else ""
        print(f"  {status}  {name}{suffix}")
    print("-" * 62)
    print(f"  result: {len(checks) - failed} of {len(checks)} passed")
    return 1 if failed else 0


def apply_cli(window, argv):
    """Preset the sliders from the command line, e.g.

    python option_profile.py --diagonal --spot 353.23 --range 28 --iv 101 --days 0 --mid-days 7 --center-spot
    """

    def value(flag):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                return argv[index + 1]
        return None

    if value("--portfolio"):
        window.load_portfolio(value("--portfolio"))
    elif "--diagonal" in argv:
        window.load_diagonal()

    if value("--spot"):
        window.spot_spin.setValue(float(value("--spot")))
    if value("--range"):
        window.range_slider.setValue(int(float(value("--range"))))
    if value("--iv"):
        window.iv_slider.setValue(int(float(value("--iv"))))
    if value("--days"):
        window.days_slider.setValue(int(float(value("--days"))))
    if value("--mid-days"):
        window.mid_curve_days_before = int(float(value("--mid-days")))
    if "--center-spot" in argv:
        window.center_on_spot = True
    window._recompute()


def main():
    if "--selftest" in sys.argv:
        sys.exit(run_selftest())

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    apply_cli(window, sys.argv)
    window.show()
    sys.exit(app.exec_())


STYLE = """
QWidget { background:#0b0f16; color:#d7dbe3; font-family:'Segoe UI'; font-size:12px; }
QLabel { background:transparent; }
QLineEdit, QDoubleSpinBox, QSpinBox, QDateEdit, QComboBox {
    background:#161b26; border:1px solid #2a3140; border-radius:4px;
    padding:2px 5px; selection-background-color:#2f6fed;
}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled,
QDateEdit:disabled, QComboBox:disabled {
    color:#4a5468; background:#11151f; border-color:#1a2130;
}
QComboBox QAbstractItemView {
    background:#161b26; color:#d7dbe3; selection-background-color:#2f6fed;
}
QTableWidget { background:#0e131c; gridline-color:#1d2430; border:1px solid #1d2430; }
QHeaderView::section {
    background:#141a24; color:#8b93a3; border:none;
    padding:5px; font-weight:600;
}
QPushButton {
    background:#1d2637; border:1px solid #2d3a50; border-radius:4px;
    padding:4px 12px;
}
QPushButton:hover { background:#24324a; }
QSlider::groove:horizontal { height:4px; background:#2a3140; border-radius:2px; }
QSlider::sub-page:horizontal { background:#2f6fed; border-radius:2px; }
QSlider::handle:horizontal {
    background:#dfe3ea; width:12px; margin:-5px 0; border-radius:6px;
}
"""


if __name__ == "__main__":
    main()
