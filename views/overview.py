from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas  # type: ignore
from matplotlib.ticker import PercentFormatter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QAbstractTableModel
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect
from data import get_kpis, get_dataframe
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")


# ─────────────────── Table model (inchangé)
class PandasTableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None): return len(self._data)
    def columnCount(self, parent=None): return len(self._data.columns)

    def data(self, index, role=Qt.DisplayRole):  # type: ignore
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return str(self._data.columns[section])
        return None


# ─────────────────── Design tokens (plus clair)
ACCENT = "#6EA8FF"
ACCENT_2 = "#8EB8FF"
TEXT_MAIN = "#F7FAFF"
TEXT_MUTED = "#B8C2E6"

GLOBAL_BG = """
qradialgradient(cx:0.25, cy:0.1, radius:1.2,
                fx:0.25, fy:0.1,
                stop:0 #24356F, stop:0.35 #111936, stop:1 #0C1226)
"""
CARD_BG = """
qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #182042, stop:1 #202B56)
"""

BASE_CSS = f"""
QWidget#overview_widget {{
    background: {GLOBAL_BG};
}}
/* Cartes génériques + cartes ML */
QFrame[card="true"], QFrame[role="ml-card"] {{
    background: {CARD_BG};
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
}}
/* Titre / sous-titre de page */
QLabel[title="true"] {{
    font-family:'Inter','Segoe UI',Arial;
    font-size: 22px; font-weight: 800; color: {TEXT_MAIN};
}}
QLabel[subtitle="true"] {{
    color: {TEXT_MUTED}; font-size: 13px;
}}
/* KPI tiles */
QLabel[kpi-caption="true"] {{
    color: {TEXT_MUTED}; font-size: 12px; font-weight: 700; letter-spacing:.2px;
}}
QLabel[kpi-value="true"] {{
    color: {TEXT_MAIN}; font-size: 24px; font-weight: 900; letter-spacing:.2px;
}}
/* Bouton refresh (pill) */
QPushButton#refreshBtn {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2});
    color:#fff; border:0; border-radius:999px; padding:10px 16px;
    font-weight:700; font-family:'Inter','Segoe UI',Arial;
}}
QPushButton#refreshBtn:hover {{ filter: brightness(108%); }}
"""

def elevate(widget: QFrame, blur: int = 26, alpha: int = 55, y: int = 10):
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, y)
    shadow.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)


def kpi_tile(title: str, min_h: int = 110):
    """Carte KPI compacte (taille explicite, lisible)."""
    card = QFrame()
    card.setProperty("card", True)
    elevate(card, blur=20, alpha=40, y=8)
    card.setMinimumHeight(min_h)
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    v = QVBoxLayout(card)
    v.setContentsMargins(14, 10, 14, 12)
    v.setSpacing(4)

    bar = QFrame()
    bar.setFixedHeight(3)
    bar.setStyleSheet(
        f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT}, stop:1 {ACCENT_2});"
        "border-radius:2px;"
    )
    cap = QLabel(title); cap.setProperty("kpi-caption", True)
    val = QLabel("—");    val.setProperty("kpi-value", True)

    v.addWidget(bar)
    v.addSpacing(2)
    v.addWidget(cap)
    v.addWidget(val)
    v.addStretch(1)
    return card, val


# ─────────────────── Overview (containers plus clairs)
class OverviewWidget(QWidget):
    """Design modernisé et tailles explicites pour plus de clarté."""
    def __init__(self):
        super().__init__()
        self.setObjectName("overview_widget")
        self.setStyleSheet(BASE_CSS)

        self.data = None
        self.kpi_data = None
        self.val_labels = {}
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(60_000)

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(16)

        title = QLabel("Hotel Dashboard — Overview"); title.setProperty("title", True)
        subtitle = QLabel("KPI principaux & tendance d’occupation — tailles de cartes uniformes.")
        subtitle.setProperty("subtitle", True)
        main.addWidget(title); main.addWidget(subtitle)

        grid = QGridLayout(); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(16)
        main.addLayout(grid)

        # ---------- Board KPI (hauteur fixe et claire)
        self.kpi_board = QFrame(); self.kpi_board.setProperty("card", True); elevate(self.kpi_board)
        self.kpi_board.setMinimumHeight(260)  # bloc clair
        kb = QVBoxLayout(self.kpi_board); kb.setContentsMargins(16, 16, 16, 16); kb.setSpacing(12)

        kgrid = QGridLayout(); kgrid.setHorizontalSpacing(12); kgrid.setVerticalSpacing(12)

        c1, v1 = kpi_tile("Average Occupancy")
        c2, v2 = kpi_tile("ADR (Average Daily Rate)")
        c3, v3 = kpi_tile("RevPAR")
        c4, v4 = kpi_tile("GOPPAR")
        c5, v5 = kpi_tile("Total Revenue")

        self.val_labels = {"avg_occ": v1, "avg_rate": v2, "revpar": v3, "goppar": v4, "total_rev": v5}

        kgrid.addWidget(c1, 0, 0); kgrid.addWidget(c2, 0, 1); kgrid.addWidget(c3, 0, 2)
        kgrid.addWidget(c4, 1, 0); kgrid.addWidget(c5, 1, 1)
        # case vide pour respirer si 2e ligne
        spacer = QFrame(); spacer.setMinimumHeight(110); spacer.setProperty("card", True)
        spacer.setStyleSheet("background: transparent; border: 1px dashed rgba(255,255,255,0.05);")
        kgrid.addWidget(spacer, 1, 2)

        kb.addLayout(kgrid)
        grid.addWidget(self.kpi_board, 0, 0)

        # ---------- Carte Graphique / ML (hauteur explicite)
        self.chart_card = QFrame(); self.chart_card.setProperty("role", "ml-card"); elevate(self.chart_card)
        self.chart_card.setMinimumHeight(360)  # plus haut = plus lisible
        self.chart_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        cl = QVBoxLayout(self.chart_card); cl.setContentsMargins(16, 16, 16, 16); cl.setSpacing(10)
        ct = QLabel("Occupancy Trend"); ct.setStyleSheet(f"color:{TEXT_MAIN}; font-weight:800; font-size:15px;")
        cl.addWidget(ct)

        self.occupancy_figure = Figure(figsize=(5, 3), dpi=100); self.occupancy_figure.patch.set_alpha(0.0)
        self.occupancy_canvas = Canvas(self.occupancy_figure)
        cl.addWidget(self.occupancy_canvas)

        grid.addWidget(self.chart_card, 0, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # ---------- Bouton Refresh (pill)
        refresh = QPushButton("⟳  Refresh Data"); refresh.setObjectName("refreshBtn")
        refresh.clicked.connect(self.refresh_data)
        main.addWidget(refresh, alignment=Qt.AlignRight)

        main.addStretch()
        self.refresh_data()

    # ───────── Helpers KPI
    @staticmethod
    def _money(v: float) -> str: return f"${v:,.2f}"
    def _set(self, key: str, txt: str):
        if key in self.val_labels: self.val_labels[key].setText(txt)

    # ───────── Refresh (données → UI)
    def refresh_data(self):
        self.data = get_dataframe()
        self.kpi_data = get_kpis()

        if self.kpi_data and "message" not in self.kpi_data:
            avg_occ = float(self.kpi_data.get("average_occupancy", 0))
            avg_rate = float(self.kpi_data.get("average_rate", 0))
            revpar   = float(self.kpi_data.get("revpar", 0))
            goppar   = float(self.kpi_data.get("goppar", 0))
            total_revenue = 0.0
            if self.data is not None and {"rate","occupancy"}.issubset(self.data.columns):
                total_revenue = float((self.data["rate"] * self.data["occupancy"]).sum())

            self._set("avg_occ", f"{avg_occ:.1f}%")
            self._set("avg_rate", self._money(avg_rate))
            self._set("revpar",   self._money(revpar))
            self._set("goppar",   self._money(goppar))
            self._set("total_rev",self._money(total_revenue))
        else:
            for k in self.val_labels: self._set(k, "—")

        # Graphique (plus clair)
        if self.data is not None and {"date","occupancy"}.issubset(self.data.columns):
            try:
                if not pd.api.types.is_datetime64_any_dtype(self.data["date"]):
                    self.data["date"] = pd.to_datetime(self.data["date"])
                monthly = (self.data.groupby(self.data["date"].dt.to_period("M"))["occupancy"].mean().sort_index())

                self.occupancy_figure.clear()
                ax = self.occupancy_figure.add_subplot(111)
                ax.set_facecolor((0,0,0,0))

                months = [p.to_timestamp() for p in monthly.index]
                y = monthly.values
                ax.plot(months, y, color=ACCENT, linewidth=3.0)
                ax.fill_between(months, y, 0, color=ACCENT, alpha=0.18)
                ax.scatter(months, y, s=22, color=ACCENT_2, zorder=3)

                ax.set_ylim(0, 1)
                ax.grid(True, linestyle="--", alpha=0.22)
                for side in ("top","right","left","bottom"):
                    ax.spines[side].set_alpha(0.15)

                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

                self.occupancy_figure.tight_layout()
                self.occupancy_canvas.draw()
            except Exception as e:
                print(f"Chart error: {e}")
                self.occupancy_figure.clear()
                ax = self.occupancy_figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#FF9800")
                ax.axis("off")
                self.occupancy_figure.tight_layout()
                self.occupancy_canvas.draw()


def display():
    """Display hotel dashboard overview."""
    return OverviewWidget()

