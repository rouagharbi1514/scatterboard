# views/kpis.py
"""
KPI Dashboard
-------------
* Auto-synchronises its date pickers with the active DataFrame.
* Calculates the core set of KPIs listed in the spec.
"""

from __future__ import annotations
import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDateEdit,
    QPushButton,
    QGridLayout,
)

# always returns current dataframe
from data.helpers import get_df

# small tile helper already in project
from views.utils import data_required, kpi_tile


# ──────────────────────────────────────────────────────────
# KPI formulas in one function
# ──────────────────────────────────────────────────────────
def _compute_kpis(df: pd.DataFrame) -> dict[str, str]:
    out = {}
    if df.empty:
        return out

    rooms_avail = df["AvailableRooms"].sum()
    rooms_occ = df["OccupiedRooms"].sum()
    room_revenue = df["RoomRevenue"].sum()
    room_cost = df["TotalRoomCost"].sum()
    guests = df["GuestID"].nunique()

    out["Occupancy"] = f"{rooms_occ / rooms_avail :.1%}" if rooms_avail else "—"
    out["ADR"] = f"${room_revenue / rooms_occ :,.0f}" if rooms_occ else "—"
    out["RevPAR"] = f"${room_revenue / rooms_avail :,.0f}" if rooms_avail else "—"
    out["Room Profit"] = f"${room_revenue - room_cost :,.0f}"
    out["Profit Margin"] = (
        f"{(room_revenue - room_cost) / room_revenue :.1%}" if room_revenue else "—"
    )
    out["GOPPAR"] = (
        f"${(room_revenue - room_cost) / rooms_avail :,.0f}" if rooms_avail else "—"
    )
    out["CPOR"] = f"${room_cost / rooms_occ :,.0f}" if rooms_occ else "—"
    out["Repeat-Guest %"] = (
        f"{1 - df.drop_duplicates('GuestID').shape[0] / guests :.1%}" if guests else "—"
    )
    out["Avg Spend / Guest"] = (
        f"${df['TotalRevenue'].sum() / guests :,.0f}" if guests else "—"
    )
    out["Avg LOS"] = f"{df['LOS'].mean():.1f} nights" if "LOS" in df else "—"
    return out


# ──────────────────────────────────────────────────────────
# build the widget
# ──────────────────────────────────────────────────────────
@data_required
def display() -> QWidget:
    base_df = get_df()  # DataFrame is always current

    root = QWidget()
    root.setLayout(QVBoxLayout())
    
    # Add spacing at the top
    root.layout().addSpacing(20)
    
    hdr = QLabel("Key Performance Indicators")
    hdr.setStyleSheet("font-size:18pt;font-weight:bold;text-align:center;")
    hdr.setAlignment(Qt.AlignCenter)
    root.layout().addWidget(hdr)
    
    # Add more spacing below the title
    root.layout().addSpacing(15)

    # date pickers
    start_edit, end_edit = QDateEdit(), QDateEdit()
    for ed in (start_edit, end_edit):
        ed.setCalendarPopup(True)

    def refresh_date_pickers():
        if "Date" not in base_df.columns:
            return
        d0, d1 = base_df["Date"].min().date(), base_df["Date"].max().date()
        start_edit.setDate(QDate(d0.year, d0.month, d0.day))
        end_edit.setDate(QDate(d1.year, d1.month, d1.day))

    refresh_date_pickers()  # initial sync

    row = QHBoxLayout()
    row.addWidget(QLabel("Date Range:"))
    row.addWidget(start_edit)
    row.addWidget(QLabel(" to "))
    row.addWidget(end_edit)
    btn = QPushButton("Apply")
    row.addWidget(btn)
    row.addStretch()
    root.layout().addLayout(row)

    # KPI grid
    grid = QGridLayout()
    grid.setSpacing(12)
    root.layout().addLayout(grid)

    # ------------------------------------------------------------------
    def render():
        d0 = pd.Timestamp(start_edit.date().toPython())
        d1 = pd.Timestamp(end_edit.date().toPython())
        df = base_df[(base_df["Date"] >= d0) & (base_df["Date"] <= d1)].copy()

        # clear old tiles
        while grid.count():
            w = grid.takeAt(0).widget()
            w.deleteLater()

        for i, (label, val) in enumerate(_compute_kpis(df).items()):
            grid.addWidget(kpi_tile(label, val), i // 4, i % 4)

    btn.clicked.connect(render)
    render()  # first paint

    # ------------------------------------------------------------------
    # expose helper so other modules can resync pickers after file upload
    display.refresh_date_pickers = refresh_date_pickers
    return root
