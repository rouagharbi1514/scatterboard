"""
Company Analysis View
====================

Shows high-level business performance metrics and company KPIs.
"""

from views.utils import data_required, kpi_tile, get_df
from PySide6.QtCore import Qt, QDate
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDateEdit,
    QPushButton,
    QGridLayout,
    QComboBox,
    QTabWidget,
    QSizePolicy,
)  # Removed unused QFrame, QSplitter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib

# Matplotlib imports should be at the top
matplotlib.use("Qt5Agg")


@data_required
def display() -> QWidget:
    """Display company performance analysis dashboard."""
    base_df = get_df()

    # Main widget
    root = QWidget()
    root.setLayout(QVBoxLayout())
    header = QLabel("Company Analysis")
    header.setStyleSheet("font-size:18pt;font-weight:bold;")
    root.layout().addWidget(header)

    # Date range filter
    filter_row = QHBoxLayout()
    filter_row.addWidget(QLabel("Date Range:"))

    # Date pickers
    start_picker = QDateEdit()
    end_picker = QDateEdit()
    for p in (start_picker, end_picker):
        p.setCalendarPopup(True)

    # Set initial date range
    def refresh_date_pickers():
        try:
            if "date" in base_df.columns:
                d0 = base_df["date"].min().date()
                d1 = base_df["date"].max().date()
                start_picker.setDate(QDate(d0.year, d0.month, d0.day))
                end_picker.setDate(QDate(d1.year, d1.month, d1.day))
        except Exception as e:
            print(f"Error setting date range: {e}")

    refresh_date_pickers()  # initial sync

    filter_row.addWidget(start_picker)
    filter_row.addWidget(QLabel(" to "))
    filter_row.addWidget(end_picker)

    # Apply button
    apply_btn = QPushButton("Apply")
    filter_row.addWidget(apply_btn)
    filter_row.addStretch()

    # View selector
    filter_row.addWidget(QLabel("View:"))
    view_combo = QComboBox()
    view_combo.addItems(["Financial Performance", "Operational Metrics"])
    filter_row.addWidget(view_combo)

    root.layout().addLayout(filter_row)

    # KPI grid
    kpi_grid = QGridLayout()
    kpi_grid.setSpacing(12)
    root.layout().addLayout(kpi_grid)

    # Content tabs
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    def _generate_company_data(df):
        """Generate synthetic company data based on the hotel data."""
        if df is None or df.empty:
            return pd.DataFrame()

        if "date" not in df.columns and "Date" in df.columns:
            df = df.copy()
            df["date"] = df["Date"]

        # Group data by month
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

        # Create company KPIs dataframe
        company_data = []

        for month in df["month"].unique():
            month_df = df[df["month"] == month]

            # Extract base metrics if available
            if "rate" in df.columns and "occupancy" in df.columns:
                avg_rate = month_df["rate"].mean()
                avg_occ = month_df["occupancy"].mean()
                # Assume 100 rooms
                revenue = (month_df["rate"] * month_df["occupancy"]).sum() * 100
                revpar = (month_df["rate"] * month_df["occupancy"]).mean()
            else:
                avg_rate = np.random.uniform(120, 180)
                avg_occ = np.random.uniform(0.6, 0.9)
                revenue = avg_rate * avg_occ * 30 * 100  # Assume 100 rooms for 30 days
                revpar = avg_rate * avg_occ

            # F&B revenue is typically 25-35% of room revenue
            fb_revenue = revenue * np.random.uniform(0.25, 0.35)

            # Other revenue (spa, parking, events, etc) is typically 10-20% of
            # room revenue
            other_revenue = revenue * np.random.uniform(0.1, 0.2)

            # Total revenue
            total_revenue = revenue + fb_revenue + other_revenue

            # Cost structure (percentages of total revenue)
            labor_cost = total_revenue * np.random.uniform(0.28, 0.35)
            operating_cost = total_revenue * np.random.uniform(0.15, 0.22)
            marketing_cost = total_revenue * np.random.uniform(0.05, 0.08)
            utility_cost = total_revenue * np.random.uniform(0.03, 0.06)
            admin_cost = total_revenue * np.random.uniform(0.08, 0.12)

            # Total costs
            total_cost = (
                labor_cost + operating_cost + marketing_cost + utility_cost + admin_cost
            )

            # Profit metrics
            gross_profit = total_revenue - total_cost
            gop_margin = gross_profit / total_revenue if total_revenue > 0 else 0

            # Fixed costs (more stable month to month)
            fixed_costs = total_revenue * 0.12

            # EBITDA
            ebitda = gross_profit - fixed_costs
            ebitda_margin = ebitda / total_revenue if total_revenue > 0 else 0

            # Operational metrics
            employee_count = int(
                np.random.uniform(0.8, 1.2) * 50
            )  # Baseline of 50 employees
            customer_satisfaction = np.random.uniform(3.9, 4.8)  # Out of 5
            repeat_customer_rate = np.random.uniform(
                0.15, 0.35
            )  # 15-35% are repeat customers

            company_data.append(
                {
                    "month": month,
                    "month_date": month.to_timestamp(),
                    "room_revenue": revenue,
                    "fb_revenue": fb_revenue,
                    "other_revenue": other_revenue,
                    "total_revenue": total_revenue,
                    "labor_cost": labor_cost,
                    "operating_cost": operating_cost,
                    "marketing_cost": marketing_cost,
                    "utility_cost": utility_cost,
                    "admin_cost": admin_cost,
                    "total_cost": total_cost,
                    "gross_profit": gross_profit,
                    "gop_margin": gop_margin,
                    "fixed_costs": fixed_costs,
                    "ebitda": ebitda,
                    "ebitda_margin": ebitda_margin,
                    "employee_count": employee_count,
                    "customer_satisfaction": customer_satisfaction,
                    "repeat_customer_rate": repeat_customer_rate,
                    "avg_rate": avg_rate,
                    "avg_occupancy": avg_occ,
                    "revpar": revpar,
                }
            )

        return pd.DataFrame(company_data)

    # Generate company data
    company_df = _generate_company_data(base_df)

    def _filter_data(df, start_date, end_date):
        """Filter data by date range."""
        if df is None or df.empty:
            return pd.DataFrame()

        mask = (df["month_date"] >= start_date) & (df["month_date"] <= end_date)
        return df[mask]

    def _render():
        """Update the dashboard with filtered data."""
        try:
            # Get selected dates
            start_date = pd.Timestamp(start_picker.date().toPython())
            end_date = pd.Timestamp(end_picker.date().toPython())

            # Filter data
            df = _filter_data(company_df, start_date, end_date)

            # Clear previous content
            while kpi_grid.count():
                item = kpi_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            tabs.clear()

            if df.empty:
                empty_label = QLabel("No data available for the selected date range")
                empty_label.setStyleSheet("color: red; font-size: 14pt;")
                kpi_grid.addWidget(empty_label, 0, 0, 1, 4, Qt.AlignCenter)
                return

            # Calculate KPIs based on selected view
            view_type = view_combo.currentText()

            if view_type == "Financial Performance":
                # Financial KPIs
                total_revenue = df["total_revenue"].sum()
                avg_gop_margin = (
                    df["gross_profit"].sum() / df["total_revenue"].sum()
                ) * 100
                avg_ebitda_margin = (
                    df["ebitda"].sum() / df["total_revenue"].sum()
                ) * 100
                revenue_growth = 0
                if len(df) >= 2:
                    first_month = df.iloc[0]["total_revenue"]
                    last_month = df.iloc[-1]["total_revenue"]
                    revenue_growth = ((last_month / first_month) - 1) * 100

                # Display KPIs
                kpis = [
                    ("Total Revenue", f"${total_revenue:,.0f}"),
                    ("GOP Margin", f"{avg_gop_margin:.1f}%"),
                    ("EBITDA Margin", f"{avg_ebitda_margin:.1f}%"),
                    ("Revenue Growth", f"{revenue_growth:.1f}%"),
                ]

                for i, (label, value) in enumerate(kpis):
                    kpi_grid.addWidget(kpi_tile(label, value), 0, i)

                # Revenue Breakdown Tab
                revenue_tab = QWidget()
                revenue_tab.setLayout(QVBoxLayout())

                # Revenue Breakdown Chart
                revenue_fig = Figure(figsize=(10, 5))
                revenue_ax = revenue_fig.add_subplot(111)

                # Stack plot of different revenue sources
                revenue_ax.stackplot(
                    df["month_date"],
                    df["room_revenue"],
                    df["fb_revenue"],
                    df["other_revenue"],
                    labels=["Room Revenue", "F&B Revenue", "Other Revenue"],
                    alpha=0.7,
                    colors=["#3b82f6", "#10b981", "#f59e0b"],
                )

                revenue_ax.set_title("Monthly Revenue Breakdown")
                revenue_ax.set_xlabel("Month")
                revenue_ax.set_ylabel("Revenue ($)")
                revenue_ax.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
                )
                revenue_ax.grid(True, linestyle="--", alpha=0.7)
                revenue_ax.legend(loc="upper left")

                revenue_fig.tight_layout()
                revenue_canvas = Canvas(revenue_fig)
                revenue_canvas.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )
                revenue_tab.layout().addWidget(revenue_canvas)

                tabs.addTab(revenue_tab, "Revenue Breakdown")

                # Profitability Tab
                profit_tab = QWidget()
                profit_tab.setLayout(QVBoxLayout())

                # Monthly Profitability Chart
                profit_fig = Figure(figsize=(10, 5))
                profit_ax = profit_fig.add_subplot(111)

                # Plot gross profit and EBITDA
                profit_ax.bar(
                    df["month_date"],
                    df["gross_profit"],
                    alpha=0.7,
                    color="#3b82f6",
                    label="Gross Profit",
                )
                profit_ax.bar(
                    df["month_date"],
                    df["ebitda"],
                    alpha=0.7,
                    color="#10b981",
                    label="EBITDA",
                )

                profit_ax.set_title("Monthly Profitability")
                profit_ax.set_xlabel("Month")
                profit_ax.set_ylabel("Amount ($)")
                profit_ax.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
                )
                profit_ax.grid(True, linestyle="--", alpha=0.7)
                profit_ax.legend(loc="upper left")

                profit_fig.tight_layout()
                profit_canvas = Canvas(profit_fig)
                profit_canvas.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )
                profit_tab.layout().addWidget(profit_canvas)

                # Margin Trend Chart
                margin_fig = Figure(figsize=(10, 5))
                margin_ax = margin_fig.add_subplot(111)

                margin_ax.plot(
                    df["month_date"],
                    df["gop_margin"] * 100,
                    marker="o",
                    linestyle="-",
                    color="#3b82f6",
                    label="GOP Margin",
                )
                margin_ax.plot(
                    df["month_date"],
                    df["ebitda_margin"] * 100,
                    marker="s",
                    linestyle="-",
                    color="#10b981",
                    label="EBITDA Margin",
                )

                margin_ax.set_title("Profitability Margins")
                margin_ax.set_xlabel("Month")
                margin_ax.set_ylabel("Margin (%)")
                margin_ax.set_ylim(0, max(df["gop_margin"].max() * 100 * 1.2, 40))
                margin_ax.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("{x:.1f}%")
                )
                margin_ax.grid(True, linestyle="--", alpha=0.7)
                margin_ax.legend(loc="upper left")

                margin_fig.tight_layout()
                margin_canvas = Canvas(margin_fig)
                margin_canvas.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )
                profit_tab.layout().addWidget(margin_canvas)

                tabs.addTab(profit_tab, "Profitability")

                # Cost Structure Tab
                cost_tab = QWidget()
                cost_tab.setLayout(QVBoxLayout())

                # Cost Breakdown Chart (Pie Chart)
                cost_fig = Figure(figsize=(8, 4))
                cost_ax = cost_fig.add_subplot(121)

                # Aggregate costs
                total_costs = {
                    "Labor": df["labor_cost"].sum(),
                    "Operating": df["operating_cost"].sum(),
                    "Marketing": df["marketing_cost"].sum(),
                    "Utilities": df["utility_cost"].sum(),
                    "Admin": df["admin_cost"].sum(),
                    "Fixed Costs": df["fixed_costs"].sum(),
                }

                # Create pie chart
                wedges, texts, autotexts = cost_ax.pie(
                    total_costs.values(),
                    labels=total_costs.keys(),
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=[
                        "#3b82f6",
                        "#f59e0b",
                        "#10b981",
                        "#8b5cf6",
                        "#ef4444",
                        "#a1a1aa",
                    ],
                )

                # Equal aspect ratio ensures that pie is drawn as a circle
                cost_ax.axis("equal")
                cost_ax.set_title("Cost Breakdown")

                # Improve text visibility
                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(9)
                    autotext.set_fontweight("bold")

                # Monthly Cost Trend
                cost_trend_ax = cost_fig.add_subplot(122)

                # Stack plot of different cost categories
                cost_trend_ax.stackplot(
                    df["month_date"],
                    df["labor_cost"],
                    df["operating_cost"],
                    df["marketing_cost"],
                    df["utility_cost"],
                    df["admin_cost"],
                    df["fixed_costs"],
                    labels=[
                        "Labor",
                        "Operating",
                        "Marketing",
                        "Utilities",
                        "Admin",
                        "Fixed",
                    ],
                    alpha=0.7,
                    colors=[
                        "#3b82f6",
                        "#f59e0b",
                        "#10b981",
                        "#8b5cf6",
                        "#ef4444",
                        "#a1a1aa",
                    ],
                )

                cost_trend_ax.set_title("Monthly Cost Structure")
                cost_trend_ax.set_xlabel("Month")
                cost_trend_ax.set_ylabel("Cost ($)")
                cost_trend_ax.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
                )

                cost_fig.tight_layout()
                cost_canvas = Canvas(cost_fig)
                cost_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                cost_tab.layout().addWidget(cost_canvas)

                tabs.addTab(cost_tab, "Cost Structure")

            else:  # Operational Metrics view
                # Operational KPIs
                avg_occupancy = df["avg_occupancy"].mean() * 100
                avg_rate = df["avg_rate"].mean()
                avg_revpar = df["revpar"].mean()
                avg_satisfaction = df["customer_satisfaction"].mean()

                # Display KPIs
                kpis = [
                    ("Average Occupancy", f"{avg_occupancy:.1f}%"),
                    ("Average Rate", f"${avg_rate:.2f}"),
                    ("Average RevPAR", f"${avg_revpar:.2f}"),
                    ("Customer Satisfaction", f"{avg_satisfaction:.1f}/5.0"),
                ]

                for i, (label, value) in enumerate(kpis):
                    kpi_grid.addWidget(kpi_tile(label, value), 0, i)

                # Occupancy & Rate Tab
                occ_tab = QWidget()
                occ_tab.setLayout(QVBoxLayout())

                # Occupancy & Rate Chart
                occ_fig = Figure(figsize=(10, 5))
                occ_ax = occ_fig.add_subplot(111)
                occ_ax2 = occ_ax.twinx()

                # Plot occupancy and rate
                occ_line = occ_ax.plot(
                    df["month_date"],
                    df["avg_occupancy"] * 100,
                    "b-o",
                    label="Occupancy",
                )
                rate_line = occ_ax2.plot(
                    df["month_date"], df["avg_rate"], "r-s", label="ADR"
                )

                occ_ax.set_title("Monthly Occupancy and ADR")
                occ_ax.set_xlabel("Month")
                occ_ax.set_ylabel("Occupancy (%)", color="b")
                occ_ax.tick_params(axis="y", colors="b")
                occ_ax.set_ylim(0, 100)

                occ_ax2.set_ylabel("ADR ($)", color="r")
                occ_ax2.tick_params(axis="y", colors="r")
                occ_ax2.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("${x:.0f}")
                )

                # Add combined legend
                lines, labels = occ_ax.get_legend_handles_labels()
                lines2, labels2 = occ_ax2.get_legend_handles_labels()
                occ_ax.legend(lines + lines2, labels + labels2, loc="upper left")

                occ_fig.tight_layout()
                occ_canvas = Canvas(occ_fig)
                occ_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                occ_tab.layout().addWidget(occ_canvas)

                tabs.addTab(occ_tab, "Occupancy & Rate")

                # Customer Metrics Tab
                cust_tab = QWidget()
                cust_tab.setLayout(QVBoxLayout())

                # Customer Metrics Chart
                cust_fig = Figure(figsize=(10, 5))
                cust_ax = cust_fig.add_subplot(111)
                cust_ax2 = cust_ax.twinx()

                # Plot customer satisfaction and repeat rate
                sat_line = cust_ax.plot(
                    df["month_date"],
                    df["customer_satisfaction"],
                    "b-o",
                    label="Satisfaction",
                )
                repeat_line = cust_ax2.plot(
                    df["month_date"],
                    df["repeat_customer_rate"] * 100,
                    "g-s",
                    label="Repeat Rate",
                )

                cust_ax.set_title("Customer Metrics")
                cust_ax.set_xlabel("Month")
                cust_ax.set_ylabel("Satisfaction (1-5)", color="b")
                cust_ax.tick_params(axis="y", colors="b")
                cust_ax.set_ylim(1, 5)

                cust_ax2.set_ylabel("Repeat Rate (%)", color="g")
                cust_ax2.tick_params(axis="y", colors="g")
                cust_ax2.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("{x:.0f}%")
                )

                # Add combined legend
                lines, labels = cust_ax.get_legend_handles_labels()
                lines2, labels2 = cust_ax2.get_legend_handles_labels()
                cust_ax.legend(lines + lines2, labels + labels2, loc="upper left")

                cust_fig.tight_layout()
                cust_canvas = Canvas(cust_fig)
                cust_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                cust_tab.layout().addWidget(cust_canvas)

                # Staffing Chart
                staff_fig = Figure(figsize=(10, 5))
                staff_ax = staff_fig.add_subplot(111)

                staff_ax.plot(df["month_date"], df["employee_count"], "b-o")
                staff_ax.set_title("Monthly Employee Count")
                staff_ax.set_xlabel("Month")
                staff_ax.set_ylabel("Number of Employees")
                staff_ax.yaxis.set_major_formatter(
                    matplotlib.ticker.StrMethodFormatter("{x:.0f}")
                )
                staff_ax.grid(True, linestyle="--", alpha=0.7)

                staff_fig.tight_layout()
                staff_canvas = Canvas(staff_fig)
                staff_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                cust_tab.layout().addWidget(staff_canvas)

                tabs.addTab(cust_tab, "Customer & Staff")

        except Exception as e:
            import traceback

            traceback.print_exc()

            # Show error in KPI grid
            error_label = QLabel(f"Error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 14pt;")
            kpi_grid.addWidget(error_label, 0, 0, 1, 4, Qt.AlignCenter)

    # Connect signals
    apply_btn.clicked.connect(_render)
    view_combo.currentIndexChanged.connect(_render)

    # Initial render
    _render()

    # Expose refresh_date_pickers for data sync
    display.refresh_date_pickers = refresh_date_pickers

    return root
