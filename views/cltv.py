"""
Customer Lifetime Value Analysis
===============================

Analyze and visualize the long-term value of hotel guests.
"""

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
)
from views.utils import data_required, kpi_tile
from data.helpers import get_df
from PySide6.QtCore import Qt, QDate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")


@data_required
def display() -> QWidget:
    """Display customer lifetime value analysis dashboard."""
    # Get base hotel data
    base_df = get_df()

    # Main widget
    root = QWidget()
    root.setLayout(QVBoxLayout())
    header = QLabel("Customer Lifetime Value Analysis")
    header.setStyleSheet("font-size:18pt;font-weight:bold;")
    root.layout().addWidget(header)

    # Date range selector
    filter_row = QHBoxLayout()
    filter_row.addWidget(QLabel("Analysis Period:"))

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
    filter_row.addWidget(QLabel("Segment:"))
    segment_combo = QComboBox()
    segment_combo.addItems(["All Guests", "Leisure", "Business", "Group"])
    filter_row.addWidget(segment_combo)

    root.layout().addLayout(filter_row)

    # KPI grid
    kpi_grid = QGridLayout()
    kpi_grid.setSpacing(12)
    root.layout().addLayout(kpi_grid)

    # Content tabs
    tabs = QTabWidget()
    root.layout().addWidget(tabs)

    def _generate_guest_cltv_data(df):
        """Generate synthetic guest CLTV data based on the hotel data."""
        if df is None or df.empty:
            # Create sample date range if no data
            date_range = pd.date_range(
                start=pd.Timestamp.now() - pd.Timedelta(days=365),
                end=pd.Timestamp.now(),
                freq="D",
            )
            df = pd.DataFrame({"date": date_range})
        elif "date" not in df.columns and "Date" in df.columns:
            df = df.copy()
            df["date"] = df["Date"]

        # Generate synthetic guest profiles
        guest_count = 1000  # Number of synthetic guest profiles

        np.random.seed(42)  # For reproducibility

        # Guest IDs
        guest_ids = [f"G{i:04d}" for i in range(1, guest_count + 1)]

        # Guest segments
        segments = np.random.choice(
            ["Leisure", "Business", "Group"], size=guest_count, p=[0.6, 0.3, 0.1]
        )

        # First stay date - distribute throughout the year
        min_date = df["date"].min()
        # Allow for repeat stays
        max_date = df["date"].max() - pd.Timedelta(days=90)

        # If date range is too short, create a reasonable range
        if min_date >= max_date:
            min_date = pd.Timestamp.now() - pd.Timedelta(days=365)
            max_date = pd.Timestamp.now() - pd.Timedelta(days=30)

        date_range = (max_date - min_date).days
        first_stay_offsets = np.random.randint(0, date_range, size=guest_count)
        first_stay_dates = [
            min_date + pd.Timedelta(days=int(offset)) for offset in first_stay_offsets
        ]

        # Generate stays data
        stays_data = []

        for i, guest_id in enumerate(guest_ids):
            # Guest characteristics based on segment
            segment = segments[i]
            first_stay = first_stay_dates[i]

            if segment == "Leisure":
                # Leisure travelers have fewer stays but higher ADR
                stay_count = np.random.poisson(1.5)  # Average 1.5 stays
                # At least 1, at most 5
                stay_count = max(1, min(5, stay_count))
                avg_stay_length = np.random.uniform(2, 4)  # 2-4 nights
                avg_daily_rate = np.random.uniform(150, 250)  # Higher rates
                ancillary_spend_ratio = np.random.uniform(
                    0.2, 0.4
                )  # Moderate ancillary spend
                seasonal = True  # More likely in high season
            elif segment == "Business":
                # Business travelers have more stays but consistent ADR
                stay_count = np.random.poisson(3)  # Average 3 stays
                # At least 1, at most 8
                stay_count = max(1, min(8, stay_count))
                avg_stay_length = np.random.uniform(1.5, 3)  # 1.5-3 nights
                avg_daily_rate = np.random.uniform(120, 200)  # Mid-range rates
                ancillary_spend_ratio = np.random.uniform(
                    0.1, 0.3
                )  # Lower ancillary spend
                repeat_probability = 0.6  # 60% chance of returning
                seasonal = False  # Less seasonal variation
            else:  # Group
                # Group bookings are larger and less frequent
                stay_count = np.random.poisson(1.2)  # Average 1.2 stays
                # At least 1, at most 3
                stay_count = max(1, min(3, stay_count))
                avg_stay_length = np.random.uniform(2, 5)  # 2-5 nights
                avg_daily_rate = np.random.uniform(
                    100, 180
                )  # Lower rates due to volume
                ancillary_spend_ratio = np.random.uniform(
                    0.3, 0.5
                )  # Higher ancillary spend (events)
                repeat_probability = 0.2  # 20% chance of returning
                seasonal = True  # More likely during conference seasons

            # Generate each stay
            current_stay = first_stay
            for stay_num in range(stay_count):
                # Stay length (in nights)
                stay_length = max(1, int(np.random.normal(avg_stay_length, 1)))

                # Room rate with some variability
                base_rate = avg_daily_rate * (1 + np.random.uniform(-0.1, 0.1))

                # If seasonal, apply seasonality factor
                if seasonal:
                    month = current_stay.month
                    # Peak in summer and December
                    if month in [6, 7, 8, 12]:
                        base_rate *= np.random.uniform(1.1, 1.3)
                    # Low in January, February
                    elif month in [1, 2]:
                        base_rate *= np.random.uniform(0.8, 0.9)

                # Ancillary revenue (food, spa, etc.)
                ancillary_revenue = base_rate * stay_length * ancillary_spend_ratio

                # Total revenue
                room_revenue = base_rate * stay_length
                total_revenue = room_revenue + ancillary_revenue

                # Add to stays data
                stays_data.append(
                    {
                        "guest_id": guest_id,
                        "segment": segment,
                        "stay_date": current_stay,
                        "stay_length": stay_length,
                        "daily_rate": base_rate,
                        "room_revenue": room_revenue,
                        "ancillary_revenue": ancillary_revenue,
                        "total_revenue": total_revenue,
                        "stay_number": stay_num + 1,
                    }
                )

                # Calculate next stay date based on repeat probability
                if stay_num < stay_count - 1:
                    # Determine days until next stay based on segment
                    if segment == "Business":
                        # Business travelers return sooner
                        days_until_next = np.random.randint(20, 90)
                    elif segment == "Leisure":
                        # Leisure travelers return less frequently
                        days_until_next = np.random.randint(60, 180)
                    else:  # Group
                        # Groups book further apart
                        days_until_next = np.random.randint(90, 270)

                    current_stay = current_stay + pd.Timedelta(days=days_until_next)

                    # Ensure stay is within data range
                    if current_stay > max_date + pd.Timedelta(days=30):
                        break

        # Convert to DataFrame
        stays_df = pd.DataFrame(stays_data)

        # Calculate CLTV metrics
        guest_metrics = stays_df.groupby("guest_id").agg(
            {
                "stay_date": ["min", "max", "count"],
                "stay_length": "sum",
                "total_revenue": "sum",
                "segment": "first",
            }
        )

        # Flatten the column hierarchy
        guest_metrics.columns = [
            "first_stay",
            "last_stay",
            "stay_count",
            "total_nights",
            "lifetime_value",
            "segment",
        ]

        # Calculate days as customer (customer tenure)
        guest_metrics["days_as_customer"] = (
            guest_metrics["last_stay"] - guest_metrics["first_stay"]
        ).dt.days + 1

        # Calculate average time between stays
        guest_metrics["avg_time_between_stays"] = (
            guest_metrics["days_as_customer"] / guest_metrics["stay_count"]
        )

        # Calculate average revenue per stay
        guest_metrics["avg_revenue_per_stay"] = (
            guest_metrics["lifetime_value"] / guest_metrics["stay_count"]
        )

        # Calculate average daily spend (during stays)
        guest_metrics["avg_daily_spend"] = (
            guest_metrics["lifetime_value"] / guest_metrics["total_nights"]
        )

        # Add prediction for future revenue (simple model based on past behavior)
        # Predict probability of returning in next year based on recency and
        # frequency

        # Recency factor: more recent = higher probability
        today = pd.Timestamp.now()
        guest_metrics["days_since_last_stay"] = (
            today - guest_metrics["last_stay"]
        ).dt.days
        guest_metrics["recency_factor"] = 1 / np.log10(
            guest_metrics["days_since_last_stay"] + 10
        )

        # Frequency factor: more stays = higher probability
        guest_metrics["frequency_factor"] = np.log1p(guest_metrics["stay_count"]) / 2

        # Combine factors to get return probability
        guest_metrics["return_probability"] = (
            guest_metrics["recency_factor"] * guest_metrics["frequency_factor"]
        ).clip(0, 0.95)

        # Predict future value (next 12 months)
        guest_metrics["predicted_stays_next_year"] = guest_metrics[
            "return_probability"
        ] * (guest_metrics["stay_count"] / (guest_metrics["days_as_customer"] / 365))
        guest_metrics["predicted_revenue_next_year"] = (
            guest_metrics["predicted_stays_next_year"]
            * guest_metrics["avg_revenue_per_stay"]
        )

        # Total CLTV (historical + predicted)
        guest_metrics["total_cltv"] = (
            guest_metrics["lifetime_value"]
            + guest_metrics["predicted_revenue_next_year"]
        )

        # Reset index to make guest_id a column
        guest_metrics.reset_index(inplace=True)

        return stays_df, guest_metrics

    # Generate CLTV data
    stays_df, guest_metrics = _generate_guest_cltv_data(base_df)

    def _filter_data(stays_df, guest_metrics, start_date, end_date, segment=None):
        """Filter data by date range and segment."""
        # Filter stays by date
        filtered_stays = stays_df[
            (stays_df["stay_date"] >= start_date) & (stays_df["stay_date"] <= end_date)
        ]

        if segment and segment != "All Guests":
            filtered_stays = filtered_stays[filtered_stays["segment"] == segment]
            # Only include guests in the selected segment
            relevant_guests = guest_metrics[guest_metrics["segment"] == segment]
        else:
            relevant_guests = guest_metrics

        # Only include guests who had stays in the filtered period
        guest_ids_in_period = filtered_stays["guest_id"].unique()
        filtered_guests = relevant_guests[
            relevant_guests["guest_id"].isin(guest_ids_in_period)
        ]

        return filtered_stays, filtered_guests

    def _render():
        """Update the dashboard with filtered data."""
        try:
            # Get selected dates and segment
            start_date = pd.Timestamp(start_picker.date().toPython())
            end_date = pd.Timestamp(end_picker.date().toPython())
            segment = segment_combo.currentText()

            # Filter data
            filtered_stays, filtered_guests = _filter_data(
                stays_df, guest_metrics, start_date, end_date, segment
            )

            # Clear previous content
            while kpi_grid.count():
                item = kpi_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            tabs.clear()

            if filtered_guests.empty:
                empty_label = QLabel("No guest data available for the selected filters")
                empty_label.setStyleSheet("color: red; font-size: 14pt;")
                kpi_grid.addWidget(empty_label, 0, 0, 1, 4, Qt.AlignCenter)
                return

            # Calculate KPIs
            guest_count = len(filtered_guests)
            avg_cltv = filtered_guests["total_cltv"].mean()
            avg_stay_count = filtered_guests["stay_count"].mean()
            avg_nights = (
                filtered_guests["total_nights"].mean()
                / filtered_guests["stay_count"].mean()
            )
            repeat_rate = (filtered_guests["stay_count"] > 1).mean() * 100
            predicted_revenue = filtered_guests["predicted_revenue_next_year"].sum()

            # Add KPIs to grid
            kpis = [
                ("Total Guests", f"{guest_count:,}"),
                ("Average CLTV", f"${avg_cltv:.2f}"),
                ("Average Stays", f"{avg_stay_count:.1f}"),
                ("Average Stay Length", f"{avg_nights:.1f} nights"),
                ("Repeat Rate", f"{repeat_rate:.1f}%"),
                ("Predicted Revenue (12m)", f"${predicted_revenue:,.2f}"),
            ]

            for i, (label, value) in enumerate(kpis):
                col = i % 3
                row = i // 3
                kpi_grid.addWidget(kpi_tile(label, value), row, col)

            # CLTV Distribution Tab
            cltv_tab = QWidget()
            cltv_tab.setLayout(QVBoxLayout())

            # CLTV distribution chart
            cltv_fig = Figure(figsize=(10, 5))
            cltv_ax = cltv_fig.add_subplot(111)

            # Create histogram with KDE
            bins = min(50, len(filtered_guests) // 10)
            cltv_ax.hist(
                filtered_guests["total_cltv"],
                bins=bins,
                alpha=0.7,
                color="#3b82f6",
                density=True,
            )

            # Add vertical line for mean
            cltv_ax.axvline(
                x=avg_cltv, color="red", linestyle="--", label=f"Mean: ${avg_cltv:.2f}"
            )

            # Add labels and legend
            cltv_ax.set_title("Customer Lifetime Value Distribution")
            cltv_ax.set_xlabel("CLTV ($)")
            cltv_ax.set_ylabel("Density")
            cltv_ax.legend()

            # Format x-axis as currency
            cltv_ax.xaxis.set_major_formatter(
                matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
            )

            cltv_fig.tight_layout()
            cltv_canvas = Canvas(cltv_fig)
            cltv_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            cltv_tab.layout().addWidget(cltv_canvas)

            # CLTV by segment/RFM chart
            segment_fig = Figure(figsize=(10, 5))
            segment_ax = segment_fig.add_subplot(111)

            # Calculate average CLTV by segment
            segment_cltv = filtered_guests.groupby("segment")["total_cltv"].agg(
                ["mean", "count"]
            )
            segment_cltv = segment_cltv.sort_values("mean", ascending=False)

            # Create bar chart
            bars = segment_ax.bar(segment_cltv.index, segment_cltv["mean"])

            # Add data labels
            for bar in bars:
                height = bar.get_height()
                segment_ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 20,
                    f"${height:.2f}\n({segment_cltv.loc[bar.get_x(), 'count']:.0f} guests)",
                    ha="center",
                    va="bottom",
                )

            segment_ax.set_title("Average CLTV by Guest Segment")
            segment_ax.set_xlabel("Segment")
            segment_ax.set_ylabel("Average CLTV ($)")

            # Format y-axis as currency
            segment_ax.yaxis.set_major_formatter(
                matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
            )

            segment_fig.tight_layout()
            segment_canvas = Canvas(segment_fig)
            segment_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            cltv_tab.layout().addWidget(segment_canvas)

            tabs.addTab(cltv_tab, "CLTV Analysis")

            # Customer Journey Tab
            journey_tab = QWidget()
            journey_tab.setLayout(QVBoxLayout())

            # Guest spend over time chart
            journey_fig = Figure(figsize=(10, 5))
            journey_ax = journey_fig.add_subplot(111)

            # Group by stay number and calculate average revenue
            stay_metrics = (
                filtered_stays.groupby("stay_number")
                .agg({"total_revenue": "mean", "guest_id": "nunique"})
                .reset_index()
            )

            # Plot average revenue by stay number
            width = 0.35
            x = np.arange(len(stay_metrics))

            # Create twin axis
            journey_ax2 = journey_ax.twinx()

            # Plot bars for revenue
            bars = journey_ax.bar(
                x,
                stay_metrics["total_revenue"],
                width,
                color="#3b82f6",
                label="Avg. Revenue",
            )

            # Plot line for guest count
            journey_ax2.plot(x, stay_metrics["guest_id"], "ro-", label="Guest Count")

            # Add labels and legend
            journey_ax.set_title("Guest Spend by Stay Number")
            journey_ax.set_xlabel("Stay Number")
            journey_ax.set_ylabel("Average Revenue ($)", color="#3b82f6")
            journey_ax.tick_params(axis="y", colors="#3b82f6")
            journey_ax.set_xticks(x)
            journey_ax.set_xticklabels(stay_metrics["stay_number"])

            journey_ax2.set_ylabel("Number of Guests", color="r")
            journey_ax2.tick_params(axis="y", colors="r")

            # Format y-axis as currency
            journey_ax.yaxis.set_major_formatter(
                matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
            )

            # Add combined legend
            lines, labels = journey_ax.get_legend_handles_labels()
            lines2, labels2 = journey_ax2.get_legend_handles_labels()
            journey_ax.legend(lines + lines2, labels + labels2, loc="upper left")

            journey_fig.tight_layout()
            journey_canvas = Canvas(journey_fig)
            journey_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            journey_tab.layout().addWidget(journey_canvas)

            # Guest retention chart
            retention_fig = Figure(figsize=(10, 5))
            retention_ax = retention_fig.add_subplot(111)

            # Calculate retention rates
            max_stay = min(10, stay_metrics["stay_number"].max())
            retention_rate = [
                (
                    (
                        stay_metrics[stay_metrics["stay_number"] == i]["guest_id"].iloc[
                            0
                        ]
                        / stay_metrics[stay_metrics["stay_number"] == 1][
                            "guest_id"
                        ].iloc[0]
                    )
                    * 100
                    if i <= len(stay_metrics)
                    else 0
                )
                for i in range(1, max_stay + 1)
            ]

            retention_ax.bar(
                range(1, len(retention_rate) + 1), retention_rate, color="#10b981"
            )
            retention_ax.set_title("Guest Retention Rate by Stay Number")
            retention_ax.set_xlabel("Stay Number")
            retention_ax.set_ylabel("Retention Rate (%)")
            retention_ax.set_ylim(0, 100)

            for i, rate in enumerate(retention_rate):
                retention_ax.text(i + 1, rate + 2, f"{rate:.1f}%", ha="center")

            retention_fig.tight_layout()
            retention_canvas = Canvas(retention_fig)
            retention_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            journey_tab.layout().addWidget(retention_canvas)

            tabs.addTab(journey_tab, "Customer Journey")

            # Predictive Analysis Tab
            predict_tab = QWidget()
            predict_tab.setLayout(QVBoxLayout())

            # Return probability chart
            predict_fig = Figure(figsize=(10, 5))
            predict_ax = predict_fig.add_subplot(111)

            # Create a scatter plot of frequency vs recency
            scatter = predict_ax.scatter(
                filtered_guests["days_since_last_stay"],
                filtered_guests["stay_count"],
                c=filtered_guests["return_probability"] * 100,
                cmap="viridis",
                alpha=0.7,
                s=50,
            )

            # Add colorbar
            cbar = predict_fig.colorbar(scatter, ax=predict_ax)
            cbar.set_label("Return Probability (%)")

            predict_ax.set_title("Return Probability by Recency and Frequency")
            predict_ax.set_xlabel("Days Since Last Stay (Recency)")
            predict_ax.set_ylabel("Number of Stays (Frequency)")

            predict_fig.tight_layout()
            predict_canvas = Canvas(predict_fig)
            predict_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            predict_tab.layout().addWidget(predict_canvas)

            # Future value prediction chart
            future_fig = Figure(figsize=(10, 5))
            future_ax = future_fig.add_subplot(111)

            # Create boxplot of predicted revenue by segment
            segment_prediction = filtered_guests.groupby("segment")[
                "predicted_revenue_next_year"
            ].apply(list)
            future_ax.boxplot(
                segment_prediction.values, labels=segment_prediction.index
            )

            future_ax.set_title("Predicted Revenue Next 12 Months by Segment")
            future_ax.set_xlabel("Guest Segment")
            future_ax.set_ylabel("Predicted Revenue ($)")

            # Format y-axis as currency
            future_ax.yaxis.set_major_formatter(
                matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
            )

            future_fig.tight_layout()
            future_canvas = Canvas(future_fig)
            future_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            predict_tab.layout().addWidget(future_canvas)

            tabs.addTab(predict_tab, "Predictive Analysis")

        except Exception as e:
            import traceback

            traceback.print_exc()

            # Show error in KPI grid
            error_label = QLabel(f"Error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 14pt;")
            kpi_grid.addWidget(error_label, 0, 0, 1, 4, Qt.AlignCenter)

    # Connect signals
    apply_btn.clicked.connect(_render)
    segment_combo.currentIndexChanged.connect(_render)

    # Initial render
    _render()

    # Expose refresh_date_pickers for data sync
    display.refresh_date_pickers = refresh_date_pickers

    return root
