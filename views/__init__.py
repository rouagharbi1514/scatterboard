# flake8: noqa
"""
Views package for the hotel dashboard.
Contains all visualization modules.
"""

import traceback
from .forecast_scatter import ForecastScatterView

# Import existing modules with try/except to handle optional components
try:
    from .marketing import display as marketing_display_imported
except ImportError:
    marketing_display_imported = None

try:
    from .company import display as company_display_imported
except ImportError:
    company_display_imported = None

try:
    from .operations import display_fb
except ImportError:
    display_fb = None

try:
    from .housekeeping import display as housekeeping_display_imported
except ImportError:
    housekeeping_display_imported = None

try:
    from .profitability import (
        display_room_profit,
        display_room_type_profitability
    )
except ImportError:
    display_room_profit = None
    display_room_type_profitability = None

try:
    from .guests import display_facilities_usage as guests_facilities_usage_imported
    from .guests import display_cancellation_analysis as guests_cancellation_analysis_imported
    from .guests import display_age_analysis as guests_age_analysis_imported
    from .guests import display_preferences as guests_preferences_display_imported
    from .guests import display as guests_display_imported
except ImportError:
    guests_facilities_usage_imported = None
    guests_cancellation_analysis_imported = None
    guests_age_analysis_imported = None
    guests_preferences_display_imported = None
    guests_display_imported = None

try:
    from .room_cost import display as room_cost_display
except ImportError:
    room_cost_display = None

try:
    from .seasonality import display as seasonality_display
except ImportError:
    seasonality_display = None

try:
    from .revenue import display as revenue_display
except ImportError:
    revenue_display = None

try:
    from .kpis import display as kpis_display
except ImportError:
    kpis_display = None

try:
    from .overview import display as overview_display
    from . import overview
except ImportError:
    overview_display = None
    overview = None

# Import all the modules that were moved to the end - move them here
try:
    from . import operations
except ImportError:
    operations = None

try:
    from . import pricing
except ImportError:
    pricing = None

try:
    from . import ai_views
except ImportError:
    ai_views = None

try:
    from . import feedback
except ImportError:
    feedback = None

try:
    from . import cltv
except ImportError:
    cltv = None

try:
    from . import scenarios
except ImportError:
    scenarios = None

try:
    from . import what_if
except ImportError:
    what_if = None

try:
    from . import cancellations
except ImportError:
    cancellations = None

try:
    from . import storytelling
except ImportError:
    storytelling = None

try:
    from . import advanced
except ImportError:
    advanced = None

try:
    from . import retention
except ImportError:
    retention = None

try:
    from . import upselling
except ImportError:
    upselling = None

# Create a placeholder module class
class PlaceholderModule:
    """Placeholder for modules that aren't implemented yet."""

    @staticmethod
    def display(*args, **kwargs):
        """Placeholder function for views that aren't implemented yet."""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel("This view is not implemented yet.")
        layout.addWidget(label)
        return widget


# Create placeholder display function
def placeholder_display(*args, **kwargs):
    """Placeholder function for views that aren't implemented yet."""
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel("This view is not implemented yet.")
    layout.addWidget(label)
    return widget


# Assign the final display functions
marketing_display = marketing_display_imported or placeholder_display
company_display = company_display_imported or placeholder_display
housekeeping_display = housekeeping_display_imported or placeholder_display
guests_display = guests_display_imported or placeholder_display
guests_preferences_display = guests_preferences_display_imported or placeholder_display
guests_age_analysis = guests_age_analysis_imported or placeholder_display
guests_cancellation_analysis = guests_cancellation_analysis_imported or placeholder_display
guests_facilities_usage = guests_facilities_usage_imported or placeholder_display

# Operations displays
operations_fb_display = display_fb or PlaceholderModule.display

# Remove unused import and fix undefined name errors
from connectors.local_server_connector import read_data_files


# Define read_csv function
def read_csv(*args, **kwargs):
    """Read CSV files."""
    return read_data_files(*args, **kwargs)


# Replace the FileUploadModule class with a direct function
def file_upload_display():
    """Display file upload interface."""
    from PySide6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QPushButton,
        QFileDialog,
        QLabel,
        QProgressBar,
        QMessageBox,
        QGroupBox,
        QGridLayout,
        QComboBox,
    )
    import pandas as pd
    import data

    root = QWidget()
    root.setLayout(QVBoxLayout())

    # Title
    title = QLabel("File Upload")
    title.setStyleSheet("font-size: 18pt; font-weight: bold;")
    root.layout().addWidget(title)

    # Instructions
    instructions_text = (
        "Upload your hotel data file. Supported formats: "
        "CSV, Excel (XLS, XLSX).\n\n"
        "After uploading, you can map your columns to the required fields."
    )
    instructions = QLabel(instructions_text)
    instructions.setWordWrap(True)
    instructions.setStyleSheet(
        "font-size: 11pt; color: #666; margin-bottom: 15px;"
    )
    root.layout().addWidget(instructions)

    # File Upload Section
    upload_group = QGroupBox("Upload Data File")
    upload_group.setLayout(QVBoxLayout())
    root.layout().addWidget(upload_group)

    upload_btn = QPushButton("Select Data File")
    upload_btn.setStyleSheet(
        """
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: none;
            padding: 8px 16px;
            font-weight: bold;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #0b5ed7;
        }
    """
    )
    upload_group.layout().addWidget(upload_btn)

    status_label = QLabel("No file selected")
    status_label.setStyleSheet("color: #666; margin-top: 10px;")
    upload_group.layout().addWidget(status_label)

    progress = QProgressBar()
    progress.setVisible(False)
    upload_group.layout().addWidget(progress)

    # Column Mapping Section (initially hidden)
    mapping_group = QGroupBox("Map Your Columns")
    mapping_group.setLayout(QVBoxLayout())
    mapping_group.setVisible(False)
    root.layout().addWidget(mapping_group)

    mapping_grid = QGridLayout()
    mapping_group.layout().addLayout(mapping_grid)

    mapping_combos = {}
    loaded_df = None

    # Add Load Data button for after mapping
    load_data_btn = QPushButton("Load Data with Selected Mappings")
    load_data_btn.setStyleSheet(
        """
        QPushButton {
            background-color: #198754;
            color: white;
            border: none;
            padding: 8px 16px;
            font-weight: bold;
            border-radius: 4px;
            margin-top: 15px;
        }
        QPushButton:hover {
            background-color: #157347;
        }
    """
    )
    load_data_btn.setVisible(False)
    mapping_group.layout().addWidget(load_data_btn)

    # Sample Data Section
    sample_group = QGroupBox("Demo Data")
    sample_group.setLayout(QVBoxLayout())
    root.layout().addWidget(sample_group)

    sample_btn = QPushButton("Load Demo Data")
    sample_btn.setStyleSheet(
        """
        QPushButton {
            background-color: #6c757d;
            color: white;
            border: none;
            padding: 8px 16px;
            font-weight: bold;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
    """
    )
    sample_group.layout().addWidget(sample_btn)

    # Function to setup the column mapping UI
    def setup_mapping(df):
        nonlocal loaded_df
        loaded_df = df.copy()

        # Clear previous mappings
        while mapping_grid.count():
            item = mapping_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        mapping_combos.clear()

        # Required field mappings with descriptions
        required_fields = {
            "date": "Date column (YYYY-MM-DD format)",
            "room_type": "Room type/category names",
            "occupancy": "Occupancy rate (0-1 or percentage)",
            "rate": "Average daily rate / ADR",
        }

        df_columns = [""] + list(df.columns)

        # Add headers
        mapping_grid.addWidget(QLabel("Required Field"), 0, 0)
        mapping_grid.addWidget(QLabel("Map to Your Column"), 0, 1)
        mapping_grid.addWidget(QLabel("Description"), 0, 2)

        # Add mapping rows
        row = 1
        for field, description in required_fields.items():
            # Find best match for auto-selection
            best_match = ""
            for col in df.columns:
                if field.lower() in col.lower() or col.lower() in field.lower():
                    best_match = col
                    break
                # Special case for date
                if field == "date" and (
                    "date" in col.lower()
                    or "day" in col.lower()
                    or "month" in col.lower()
                ):
                    best_match = col
                    break
                # Special case for room_type
                if field == "room_type" and (
                    "room" in col.lower()
                    or "type" in col.lower()
                    or "category" in col.lower()
                ):
                    best_match = col
                    break
                # Special case for occupancy
                if field == "occupancy" and (
                    "occup" in col.lower()
                    or "fill" in col.lower()
                    or "util" in col.lower()
                ):
                    best_match = col
                    break
                # Special case for rate
                if field == "rate" and (
                    "rate" in col.lower()
                    or "adr" in col.lower()
                    or "price" in col.lower()
                    or "revenue" in col.lower()
                ):
                    best_match = col
                    break

            # Add field label
            field_label = QLabel(field)
            field_label.setStyleSheet("font-weight: bold;")
            mapping_grid.addWidget(field_label, row, 0)

            # Add dropdown
            combo = QComboBox()
            combo.addItems(df_columns)
            if best_match and best_match in df_columns:
                combo.setCurrentText(best_match)
            mapping_grid.addWidget(combo, row, 1)

            # Store combo for later access
            mapping_combos[field] = combo

            # Add description
            mapping_grid.addWidget(QLabel(description), row, 2)

            row += 1

        # Show the mapping section and load button
        mapping_group.setVisible(True)
        load_data_btn.setVisible(True)

    # Function to handle file upload
    def handle_upload():
        file_path, _ = QFileDialog.getOpenFileName(
            root,
            "Select Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls);;"
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )

        if not file_path:
            return

        try:
            status_label.setText(f"Loading {file_path}...")
            progress.setVisible(True)
            progress.setValue(10)

            # Load file based on extension
            file_ext = file_path.lower().split(".")[-1]

            if file_ext == "csv":
                # Try different encodings if one fails
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding="latin1")
            elif file_ext in ["xlsx", "xls"]:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            progress.setValue(50)

            # Show preview of the data
            status_label.setText(
                f"File loaded. {len(df)} rows, {len(df.columns)} columns."
            )

            # Setup column mapping
            setup_mapping(df)

        except Exception as e:
            import traceback
            traceback.print_exc()
            progress.setVisible(False)
            status_label.setText("Upload failed with error")
            QMessageBox.critical(root, "Error", f"An error occurred: {str(e)}")
        finally:
            progress.setVisible(False)

    # Modify the process_mappings function in file_upload_display
    def process_mappings():
        import traceback
        if loaded_df is None:
            return

        try:
            progress.setVisible(True)
            progress.setValue(10)

            # Create a copy of the dataframe to modify
            df = loaded_df.copy()

            # Apply mappings
            mapped_df = pd.DataFrame()

            # First, check if all required mappings are selected
            all_mappings_selected = True
            missing_fields = []

            for field, combo in mapping_combos.items():
                source_col = combo.currentText()
                if not source_col:
                    all_mappings_selected = False
                    missing_fields.append(field)

            if not all_mappings_selected:
                QMessageBox.warning(
                    root,
                    "Missing Mappings",
                    f"Please select mappings for these required fields: "
                    f"{', '.join(missing_fields)}"
                )
                progress.setVisible(False)
                return

            # Now apply mappings
            for field, combo in mapping_combos.items():
                source_col = combo.currentText()

                # Copy data from source column to target field
                mapped_df[field] = df[source_col]

                # Special handling for date column
                if field == "date":
                    try:
                        mapped_df[field] = pd.to_datetime(
                            mapped_df[field],
                            errors='coerce'
                        )
                        # Check if we got valid dates
                        if mapped_df[field].isna().all():
                            raise ValueError("Could not parse any valid dates")
                    except Exception as e:
                        print(f"Error parsing dates: {e}")
                        # Try different date format patterns
                        date_formats = [
                            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
                            "%Y/%m/%d", "%m-%d-%Y"
                        ]
                        for fmt in date_formats:
                            try:
                                mapped_df[field] = pd.to_datetime(
                                    mapped_df[field],
                                    format=fmt,
                                    errors='coerce'
                                )
                                if not mapped_df[field].isna().all():
                                    print(f"Success with format {fmt}")
                                    break
                            except Exception:
                                continue

                # Special handling for occupancy (convert % to decimal)
                if field == "occupancy" and mapped_df[field].max() > 1:
                    mapped_df[field] = mapped_df[field] / 100

            progress.setValue(40)

            # Ensure 'Date' column for filtering
            mapped_df["Date"] = mapped_df["date"]

            # Copy original data columns that might be useful
            for col in df.columns:
                selected_columns = [
                    mapping_combos[field].currentText() 
                    for field in mapping_combos
                ]
                if col not in mapped_df.columns and col not in selected_columns:
                    mapped_df[col] = df[col]

            progress.setValue(80)
            print("Mapped dataframe preview:")
            print(mapped_df.head())
            print("Columns:", mapped_df.columns.tolist())

            # Debug the data types to identify potential issues
            print("Data types:")
            print(mapped_df.dtypes)

            # Check if any required columns have NaN values
            has_nulls = False
            for field in mapping_combos.keys():
                if mapped_df[field].isna().any():
                    null_count = mapped_df[field].isna().sum()
                    print(f"Warning: Column '{field}' has {null_count} null values")
                    has_nulls = True

            if has_nulls:
                print("Note: Nulls found in required columns")

            # Load into app using adapt_data_for_views
            from data import adapt_data_for_views, load_dataframe
            try:
                # Make sure we have the required columns
                for field in ["date", "room_type", "occupancy", "rate"]:
                    if field not in mapped_df.columns:
                        raise ValueError(
                            f"Required column '{field}' is missing after mapping"
                        )

                # Convert occupancy to decimal if it's in percentage
                if mapped_df["occupancy"].max() > 1:
                    mapped_df["occupancy"] = mapped_df["occupancy"] / 100

                # Apply data adaptation for derived columns
                adapted_df = adapt_data_for_views(mapped_df)
                if adapted_df is None:
                    raise ValueError("Data adaptation failed - returned None")

                # Verify the adapted dataframe has required columns
                missing_cols = []
                for col in ["date", "room_type", "occupancy", "rate"]:
                    if col not in adapted_df.columns:
                        missing_cols.append(col)

                if missing_cols:
                    raise ValueError(
                        f"Adapted dataframe missing required columns: {missing_cols}"
                    )

                # Load the adapted dataframe
                print("Loading dataframe with columns:", adapted_df.columns.tolist())
                success = load_dataframe(adapted_df)
                if not success:
                    raise ValueError("Failed to load dataframe")

            except Exception as e:
                print(f"Error during data mapping: {str(e)}")
                traceback.print_exc()
                success = False

            if success:
                status_label.setText(f"Successfully loaded {len(mapped_df)} rows")
                QMessageBox.information(
                    root,
                    "Upload successful",
                    f"Successfully loaded {len(mapped_df)} rows "
                    "with your column mappings.",
                )

                # Refresh date pickers in all relevant views
                import views

                for view_name in dir(views):
                    view = getattr(views, view_name)
                    if hasattr(view, "refresh_date_pickers"):
                        try:
                            view.refresh_date_pickers()
                        except BaseException:
                            pass
            else:
                status_label.setText("Upload failed")
                QMessageBox.warning(
                    root,
                    "Upload failed",
                    "Failed to load the data with your mappings."
                )
        except Exception as e:
            traceback.print_exc()
            progress.setVisible(False)
            status_label.setText("Data processing failed with error")
            QMessageBox.critical(root, "Error", f"An error occurred: {str(e)}")
        finally:
            progress.setVisible(False)

    # Function to load demo data
    def load_demo_data():
        try:
            progress.setVisible(True)
            progress.setValue(30)

            # Load demo data
            success = data.load_demo_data()
            progress.setValue(100)

            if success:
                status_label.setText("Successfully loaded demo data")
                QMessageBox.information(
                    root, "Demo data loaded", "Successfully loaded demo data"
                )

                # Refresh date pickers in all relevant views
                import views

                for view_name in dir(views):
                    view = getattr(views, view_name)
                    if hasattr(view, "refresh_date_pickers"):
                        try:
                            view.refresh_date_pickers()
                        except BaseException:
                            pass
            else:
                status_label.setText("Failed to load demo data")
                QMessageBox.warning(root, "Load failed", "Failed to load demo data")
        except Exception as e:
            progress.setVisible(False)
            status_label.setText("Error loading demo data")
            QMessageBox.critical(root, "Error", f"An error occurred: {str(e)}")
        finally:
            progress.setVisible(False)

    # Connect buttons
    upload_btn.clicked.connect(handle_upload)
    load_data_btn.clicked.connect(process_mappings)
    sample_btn.clicked.connect(load_demo_data)

    return root


# Replace the placeholder file_upload with the functional one
file_upload = file_upload_display

# Define all the display functions that rely on imported modules
# Make these assignments AFTER all imports but BEFORE using them
pricing_display = pricing.display if pricing else PlaceholderModule.display
ml_pricing_display = ai_views.display_ml_pricing if ai_views else PlaceholderModule.display
ml_segmentation_display = ai_views.show_ml_segmentation if ai_views else PlaceholderModule.display
feedback_display = feedback.display if feedback else PlaceholderModule.display
retention_display = retention.display if retention else PlaceholderModule.display
cltv_display = cltv.display if cltv else PlaceholderModule.display
upselling_display = upselling.display if upselling else PlaceholderModule.display
operations_efficiency_display = operations.display_efficiency if operations else PlaceholderModule.display
operations_custom_charts_display = operations.display_custom_charts if operations else PlaceholderModule.display
scenarios_display = scenarios.display if scenarios else PlaceholderModule.display
what_if_display = what_if.display if what_if else PlaceholderModule.display
cancellations_display = cancellations.display if cancellations else PlaceholderModule.display
storytelling_display = storytelling.display if storytelling else PlaceholderModule.display
advanced_display = advanced.display if advanced else PlaceholderModule.display
advanced_dig_deeper_display = advanced.display_dig_deeper if advanced else PlaceholderModule.display


# Define the missing get_view_function that's used in revenue.py
def get_view_function(view_name):
    """Return a view function based on its name."""
    # Import here to avoid circular imports
    import views

    # Map view names to their corresponding functions
    view_functions = {
        "revenue_analysis": views.revenue_display,
        "overview": views.overview_display,
        "kpis": views.kpis_display,
        "seasonality": views.seasonality_display,
        "room_cost": views.room_cost_display,
        "room_profit": views.display_room_profit,
        "room_type_profitability": views.display_room_type_profitability,
        "guests": views.guests_display,
        "guests_preferences": views.guests_preferences_display,
        "guests_age_analysis": views.guests_age_analysis,
        "guests_cancellation_analysis": views.guests_cancellation_analysis,
        "guests_facilities_usage": views.guests_facilities_usage,
        "file_upload": views.file_upload_display,
        "pricing": views.pricing_display,
        "ml_pricing": views.ml_pricing_display,
        "ml_segmentation": views.ml_segmentation_display,
        "feedback": views.feedback_display,
        "retention": views.retention_display,
        "cltv": views.cltv_display,
        "upselling": views.upselling_display,
        "housekeeping": views.housekeeping_display,
        "operations_fb": views.operations_fb_display,
        "operations_efficiency": views.operations_efficiency_display,
        "operations_custom_charts": views.operations_custom_charts_display,
        "scenarios": views.scenarios_display,
        "what_if": views.what_if_display,
        "cancellations": views.cancellations_display,
        "company": views.company_display,
        "marketing": views.marketing_display,
        "storytelling": views.storytelling_display,
        "advanced": views.advanced_display,
        "advanced_dig_deeper": views.advanced_dig_deeper_display,
    }

    # Return the function if it exists, otherwise return None
    return view_functions.get(view_name)


__all__ = [
    "overview_display",
    "kpis_display",
    "revenue_display",
    "seasonality_display",
    "room_cost_display",
    "display_room_profit",
    "display_room_type_profitability",
    "guests_display",
    "guests_preferences_display",
    "guests_age_analysis",
    "guests_cancellation_analysis",
    "guests_facilities_usage",
    "file_upload_display",
    "pricing_display",
    "ml_pricing_display",
    "ml_segmentation_display",
    "feedback_display",
    "retention_display",
    "cltv_display",
    "upselling_display",
    "housekeeping_display",
    "operations_fb_display",
    "operations_efficiency_display",
    "operations_custom_charts_display",
    "scenarios_display",
    "what_if_display",
    "cancellations_display",
    "company_display",
    "marketing_display",
    "storytelling_display",
    "advanced_display",
    "advanced_dig_deeper_display",
    "ForecastScatterView",
]