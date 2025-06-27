# jobs/elasticity_trainer.py
import pandas as pd
import numpy as np
import xgboost as xgb
import redis
import json
from datetime import datetime, timedelta
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from typing import Dict, List, Tuple, Any

def train_elasticity_matrix(year_month: str) -> np.ndarray:
    """
    Trains elasticity models using historical data from Snowflake and
    generates a matrix of elasticity coefficients.
    
    Returns: NumPy array of shape (n_features, n_outputs)
    """
    # Parse year-month and set date range
    dt = datetime.strptime(year_month, "%Y-%m")
    end_date = dt.replace(day=1) + timedelta(days=32)
    end_date = end_date.replace(day=1) - timedelta(days=1)  # Last day of month
    start_date = end_date - timedelta(days=365)  # One year of data
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user="ANALYTICS_USER",
        password="***********",
        account="yourcompany.snowflakecomputing.com",
        warehouse="ANALYTICS_WH",
        database="HOTEL_DATA",
        schema="OPERATIONS"
    )
    
    # Query for historical data
    query = f"""
    SELECT
        date,
        room_type,
        base_rate,
        actual_rate,
        occupancy_percentage,
        revpar,
        goppar,
        total_revenue,
        total_cost,
        net_profit,
        housekeeping_fte,
        fnb_fte,
        promo_spa_discount,
        promo_breakfast_included,
        promo_resort_credit,
        promo_late_checkout
    FROM
        financial_daily
    WHERE
        date BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
    ORDER BY
        date
    """
    
    # Load into pandas
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Get unique room types
    room_types = df['room_type'].unique().tolist()
    
    # Initialize matrix (features x outputs)
    n_features = len(room_types) + 2 + 4  # room types + staff types + promo types
    n_outputs = 5  # RevPAR, GOPPAR, Revenue, Cost, Profit
    elasticity_matrix = np.zeros((n_features, n_outputs), dtype=np.float32)
    
    # Train models and extract elasticities
    
    # 1. Price elasticity (per room type)
    for i, room_type in enumerate(room_types):
        room_df = df[df['room_type'] == room_type]
        
        # Price to demand relationship
        X = room_df[['actual_rate']].values
        y = room_df[['occupancy_percentage', 'revpar', 'goppar', 'total_revenue', 'net_profit']].values
        
        if len(X) < 30:  # Skip if insufficient data
            continue
            
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08)
        model.fit(X, y)
        
        # Calculate elasticity (% change in output / % change in input)
        base_rate = np.median(X)
        base_prediction = model.predict([base_rate])[0]
        
        # Test with 1% increase
        test_rate = base_rate * 1.01
        test_prediction = model.predict([test_rate])[0]
        
        # Calculate elasticities
        elasticities = (test_prediction - base_prediction) / base_prediction / 0.01
        
        # Store in matrix
        elasticity_matrix[i, :] = elasticities
    
    # 2. Staff elasticity
    X_staff = df[['housekeeping_fte', 'fnb_fte']].values
    y_staff = df[['total_cost', 'goppar', 'net_profit']].values
    
    staff_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08)
    staff_model.fit(X_staff, y_staff)
    
    # Calculate elasticities for each staff type
    base_hk = np.median(df['housekeeping_fte'])
    base_fnb = np.median(df['fnb_fte'])
    base_staffing = np.array([[base_hk, base_fnb]])
    base_staff_prediction = staff_model.predict(base_staffing)[0]
    
    # Test housekeeping +1 FTE
    test_hk = np.array([[base_hk + 1, base_fnb]])
    test_hk_prediction = staff_model.predict(test_hk)[0]
    hk_elasticity = (test_hk_prediction - base_staff_prediction) / base_staff_prediction / (1/base_hk)
    
    # Test F&B +1 FTE
    test_fnb = np.array([[base_hk, base_fnb + 1]])
    test_fnb_prediction = staff_model.predict(test_fnb)[0]
    fnb_elasticity = (test_fnb_prediction - base_staff_prediction) / base_staff_prediction / (1/base_fnb)
    
    # Store staff elasticities
    elasticity_matrix[len(room_types), :3] = np.append(hk_elasticity[0], [0, 0])
    elasticity_matrix[len(room_types) + 1, :3] = np.append(fnb_elasticity[0], [0, 0])
    
    # 3. Promotion elasticities
    promo_cols = ['promo_spa_discount', 'promo_breakfast_included', 
                 'promo_resort_credit', 'promo_late_checkout']
    X_promo = df[promo_cols].values
    y_promo = df[['revpar', 'total_revenue', 'total_cost', 'goppar', 'net_profit']].values
    
    promo_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08)
    promo_model.fit(X_promo, y_promo)
    
    # Calculate elasticity for each promo
    base_promos = np.zeros((1, len(promo_cols)))
    base_promo_prediction = promo_model.predict(base_promos)[0]
    
    for i, promo in enumerate(promo_cols):
        test_promo = base_promos.copy()
        test_promo[0, i] = 1  # Enable this promo
        
        test_promo_prediction = promo_model.predict(test_promo)[0]
        promo_elasticity = test_promo_prediction - base_promo_prediction
        
        # Store in matrix (with direct impact, not %)
        elasticity_matrix[len(room_types) + 2 + i, :] = promo_elasticity
    
    # Calculate baseline metrics for the month
    target_month_df = df[df['date'].dt.strftime('%Y-%m') == year_month]
    baseline = {
        "revpar": float(np.mean(target_month_df['revpar'])),
        "goppar": float(np.mean(target_month_df['goppar'])),
        "total_revenue": float(np.sum(target_month_df['total_revenue'])),
        "total_cost": float(np.sum(target_month_df['total_cost'])),
        "net_profit": float(np.sum(target_month_df['net_profit'])),
        "occupancy": float(np.mean(target_month_df['occupancy_percentage'])),
        "staff_fte": {
            "housekeeping": float(np.median(target_month_df['housekeeping_fte'])),
            "fnb": float(np.median(target_month_df['fnb_fte']))
        }
    }
    
    # Store results in Redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Store the elasticity matrix
    r.set(f"ELAST_MATRIX:{year_month}", elasticity_matrix.tobytes())
    
    # Store metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "sample_size": len(df),
        "r_squared": 0.87,  # Example value
        "version": "1.2"
    }
    r.set(f"ELAST_MATRIX:{year_month}:meta", json.dumps(metadata))
    
    # Store room types
    r.delete(f"ELAST_MATRIX:{year_month}:room_types")
    r.rpush(f"ELAST_MATRIX:{year_month}:room_types", *room_types)
    
    # Store baseline values
    r.set(f"ELAST_MATRIX:{year_month}:baseline", json.dumps(baseline))
    
    print(f"Elasticity matrix for {year_month} saved to Redis")
    return elasticity_matrix

if __name__ == "__main__":
    # Train for current month
    current_month = datetime.now().strftime("%Y-%m")
    train_elasticity_matrix(current_month)
    
    # Also train for next month (for forecasting)
    next_month = (datetime.now().replace(day=1) + timedelta(days=32)).strftime("%Y-%m")
    train_elasticity_matrix(next_month)