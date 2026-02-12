import pandas as pd
import numpy as np
import logging
import os
import json

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataValidator:
    """Handles basic data validation for the pipeline."""
    @staticmethod
    def validate_columns(df, required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

def generate_dummy_data(file_path):
    """Generates a dummy CSV file for demonstration."""
    logging.info(f"Generating dummy data at {file_path}...")
    
    np.random.seed(42)
    data = {
        'transaction_id': range(1, 101),
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Toys'], 100),
        'amount': np.random.uniform(5.0, 1000.0, 100).round(2),
        'customer_id': np.random.randint(1000, 1100, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues: Nulls and outliers
    df.loc[np.random.choice(df.index, 8), 'amount'] = np.nan
    df.loc[0, 'amount'] = 50000.0  # An intentional outlier
    
    df.to_csv(file_path, index=False)
    logging.info("Dummy data generated successfully.")

def run_data_quality_report(df):
    """Performs basic data quality checks and returns a report."""
    logging.info("Running Data Quality Checks...")
    report = {
        'total_rows': len(df),
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'amount_outliers': len(df[df['amount'] > 10000]) # Basic outlier detection
    }
    
    for col, pct in report['null_percentages'].items():
        if pct > 0:
            logging.warning(f"Data Quality Alert: Column '{col}' has {pct:.1f}% null values.")
            
    return report

def run_pipeline(input_file, output_base_name):
    """
    Enhanced E-T-L Pipeline with Validation and Quality Checks.
    """
    try:
        # 1. EXTRACT
        logging.info(f"EXTRACT: Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # 2. VALIDATE
        DataValidator.validate_columns(df, ['transaction_id', 'date', 'amount', 'product_category'])
        
        # 3. TRANSFORM
        logging.info("TRANSFORM: Starting data transformations...")
        
        # Quality Check Step
        dq_report = run_data_quality_report(df)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean: Handle missing values in 'amount' using median (more robust to outliers)
        if dq_report['null_counts']['amount'] > 0:
            median_val = df['amount'].median()
            logging.info(f"Filling nulls in 'amount' with median value: {median_val}")
            df['amount'] = df['amount'].fillna(median_val)
            
        # Process Category Metrics
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        agg_df = df.groupby('product_category').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        
        agg_df.columns = [
            'product_category', 
            'total_revenue', 
            'avg_transaction_value', 
            'transaction_count', 
            'unique_customers'
        ]
        
        # Load Final Summary
        summary = agg_df.round(2).sort_values(by='total_revenue', ascending=False)
        
        # 4. LOAD (Multi-format)
        logging.info(f"LOAD: Writing results to {output_base_name} formats...")
        
        # Save as CSV
        summary.to_csv(f"{output_base_name}.csv", index=False)
        
        # Save as JSON (Pretty printed)
        summary.to_json(f"{output_base_name}.json", orient='records', indent=4)
        
        logging.info("Pipeline execution finished successfully.")
        return summary

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    input_path = "raw_transactions.csv"
    output_base = "enhanced_category_report"
    
    if not os.path.exists(input_path):
        generate_dummy_data(input_path)
    
    final_report = run_pipeline(input_path, output_base)
    
    print("\n--- ENHANCED PIPELINE SUMMARY ---")
    print(final_report)
    print("---------------------------------")

