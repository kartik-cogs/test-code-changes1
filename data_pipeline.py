import pandas as pd
import numpy as np
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_dummy_data(file_path):
    """Generates a dummy CSV file for demonstration."""
    logging.info(f"Generating dummy data at {file_path}...")
    
    np.random.seed(42) # For reproducibility
    data = {
        'transaction_id': range(1, 101),
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D').strftime('%Y-%m-%d'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Toys'], 100),
        'amount': np.random.uniform(5.0, 1000.0, 100).round(2),
        'customer_id': np.random.randint(1000, 1100, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Add some nulls for cleaning demonstration
    df.loc[np.random.choice(df.index, 5), 'amount'] = np.nan
    
    df.to_csv(file_path, index=False)
    logging.info("Dummy data generated successfully.")

def run_pipeline(input_file, output_file):
    """
    Simple E-T-L Pipeline:
    Extract: Read CSV
    Transform: Date conversion, Data cleaning (NaNs), Aggregations
    Load: Write summary CSV
    """
    try:
        # 1. EXTRACT
        logging.info(f"EXTRACT: Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # 2. TRANSFORM
        logging.info("TRANSFORM: Starting data transformations...")
        
        # Convert date column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Clean: Handle missing values in 'amount'
        missing_count = df['amount'].isna().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values in 'amount'. Filling with column mean.")
            mean_amount = df['amount'].mean()
            df['amount'] = df['amount'].fillna(mean_amount)
            
        # Feature Engineering: Extract month from date
        df['month'] = df['date'].dt.strftime('%Y-%m')
        
        # Aggregate: Performance summary by category
        # - Total Revenue
        # - Average Transaction Value
        # - Number of Transactions
        # - Unique Customers reached
        agg_df = df.groupby('product_category').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).reset_index()
        
        # Flatten Multi-index columns
        agg_df.columns = [
            'product_category', 
            'total_revenue', 
            'avg_transaction_value', 
            'transaction_count', 
            'unique_customers'
        ]
        
        # Round numeric values for cleanliness
        agg_df = agg_df.round(2)
        
        # Filter: Keep only high-performing categories (Revenue > 5000 as example)
        # Note: With 100 rows and ~500 avg, total is ~50k. Total categories=5, so ~10k each.
        threshold = 10000
        filtered_df = agg_df[agg_df['total_revenue'] > threshold].sort_values(by='total_revenue', ascending=False)
        
        logging.info(f"TRANSFORM: Processing complete. Filtered {len(agg_df)} categories down to {len(filtered_df)}.")
        
        # 3. LOAD
        logging.info(f"LOAD: Writing results to {output_file}...")
        filtered_df.to_csv(output_file, index=False)
        logging.info("Pipeline execution finished successfully.")
        
        return filtered_df

    except Exception as e:
        logging.error(f"Pipeline failed during execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Define file paths
    input_path = "raw_transactions.csv"
    output_path = "category_performance_report.csv"
    
    # Step 0: Ensure we have data to work with
    if not os.path.exists(input_path):
        generate_dummy_data(input_path)
    
    # Step 1: Run the ETL pipeline
    report = run_pipeline(input_path, output_path)
    
    # Preview the results
    print("\n--- DATA ENGINEERING PIPELINE SUMMARY ---")
    print(report)
    print("------------------------------------------")
    print(f"Summary saved to: {os.path.abspath(output_path)}")
