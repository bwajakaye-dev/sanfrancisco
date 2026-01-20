import pandas as pd
import os
from src.utils.config import DATA_PROCESSED

def create_aggregated_features(years=[2024, 2023, 2022], output_file=None):
    """
    Create aggregated features by ZIP code and year.
    
    Args:
        years: List of years to process
        output_file: Path to save output (default: sf_ev_features.csv)
    
    Returns:
        pd.DataFrame: Aggregated features
    """
    print("="*60)
    print("Creating Aggregated Features")
    print("="*60)
    
    dfs = []
    
    for year in years:
        input_file = os.path.join(DATA_PROCESSED, f'sf_vehicles_{year}_cleaned.csv')
        
        if not os.path.exists(input_file):
            print(f"⚠️  Skipping {year}: File not found ({input_file})")
            continue
        
        print(f"\n📊 Processing {year}...")
        df = pd.read_csv(input_file)
        df['year'] = year
        dfs.append(df)
    
    if not dfs:
        print("❌ No data to process!")
        return None
    
    # Combine all years
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Combined data shape: {df_all.shape}")
    
    # Find column names (they may vary)
    zip_col = [col for col in df_all.columns if 'zip' in col.lower()][0]
    
    # Try to find count/vehicles column
    count_col = None
    for col in ['vehicles', 'count', 'total']:
        if col in df_all.columns:
            count_col = col
            break
    
    if count_col is None:
        # Assume each row is one vehicle
        df_all['vehicles'] = 1
        count_col = 'vehicles'
    
    # Aggregate by ZIP and year
    print("\n📈 Calculating aggregate statistics...")
    
    agg_dict = {
        count_col: 'sum',
        'is_ev': lambda x: (x * df_all.loc[x.index, count_col]).sum()
    }
    
    agg_df = df_all.groupby([zip_col, 'year']).agg(agg_dict).reset_index()
    
    # Calculate total vehicles and EV percentage
    agg_df['total_vehicles'] = agg_df[count_col]
    agg_df['ev_count'] = agg_df['is_ev']
    agg_df['ev_percentage'] = (agg_df['ev_count'] / agg_df['total_vehicles'] * 100).round(2)
    
    # Calculate year-over-year growth
    agg_df = agg_df.sort_values([zip_col, 'year'])
    agg_df['ev_growth'] = agg_df.groupby(zip_col)['ev_percentage'].diff().round(2)
    
    # Clean up columns
    agg_df = agg_df[[zip_col, 'year', 'total_vehicles', 'ev_count', 'ev_percentage', 'ev_growth']]
    agg_df.columns = ['zip_code', 'year', 'total_vehicles', 'ev_count', 'ev_percentage', 'ev_growth']
    
    print("\n📋 Feature Summary:")
    print(f"   ZIP codes: {agg_df['zip_code'].nunique()}")
    print(f"   Years: {sorted(agg_df['year'].unique())}")
    print(f"   Avg EV %: {agg_df['ev_percentage'].mean():.2f}%")
    print(f"   Max EV %: {agg_df['ev_percentage'].max():.2f}%")
    
    # Save
    if output_file is None:
        output_file = os.path.join(DATA_PROCESSED, 'sf_ev_features.csv')
    
    agg_df.to_csv(output_file, index=False)
    print(f"\n✅ Features saved to {output_file}")
    
    return agg_df


if __name__ == '__main__':
    create_aggregated_features()