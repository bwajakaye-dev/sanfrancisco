import pandas as pd
import numpy as np
import os
from src.utils.config import SF_ZIP_CODES, FUEL_TYPE_MAPPING, DATA_RAW, DATA_PROCESSED

def clean_vehicle_data(input_file, output_file):
    """
    Clean and filter CA DMV data for San Francisco ZIP codes.
    
    Args:
        input_file: Path to raw data file
        output_file: Path to save cleaned data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print(f"\n🧹 Cleaning {os.path.basename(input_file)}...")
    
    # Load data
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   Original shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Standardize column names (they may vary by year)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Identify ZIP code column (might be named differently)
    zip_col = None
    for col in df.columns:
        if 'zip' in col.lower():
            zip_col = col
            break
    
    if zip_col is None:
        print("❌ Error: Could not find ZIP code column")
        return None
    
    # Standardize ZIP codes (ensure 5 digits)
    df[zip_col] = df[zip_col].astype(str).str.zfill(5)
    
    # Filter for SF ZIP codes
    df_sf = df[df[zip_col].isin(SF_ZIP_CODES)].copy()
    print(f"   SF filtered shape: {df_sf.shape}")
    
    if len(df_sf) == 0:
        print("⚠️  Warning: No SF ZIP codes found in data!")
        return df_sf
    
    # Identify fuel type column
    fuel_col = None
    for col in df.columns:
        if 'fuel' in col.lower():
            fuel_col = col
            break
    
    if fuel_col:
        # Standardize fuel types
        df_sf['fuel_category'] = df_sf[fuel_col].map(FUEL_TYPE_MAPPING)
        df_sf['fuel_category'] = df_sf['fuel_category'].fillna('Other')
        
        # Add EV flag
        df_sf['is_ev'] = df_sf['fuel_category'].isin(['EV', 'PHEV']).astype(int)
        
        # Show fuel type distribution
        print("\n   Fuel Type Distribution:")
        print(df_sf['fuel_category'].value_counts())
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_sf.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to {output_file}")
    
    return df_sf


def clean_all_years(years=[2024, 2023, 2022]):
    """Clean data for all specified years."""
    print("="*60)
    print("Cleaning Vehicle Data for All Years")
    print("="*60)
    
    for year in years:
        input_file = os.path.join(DATA_RAW, f'ca_dmv_vehicle_{year}.csv')
        output_file = os.path.join(DATA_PROCESSED, f'sf_vehicles_{year}_cleaned.csv')

        if not os.path.exists(input_file):
            print(f"ℹ️  {year}: Raw CA DMV file missing; downloading from data.ca.gov...")
            try:
                from src.data_collection.download_ca_dmv import download_ca_dmv_data

                if not download_ca_dmv_data(years=[year], output_dir=DATA_RAW):
                    print(f"⚠️  Skipping {year}: download failed")
                    continue
            except Exception as e:
                print(f"⚠️  Skipping {year}: download failed ({e})")
                continue
        
        clean_vehicle_data(input_file, output_file)
    
    print("\n" + "="*60)
    print("✅ All cleaning complete!")
    print("="*60)


if __name__ == '__main__':
    clean_all_years()