import pandas as pd
import requests
import os
from src.utils.config import DATA_RAW

def download_sf_parking_data(limit=50000, output_dir=DATA_RAW):
    """
    Download SF parking meter data using Socrata API.
    
    Args:
        limit: Maximum number of records to download
        output_dir: Directory to save data
    
    Returns:
        bool: True if successful
    """
    print("\n📥 Downloading SF Parking Meter data...")

    output_path = os.path.join(output_dir, 'sf_parking_meters.csv')
    if os.path.exists(output_path):
        print(f"ℹ️  Using existing file: {output_path}")
        return True
    
    # Socrata API endpoint for Parking Meters dataset
    url = "https://data.sfgov.org/resource/8vzz-qzz9.json"
    params = {'$limit': limit}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        df.to_csv(output_path, index=False)
        
        print(f"✅ Success! Saved to {output_path}")
        print(f"   Records: {len(df):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == '__main__':
    download_sf_parking_data()