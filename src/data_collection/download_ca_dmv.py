import os
from typing import Optional

import pandas as pd
import requests

from src.utils.config import (
    CA_DMV_CKAN_API_BASE,
    CA_DMV_CKAN_PACKAGE_ID,
    CA_DMV_FALLBACK_DOWNLOAD_URLS,
    CA_DMV_RESOURCE_IDS,
    DATA_RAW,
)


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ('.', '-', '_'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep).strip('_')


def _download_to_file(url: str, output_path: str, timeout_s: int = 60) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _ckan_package_show(package_id: str, timeout_s: int = 30) -> dict:
    url = f"{CA_DMV_CKAN_API_BASE}/package_show"
    resp = requests.get(url, params={"id": package_id}, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {payload}")
    return payload["result"]


def _resolve_ca_dmv_csv_url(year: int) -> str:
    """Resolve a year-specific CSV download URL from the CKAN dataset metadata.

    Falls back to a known direct download URL if metadata lookup fails.
    """
    try:
        pkg = _ckan_package_show(CA_DMV_CKAN_PACKAGE_ID)
        resources = pkg.get("resources", [])
        wanted_resource_id = CA_DMV_RESOURCE_IDS.get(year)

        if wanted_resource_id:
            for r in resources:
                if r.get("id") == wanted_resource_id and r.get("url"):
                    return r["url"]

        year_s = str(year)
        for r in resources:
            haystack = " ".join(
                str(v or "")
                for v in (r.get("name"), r.get("description"), r.get("url"), r.get("format"))
            ).lower()
            if year_s in haystack and "csv" in haystack and r.get("url"):
                return r["url"]
    except Exception:
        pass

    fallback = CA_DMV_FALLBACK_DOWNLOAD_URLS.get(year)
    if not fallback:
        raise KeyError(
            f"No CA DMV download URL known for year {year}. "
            f"Update CA_DMV_RESOURCE_IDS/CA_DMV_FALLBACK_DOWNLOAD_URLS in src/utils/config.py."
        )
    return fallback

def download_ca_dmv_data(years=[2024, 2023, 2022], output_dir=DATA_RAW, force: bool = False):
    """
    Download CA DMV vehicle fuel type data for specified years.
    
    Args:
        years: List of years to download
        output_dir: Directory to save downloaded files
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("="*60)
    print("Downloading CA DMV Vehicle Data")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for year in years:
        output_path = os.path.join(output_dir, f"ca_dmv_vehicle_{year}.csv")

        if os.path.exists(output_path) and not force:
            print(f"\n✅ {year}: already cached at {output_path}")
            continue

        print(f"\n📥 Downloading {year} CA DMV data from data.ca.gov...")

        try:
            url = _resolve_ca_dmv_csv_url(year)
            print(f"   URL: {url}")
            tmp_path = os.path.join(output_dir, f".tmp_ca_dmv_{year}_{_safe_filename(os.path.basename(url)) or 'data'}.csv")
            _download_to_file(url, tmp_path)
            os.replace(tmp_path, output_path)

            preview = pd.read_csv(output_path, low_memory=False, nrows=5)
            print(f"✅ Success! Saved to {output_path}")
            print(f"   Preview columns: {list(preview.columns)}")

        except Exception as e:
            print(f"❌ Error downloading {year} data: {e}")
            return False
    
    print("\n" + "="*60)
    print("✅ All downloads complete!")
    print("="*60)
    return True


if __name__ == '__main__':
    download_ca_dmv_data()