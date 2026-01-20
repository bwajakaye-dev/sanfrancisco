import os


def _pick_existing(*candidates: str) -> str:
    """Return the first path that exists; otherwise return the first candidate."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Data collection paths (CSV sources live here)
DATA_RAW = os.path.join(PROJECT_ROOT, 'src', 'data_collection', 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'src', 'data_collection', 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# San Francisco ZIP codes
SF_ZIP_CODES = [
    '94102', '94103', '94104', '94105', '94107', '94108', '94109', '94110',
    '94111', '94112', '94114', '94115', '94116', '94117', '94118', '94121',
    '94122', '94123', '94124', '94127', '94128', '94129', '94130', '94131',
    '94132', '94133', '94134', '94158'
]

# Reproducible CA Open Data (data.ca.gov / CKAN) source for CA DMV aggregates.
# Dataset: https://data.ca.gov/dataset/vehicle-fuel-type-count-by-zip-code
CA_DMV_CKAN_PACKAGE_ID = 'vehicle-fuel-type-count-by-zip-code'
CA_DMV_CKAN_API_BASE = 'https://data.ca.gov/api/3/action'

# Known CKAN resource IDs for the yearly CSVs (as of 2026-01).
CA_DMV_RESOURCE_IDS = {
    2024: '66b0121e-5eab-4fcf-aa0d-2b1dfb5510ab',
    2023: 'd599c3d3-87af-4e8c-8694-9c01f49e3d93',
    2022: '9aa5b4c5-252c-4d68-b1be-ffe19a2f1d26',
}

# Fallback direct download URLs (kept as a safety net if CKAN metadata changes).
CA_DMV_FALLBACK_DOWNLOAD_URLS = {
    2024: 'https://data.ca.gov/dataset/15179472-adeb-4df6-920a-20640d02b08c/resource/66b0121e-5eab-4fcf-aa0d-2b1dfb5510ab/download/vehicle-fuel-type-counts-2024.csv',
    2023: 'https://data.ca.gov/dataset/15179472-adeb-4df6-920a-20640d02b08c/resource/d599c3d3-87af-4e8c-8694-9c01f49e3d93/download/vehicle-fuel-type-count-by-zip-code-20231.csv',
    2022: 'https://data.ca.gov/dataset/15179472-adeb-4df6-920a-20640d02b08c/resource/9aa5b4c5-252c-4d68-b1be-ffe19a2f1d26/download/vehicle-fuel-type-count-by-zip-code-2022.csv',
}

# Fuel type standardization
FUEL_TYPE_MAPPING = {
    'Battery Electric': 'EV',
    'Plug-In Hybrid': 'PHEV',
    'Gasoline': 'ICE_Gas',
    'Diesel': 'ICE_Diesel',
    'Hybrid Gasoline': 'Hybrid',
    'Flexible Fuel': 'ICE_Flex',
    'Natural Gas': 'ICE_NG'
}

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)