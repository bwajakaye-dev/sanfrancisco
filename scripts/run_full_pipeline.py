"""
Main pipeline to run complete EV vs ICE analysis.
Execute all steps from data collection to analysis.
"""

import sys
import os

# Add project root to path so `src.*` imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data_collection.download_ca_dmv import download_ca_dmv_data
from src.data_collection.download_sf_data import download_sf_parking_data
from src.preprocessing.clean_vehicle_data import clean_all_years
from src.preprocessing.feature_engineering import create_aggregated_features
from src.analysis.clustering import perform_clustering_analysis
from src.analysis.association_rules import run_association_rule_discovery
from src.analysis.predictive_modeling import run_predictive_modeling
from src.analysis.anomaly_detection import run_anomaly_detection

def main():
    """Run complete analysis pipeline."""
    print("\n" + "="*60)
    print("🚗 SF EV vs ICE Vehicle Analysis Pipeline")
    print("="*60)
    
    try:
        # Step 1: Data Collection
        print("\n[STEP 1/7] 📥 Data Collection")
        print("-" * 60)
        if not download_ca_dmv_data():
            print("\n❌ Step 1 failed: CA DMV download step did not complete.")
            print("   Fix: confirm the raw CSVs exist under src/data_collection/data/raw")
            return 1

        if not download_sf_parking_data():
            print("\n⚠️  SF parking download failed; continuing pipeline.")
            print("   Fix: re-run later or ensure src/data_collection/data/raw/sf_parking_meters.csv exists")
        
        # Step 2: Data Cleaning
        print("\n[STEP 2/7] 🧹 Data Cleaning")
        print("-" * 60)
        clean_all_years()
        
        # Step 3: Feature Engineering
        print("\n[STEP 3/7] 🔧 Feature Engineering")
        print("-" * 60)
        features_df = create_aggregated_features()
        if features_df is None:
            print("\n❌ Step 3 failed: no features were generated (likely missing cleaned CSVs).")
            return 1
        
        # Step 4: Clustering Analysis
        print("\n[STEP 4/7] 🔍 Clustering Analysis")
        print("-" * 60)
        perform_clustering_analysis()

        # Step 5: Association Rules
        print("\n[STEP 5/7] 🧩 Association Rule Discovery")
        print("-" * 60)
        try:
            run_association_rule_discovery()
        except Exception as e:
            print(f"\n⚠️  Step 5 warning: association rules step failed: {e}")
            print("   Continuing pipeline...")

        # Step 6: Predictive Modeling
        print("\n[STEP 6/7] 🧠 Predictive Modeling")
        print("-" * 60)
        try:
            run_predictive_modeling()
        except Exception as e:
            print(f"\n⚠️  Step 6 warning: predictive modeling step failed: {e}")
            print("   Continuing pipeline...")

        # Step 7: Anomaly Detection
        print("\n[STEP 7/7] 🚨 Anomaly Detection")
        print("-" * 60)
        try:
            run_anomaly_detection()
        except Exception as e:
            print(f"\n⚠️  Step 7 warning: anomaly detection step failed: {e}")
            print("   Continuing pipeline...")
        
        # Summary
        print("\n📊 Analysis Complete!")
        print("-" * 60)
        print("\n✅ Pipeline executed successfully!")
        print("\n📁 Check results in:")
        print("   - results/clustering_results.csv")
        print("   - results/figures/clustering_analysis.png")
        print("   - results/association_rules.csv")
        print("   - results/figures/association_rules_top.png")
        print("   - results/predictive_modeling_metrics.csv")
        print("   - results/predictive_modeling_predictions.csv")
        print("   - results/figures/predictive_modeling_regression.png")
        print("   - results/anomaly_detection_results.csv")
        print("   - results/figures/anomaly_detection.png")
        print("   - results/models/*.joblib")
        print("\n🎉 Ready for demo!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())