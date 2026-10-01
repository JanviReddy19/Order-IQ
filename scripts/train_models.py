"""Optional offline training script.

It downloads/loads the dataset, fits the two ML components once, and saves
joblib artifacts. Streamlit itself also caches the fitted objects for the
lifetime of the running app.
"""
from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_data
from analytics import order_level, match_cancellations, fit_anomaly_model, fit_cancellation_model

ART = ROOT / "models"
ART.mkdir(exist_ok=True)

df, source = load_data(False)
orders = order_level(df)
match_df, _ = match_cancellations(df)
anomaly = fit_anomaly_model(orders)
cancel = fit_cancellation_model(orders, match_df)
joblib.dump(anomaly, ART / "anomaly_bundle.joblib")
if cancel is not None:
    joblib.dump(cancel, ART / "cancellation_model.joblib")
print(f"Saved model artifacts for: {source}")
