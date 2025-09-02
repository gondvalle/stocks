# /sp500_screener/core/config.py
"""Configuración de la aplicación."""
import os
from pathlib import Path

RISK_FREE = float(os.getenv("RISK_FREE", "4.0")) / 100.0  # 4% por defecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
