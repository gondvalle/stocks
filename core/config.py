"""Configuración de la aplicación."""
import os
from pathlib import Path

RISK_FREE = float(os.getenv("RISK_FREE", "4.0")) / 100.0
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"
