# /ui/__init__.py
"""Paquete UI."""
# No importes submódulos aquí para evitar imports circulares.
from . import pages  # expone el subpaquete 'pages'
__all__ = ["pages"]
