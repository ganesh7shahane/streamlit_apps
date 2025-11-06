"""
Configuration Settings for SAR Analysis Application
Centralized configuration using dataclass for type safety and easy modification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class AppConfig:
    """Immutable configuration for the SAR Analysis application."""
    
    # Page Configuration
    PAGE_TITLE: str = "SAR Analysis Tools"
    PAGE_ICON: str = "⚛️"
    LAYOUT: str = "wide"
    SIDEBAR_STATE: str = "expanded"
    PAGE_MAX_WIDTH: str = "55rem"
    
    # Menu Configuration
    MENU_TITLE: str = "Structure-Activity Relationship (SAR) Analysis Tools"
    MENU_ICON: str = "bar-chart-fill"
    MENU_ORIENTATION: str = "horizontal"
    MENU_OPTIONS: Tuple[str, ...] = (
        "DataFrame Wizard",
        "Scaffold Hunter", 
        "SMILES Analysis",
        "Taylor-Butina Clustering"
    )
    MENU_ICONS: Tuple[str, ...] = ("0-square", "1-square", "2-square", "3-square")
    
    # Default Data Sources
    DEFAULT_DATAFRAME_URL: str = 'https://raw.githubusercontent.com/ganesh7shahane/useful_cheminformatics/refs/heads/main/data/FINE_TUNING_pi3k-mtor_objectives.csv'
    DEFAULT_SCAFFOLD_URL: str = 'https://raw.githubusercontent.com/ganesh7shahane/streamlit_apps/refs/heads/main/data/chembl208.csv'
    DEFAULT_CLUSTERING_URL: str = 'https://raw.githubusercontent.com/ganesh7shahane/streamlit_apps/refs/heads/main/data/chembl1075104.csv'
    
    # Styling
    SIDEBAR_BG_COLOR: str = "#f0f2f6"
    PRIMARY_COLOR: str = "#1E88E5"
    SECONDARY_COLOR: str = "#FFA726"
    
    # Plot Settings
    PLOT_DPI: int = 300
    PLOT_FIGSIZE: Tuple[int, int] = (14, 15)
    HISTOGRAM_COLOR: str = '#82EEDE'
    BARPLOT_COLOR: str = '#125AF7'
    DEFAULT_BINS: int = 30
    
    # Molecule Visualization
    MOL_GRID_SIZE: Tuple[int, int] = (200, 200)
    MOL_ITEMS_PER_PAGE: int = 18
    MOL_FIXED_BOND_LENGTH: int = 25
    MOL_CLEAR_BACKGROUND: bool = False
    
    # Memory & Performance
    MAX_ROWS: int = 100000
    CHUNK_SIZE: int = 10000
    CACHE_TTL: int = 3600  # seconds
    
    # Fingerprint Settings
    FP_RADIUS: int = 2
    FP_BITS: int = 2048
    
    # Descriptor Lists
    BASIC_DESCRIPTORS: Tuple[str, ...] = (
        'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3'
    )
