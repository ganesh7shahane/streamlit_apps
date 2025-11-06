"""
Base classes for page analyzers.
Defines common interface and behavior for all analysis pages.
"""

from abc import ABC, abstractmethod
import streamlit as st
import pandas as pd
from typing import Optional
import gc

from sar_app.config import AppConfig
from sar_app.utils import DataFrameUtils, MemoryUtils


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzer pages."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self._df: Optional[pd.DataFrame] = None
        self._smiles_col: Optional[str] = None
    
    @abstractmethod
    def render(self):
        """Render the analyzer page. Must be implemented by subclasses."""
        pass
    
    def cleanup(self):
        """Clean up resources and free memory."""
        if self._df is not None:
            del self._df
            self._df = None
        gc.collect()
    
    def _load_data(self, uploaded_file, default_url: str) -> bool:
        """
        Load data from uploaded file or default URL.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            default_url: Default URL to load if no file uploaded
            
        Returns:
            bool: True if data loaded successfully
        """
        if uploaded_file is not None:
            self._df = DataFrameUtils.load_csv(uploaded_file, self.config.MAX_ROWS)
            st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            self._df = DataFrameUtils.load_csv(default_url, self.config.MAX_ROWS)
            st.info("📊 Using default dataset if no CSV given...")
        
        if self._df.empty:
            st.error("❌ Failed to load data")
            return False
        
        # Optimize memory
        self._df = DataFrameUtils.optimize_dtypes(self._df)
        MemoryUtils.check_memory_warning(self._df)
        
        # Find SMILES column
        self._smiles_col = DataFrameUtils.find_smiles_column(self._df)
        if self._smiles_col:
            st.success(f"✅ SMILES column: `{self._smiles_col}`")
        
        return True
    
    @property
    def df(self) -> Optional[pd.DataFrame]:
        """Get the loaded DataFrame."""
        return self._df
    
    @property
    def smiles_col(self) -> Optional[str]:
        """Get the SMILES column name."""
        return self._smiles_col
