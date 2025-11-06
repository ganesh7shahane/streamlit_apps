"""
Core utility classes for molecular operations and data handling.
Implements caching and memory-efficient operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MolFromSmiles, MolToSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Optional, List, Tuple, Union
import psutil
import os


class MoleculeUtils:
    """Utilities for molecular operations with caching."""
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit molecule object."""
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def mol_to_smiles(mol: Chem.Mol) -> Optional[str]:
        """Convert RDKit molecule to canonical SMILES."""
        try:
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    @staticmethod
    def mol_to_image(mol: Chem.Mol, size: Tuple[int, int] = (300, 300)):
        """Convert RDKit molecule to PIL Image."""
        try:
            from rdkit.Chem import Draw
            img = Draw.MolToImage(mol, size=size)
            return img
        except:
            return None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def get_murcko_scaffold(smiles: str) -> Optional[str]:
        """Extract Murcko scaffold from SMILES."""
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            return None
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            return None
    
    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """Validate if SMILES string is valid."""
        return MoleculeUtils.smiles_to_mol(smiles) is not None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_descriptors(smiles: str, descriptor_names: List[str]) -> dict:
        """Calculate specified RDKit descriptors for a molecule."""
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            return {name: None for name in descriptor_names}
        
        results = {}
        for name in descriptor_names:
            try:
                func = getattr(Descriptors, name)
                results[name] = func(mol)
            except:
                results[name] = None
        return results
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def generate_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[str]:
        """Generate Morgan fingerprint as bit string."""
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            return None
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
            return fp.ToBitString()
        except:
            return None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_sa_score(smiles: str) -> Optional[float]:
        """Calculate Synthetic Accessibility Score."""
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            return None
        try:
            import sys
            import os
            from rdkit import RDConfig
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            return sascorer.calculateScore(mol)
        except:
            return None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def calculate_qed(smiles: str) -> Optional[float]:
        """Calculate QED (Quantitative Estimate of Drug-likeness)."""
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            return None
        try:
            return Chem.QED.qed(mol)
        except:
            return None
    
    @staticmethod
    def calculate_all_descriptors(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """
        Calculate all 200+ RDKit 2D descriptors for molecules in a DataFrame.
        
        Args:
            df: DataFrame containing molecules
            smiles_col: Name of the SMILES column
            
        Returns:
            DataFrame with all descriptors added (with _2 suffix)
        """
        result_df = df.copy()
        
        # Get all available descriptors from RDKit
        desc_list = Descriptors._descList
        
        # Convert SMILES to mol objects once
        mols = [MoleculeUtils.smiles_to_mol(smi) for smi in result_df[smiles_col]]
        
        # Calculate each descriptor
        progress_bar = st.progress(0)
        total_descriptors = len(desc_list)
        
        for idx, (desc_name, desc_func) in enumerate(desc_list):
            try:
                result_df[f"{desc_name}_2"] = [
                    desc_func(mol) if mol else None for mol in mols
                ]
            except Exception as e:
                st.warning(f"Failed to calculate {desc_name}: {str(e)}")
                result_df[f"{desc_name}_2"] = None
            
            # Update progress
            progress_bar.progress((idx + 1) / total_descriptors)
        
        progress_bar.empty()
        st.success(f"✅ Calculated {total_descriptors} RDKit descriptors!")
        
        return result_df


class DataFrameUtils:
    """Utilities for DataFrame operations with memory optimization."""
    
    @staticmethod
    def load_csv(file_source: Union[str, any], max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load CSV from file or URL with optional row limit."""
        try:
            df = pd.read_csv(file_source, index_col=False, nrows=max_rows)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
        """Automatically detect SMILES column."""
        smiles_keywords = ['smiles', 'SMILES', 'Canonical_Smiles', 'CANONICAL_SMILES', 'canonical_smiles']
        
        for col in df.columns:
            if col in smiles_keywords:
                return col
            if 'smiles' in col.lower():
                return col
        return None
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df
    
    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Get list of numeric column names."""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    @staticmethod
    def validate_smiles_batch(df: pd.DataFrame, smiles_col: str, chunk_size: int = 1000) -> Tuple[List[str], List[str]]:
        """Validate SMILES in batches, return valid and invalid lists."""
        valid = []
        invalid = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df[smiles_col].iloc[i:i+chunk_size]
            for smiles in chunk:
                if MoleculeUtils.validate_smiles(smiles):
                    valid.append(smiles)
                else:
                    invalid.append(smiles)
        
        return valid, invalid


class MemoryUtils:
    """Utilities for memory monitoring and management."""
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get system memory usage statistics."""
        try:
            mem = psutil.virtual_memory()
            return {
                'total_gb': mem.total / (1024 ** 3),
                'available_gb': mem.available / (1024 ** 3),
                'used_gb': mem.used / (1024 ** 3),
                'percent': mem.percent
            }
        except:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent': 0
            }
    
    @staticmethod
    def get_process_memory() -> str:
        """Get current process memory usage."""
        try:
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss
            return MemoryUtils._format_bytes(mem_bytes)
        except:
            return "N/A"
    
    @staticmethod
    def get_dataframe_memory(df: pd.DataFrame) -> str:
        """Get DataFrame memory usage."""
        mem_bytes = df.memory_usage(deep=True).sum()
        return MemoryUtils._format_bytes(mem_bytes)
    
    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"
    
    @staticmethod
    def check_memory_warning(df: pd.DataFrame, threshold_mb: int = 500):
        """Display warning if DataFrame exceeds memory threshold."""
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 ** 2)
        
        if mem_mb > threshold_mb:
            st.warning(
                f"⚠️ Large dataset: {mem_mb:.1f} MB. Consider reducing rows for better performance."
            )


class PlotUtils:
    """Utilities for plotting operations."""
    
    # Color palette for plots
    COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def prepare_histogram_data(data: np.ndarray, bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare histogram data with caching."""
        counts, bin_edges = np.histogram(data, bins=bins)
        return bin_edges, counts
    
    @staticmethod
    def format_number(num: float, decimals: int = 2) -> str:
        """Format number with specified decimals."""
        return f"{num:.{decimals}f}"
    
    @staticmethod
    def format_plot(fig, ax):
        """Apply consistent formatting to matplotlib plots."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        fig.tight_layout()
