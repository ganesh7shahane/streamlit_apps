"""
Taylor-Butina Clustering Page
Cluster molecules based on structural similarity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt
import seaborn as sns
import mols2grid
from typing import List, Tuple

from sar_app.base import BaseAnalyzer
from sar_app.utils import MoleculeUtils, DataFrameUtils, PlotUtils


class ClusteringAnalyzer(BaseAnalyzer):
    """Taylor-Butina clustering for molecular datasets."""
    
    def render(self):
        """Render the clustering page."""
        st.title("🔗 Taylor-Butina Clustering")
        
        st.markdown("""
        Cluster molecules based on structural similarity:
        - :red[Generate] molecular fingerprints
        - :red[Calculate] distance matrix
        - :red[Cluster] using Butina algorithm
        - :red[Visualize] clusters and representatives
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV with SMILES",
            type=['csv'],
            help="CSV file containing molecular SMILES"
        )
        
        if not self._load_data(uploaded_file, self.config.DEFAULT_CLUSTERING_URL):
            return
        
        # Preview dataset
        with st.expander("📊 Preview Dataset", expanded=True):
            n_rows = st.slider(
                "Number of rows to display",
                min_value=3,
                max_value=min(100, len(self.df)),
                value=3,
                step=1,
                key="clustering_preview_rows"
            )
            st.dataframe(self.df.head(n_rows), use_container_width=True)
            st.info(f"Total rows in dataset: {len(self.df)}")
        
        # Configuration
        config = self._get_clustering_config()
        
        # Perform clustering only if button was clicked
        if config:
            with st.spinner("Clustering molecules..."):
                results = self._perform_clustering(self.df, config)
            
            if results:
                # Store results in session state
                st.session_state['clustering_results'] = results
                st.session_state['clustering_df'] = self.df
                st.session_state['clustering_config'] = config
        
        # Display results if available in session state (regardless of button click)
        if 'clustering_results' in st.session_state:
            self._display_results(
                st.session_state['clustering_results'],
                st.session_state['clustering_df'],
                st.session_state['clustering_config']
            )
    
    def _get_clustering_config(self) -> dict:
        """Get clustering configuration."""
        with st.expander("⚙️ Clustering Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fp_type = st.selectbox(
                    "Fingerprint",
                    ["Morgan", "RDKit", "MACCS"],
                    help="Type of molecular fingerprint"
                )
            
            with col2:
                cutoff = st.slider(
                    "Distance Cutoff",
                    0.1, 0.9, 0.4, 0.05,
                    help="Max distance for clustering (lower = tighter)"
                )
            
            with col3:
                radius = st.number_input(
                    "Morgan Radius",
                    1, 4, 2,
                    disabled=(fp_type != "Morgan"),
                    help="Radius for Morgan fingerprint"
                )
            
            if st.button("🚀 Run Clustering", type="primary"):
                return {
                    'fp_type': fp_type,
                    'cutoff': cutoff,
                    'radius': radius
                }
        
        return None
    
    def _perform_clustering(self, df: pd.DataFrame, config: dict) -> dict:
        """Perform Butina clustering."""
        try:
            # Generate fingerprints
            fps = self._generate_fingerprints(df, config)
            if not fps:
                st.error("Failed to generate fingerprints")
                return None
            
            # Calculate distance matrix
            dists = self._calculate_distances(fps)
            
            # Perform clustering
            clusters = Butina.ClusterData(
                dists,
                len(fps),
                config['cutoff'],
                isDistData=True
            )
            
            return {
                'clusters': clusters,
                'fps': fps,
                'n_clusters': len(clusters),
                'n_molecules': len(fps)
            }
        except Exception as e:
            st.error(f"Clustering error: {str(e)}")
            return None
    
    def _generate_fingerprints(self, df: pd.DataFrame, config: dict) -> List:
        """Generate molecular fingerprints."""
        fps = []
        smiles_col = self.smiles_col
        
        for smi in df[smiles_col]:
            mol = MoleculeUtils.smiles_to_mol(smi)
            if not mol:
                continue
            
            try:
                if config['fp_type'] == "Morgan":
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, config['radius'], 2048
                    )
                elif config['fp_type'] == "RDKit":
                    fp = Chem.RDKFingerprint(mol)
                else:  # MACCS
                    from rdkit.Chem import MACCSkeys
                    fp = MACCSkeys.GenMACCSKeys(mol)
                
                fps.append(fp)
            except:
                continue
        
        return fps
    
    def _calculate_distances(self, fps: List) -> List[float]:
        """Calculate distance matrix."""
        n = len(fps)
        dists = []
        
        # Progress bar for large datasets
        if n > 100:
            progress = st.progress(0)
        
        for i in range(n):
            for j in range(i):
                dist = 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(dist)
            
            if n > 100 and i % 10 == 0:
                progress.progress((i+1) / n)
        
        if n > 100:
            progress.empty()
        
        return dists
    
    def _display_results(self, results: dict, df: pd.DataFrame, config: dict):
        """Display clustering results."""
        st.success(f"✅ Found {results['n_clusters']} clusters from {results['n_molecules']} molecules")
        
        # Statistics
        self._display_statistics(results)
        
        # Cluster distribution
        self._display_distribution(results)
        
        # Cluster details
        self._display_cluster_details(results, df)
    
    def _display_statistics(self, results: dict):
        """Display clustering statistics."""
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        clusters = results['clusters']
        sizes = [len(c) for c in clusters]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Clusters", len(clusters))
        
        with col2:
            st.metric("Avg Size", f"{np.mean(sizes):.1f}")
        
        with col3:
            st.metric("Max Size", max(sizes))
        
        with col4:
            st.metric("Singletons", sum(1 for s in sizes if s == 1))
    
    def _display_distribution(self, results: dict):
        """Display cluster size distribution."""
        st.markdown("---")
        st.subheader("📈 Cluster Size Distribution")
        
        sizes = [len(c) for c in results['clusters']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sizes, bins=20, color=PlotUtils.COLORS[0], alpha=0.7, edgecolor='black')
            ax.set_xlabel("Cluster Size")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Cluster Sizes")
            PlotUtils.format_plot(fig, ax)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Summary stats
            st.markdown("#### Summary")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{np.mean(sizes):.2f}",
                    f"{np.median(sizes):.0f}",
                    f"{np.std(sizes):.2f}",
                    f"{min(sizes)}",
                    f"{max(sizes)}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)
    
    def _display_cluster_details(self, results: dict, df: pd.DataFrame):
        """Display individual cluster details."""
        st.markdown("---")
        st.subheader("🔍 Cluster Details")
        
        clusters = results['clusters']
        smiles_col = self.smiles_col
        
        # Sort by size
        sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)
        
        # Display top N clusters and legend selector side by side
        n_clusters_available = len(sorted_clusters)
        max_display = max(1, n_clusters_available)
        default_display = min(5, max_display)

        col1, col2 = st.columns([3, 1])
        with col1:
            n_display = st.slider(
            "Number of clusters to display",
            min_value=1,
            max_value=max_display,
            value=default_display,
            step=1,
            key="cluster_display_slider"
            )

        with col2:
            # Give option from selectbox to choose column name as legend
            legend_option = st.selectbox(
            "Select Legend Display",
            df.columns.tolist(),
            key="cluster_legend_selectbox"
            )
        
        for rank, (idx, cluster) in enumerate(sorted_clusters[:n_display], 1):
            with st.expander(f"Cluster {idx+1} (Size: {len(cluster)}, Rank: {rank})"):
                # Get molecules
                cluster_df = df.iloc[list(cluster)].copy()
                cluster_df.loc[:, 'mol'] = cluster_df[smiles_col].apply(MoleculeUtils.smiles_to_mol)
                cluster_df.loc[:, 'Cluster_Index'] = [cluster[i] for i in range(len(cluster))]
                cluster_df.loc[:, 'Is_Representative'] = ['Yes' if i == 0 else 'No' for i in range(len(cluster))]
                
                st.write(f"📌 Showing {min(len(cluster), 20)} of {len(cluster)} molecules")
                
                # Display using mols2grid
                try:
                    # Prepare display columns; format legend to 2 decimals if numeric
                    display_cols = ['img', 'Cluster_Index']
                    if legend_option in cluster_df.columns:
                        def _format_legend(v):
                            try:
                                if pd.isna(v):
                                    return ""
                                return f"{float(v):.2f}"
                            except Exception:
                                return str(v)
                        cluster_df['legend_display'] = cluster_df[legend_option].apply(_format_legend)
                        display_cols.append('legend_display')
                    
                    # Add additional columns if available
                    if 'MW' in cluster_df.columns:
                        display_cols.append('MW')
                    if 'pchembl_value' in cluster_df.columns:
                        display_cols.append('pchembl_value')
                    
                    html_data = mols2grid.display(
                        cluster_df.head(20),
                        mol_col='mol',
                        size=(200, 200),
                        subset=display_cols,
                        n_items_per_page=16,
                        fixedBondLength=25,
                        clearBackground=False
                    ).data
                    
                    st.components.v1.html(html_data, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying molecules: {str(e)}")
                    
                    # Fallback: show dataframe
                    display_df_cols = [smiles_col, 'Cluster_Index', 'Is_Representative']
                    if 'MW' in cluster_df.columns:
                        display_df_cols.append('MW')
                    if 'pchembl_value' in cluster_df.columns:
                        display_df_cols.append('pchembl_value')
                    
                    st.dataframe(cluster_df[display_df_cols].head(20), hide_index=False)
