"""
Taylor-Butina Clustering Page
Cluster molecules based on structural similarity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdRGroupDecomposition
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
        st.title("🔗 Butina Clustering & R-Group Decomposition")
        
        st.markdown("""
        Cluster molecules based on structural similarity:
        - :red[Generate] molecular fingerprints
        - :red[Calculate] distance matrix
        - :red[Cluster] using Butina algorithm
        - :red[Visualize] clusters and representatives
        - :red[Perform] R-group decomposition based on selected scaffold
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
            st.dataframe(self.df.head(n_rows), width='stretch')
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

        # Select legend columns with multiselect
        available_legend_cols = [col for col in df.columns if col not in ['mol']]
        default_cluster_legends = available_legend_cols[:min(2, len(available_legend_cols))]
        
        selected_cluster_legend_cols = st.multiselect(
            "Choose columns to display as legends:",
            options=available_legend_cols,
            default=default_cluster_legends,
            key="cluster_legend_cols"
        )
        
        if not selected_cluster_legend_cols:
            st.warning("⚠️ No legend columns selected. Please select at least one column to display.")
            return
        
        for rank, (idx, cluster) in enumerate(sorted_clusters[:n_display], 1):
            with st.expander(f"Cluster {idx+1} (Size: {len(cluster)}, Rank: {rank})"):
                # Get molecules
                cluster_df = df.iloc[list(cluster)].copy()
                cluster_df.loc[:, 'mol'] = cluster_df[smiles_col].apply(MoleculeUtils.smiles_to_mol)
                cluster_df.loc[:, 'Cluster_Index'] = [cluster[i] for i in range(len(cluster))]
                cluster_df.loc[:, 'Is_Representative'] = ['Yes' if i == 0 else 'No' for i in range(len(cluster))]
                
                #st.write(f"📌 Showing all molecules")
                
                # Display using mols2grid
                try:
                    # Format numeric legend columns to 2 decimal places for display
                    cluster_df_display = cluster_df.copy()
                    for col in selected_cluster_legend_cols:
                        if col in cluster_df_display.columns and pd.api.types.is_numeric_dtype(cluster_df_display[col]):
                            cluster_df_display.loc[:, col] = cluster_df_display[col].apply(
                                lambda v: f"{v:.2f}" if pd.notnull(v) else ""
                            )
                    
                    # Create tooltip with all available columns (except 'mol')
                    tooltip_cols = [col for col in cluster_df_display.columns if col != 'mol']
                    
                    # Build subset list: img first, then selected legend columns
                    subset_list = ["img"] + selected_cluster_legend_cols
                    
                    html_data = mols2grid.display(
                        cluster_df_display,
                        mol_col='mol',
                        size=(200, 200),
                        subset=subset_list,
                        tooltip=tooltip_cols,
                        n_items_per_page=18,
                        fixedBondLength=25,
                        clearBackground=False
                    ).data
                    
                    # Calculate dynamic height based on number of legend columns
                    base_height = 670
                    height_per_legend = 80
                    dynamic_height = base_height + (len(selected_cluster_legend_cols) * height_per_legend - 10)
                    
                    st.components.v1.html(html_data, height=dynamic_height, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying molecules: {str(e)}")
                    
                    # Fallback: show dataframe
                    display_df_cols = [smiles_col, 'Cluster_Index', 'Is_Representative']
                    if 'MW' in cluster_df.columns:
                        display_df_cols.append('MW')
                    if 'pchembl_value' in cluster_df.columns:
                        display_df_cols.append('pchembl_value')
                    
                    st.dataframe(cluster_df[display_df_cols], hide_index=False)
        
        # R-group table section
        self._display_rgroup_table(results, df)
    
    def _display_rgroup_table(self, results: dict, df: pd.DataFrame):
        """Display R-group decomposition table for selected cluster."""
        st.markdown("---")
        st.subheader("🧪 R-group Table Analysis")
        
        st.markdown("""
        Perform R-group decomposition on a selected cluster:
        - Enter a scaffold/core SMILES structure
        - Select a cluster to analyze
        - View R-group substitutions and their frequencies
        """)
        
        clusters = results['clusters']
        smiles_col = self.smiles_col
        
        # Sort clusters by size
        sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text area for core SMILES input
            core_smiles = st.text_area(
                "Enter Core/Scaffold SMILES with R-groups",
                value="[*]C(=O)c1ccc(N[*])c(O[*])c1",
                height=100,
                help="Use [*] to indicate attachment points for R-groups"
            )
        
        with col2:
            # Dropdown to select cluster
            cluster_options = [f"Cluster {idx+1} (Size: {len(cluster)})" 
                             for idx, cluster in sorted_clusters]
            selected_cluster_idx = st.selectbox(
                "Select Cluster for R-group Analysis",
                range(len(cluster_options)),
                format_func=lambda x: cluster_options[x]
            )
        
        # Display core structure
        if core_smiles:
            try:
                core_mol = Chem.MolFromSmiles(core_smiles)
                if core_mol:
                    st.markdown("**Core Structure:**")
                    col_img1, col_img2 = st.columns([1, 2])
                    with col_img1:
                        st.image(Draw.MolToImage(core_mol, size=(250, 250)))
                    with col_img2:
                        # Show molblock
                        molblock = Chem.MolToMolBlock(core_mol)
                        with st.expander("View MolBlock"):
                            st.code(molblock, language="text")
                else:
                    st.error("Invalid SMILES. Please enter a valid structure.")
                    return
            except Exception as e:
                st.error(f"Error parsing SMILES: {str(e)}")
                return
        else:
            st.info("Enter a core SMILES to begin analysis")
            return
        
        # Button to perform R-group decomposition
        if st.button("🚀 Perform R-group Decomposition", type="primary"):
            try:
                # Get selected cluster
                cluster_idx, cluster = sorted_clusters[selected_cluster_idx]
                cluster_df = df.iloc[list(cluster)].copy()
                cluster_df.loc[:, 'mol'] = cluster_df[smiles_col].apply(Chem.MolFromSmiles)
                cluster_df.loc[:, 'index'] = range(len(cluster_df))
                
                st.info(f"Analyzing {len(cluster_df)} molecules from Cluster {cluster_idx+1}")
                
                # Perform R-group decomposition
                with st.spinner("Performing R-group decomposition..."):
                    rgd, failed = rdRGroupDecomposition.RGroupDecompose(
                        [core_mol], 
                        cluster_df['mol'].values,
                        asRows=False
                    )
                
                # Store results in session state
                st.session_state['rgd_results'] = {
                    'rgd': rgd,
                    'failed': failed,
                    'cluster_df': cluster_df,
                    'cluster_idx': cluster_idx
                }
                
            except Exception as e:
                st.error(f"Error during R-group decomposition: {str(e)}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                return
        
        # Display R-group results if available
        if 'rgd_results' in st.session_state:
            self._display_rgroup_results(st.session_state['rgd_results'])
    
    def _display_rgroup_results(self, rgd_results: dict):
        """Display R-group decomposition results."""
        rgd = rgd_results['rgd']
        failed = rgd_results['failed']
        cluster_df = rgd_results['cluster_df']
        cluster_idx = rgd_results['cluster_idx']
        
        st.markdown("---")
        st.subheader("📊 R-group Decomposition Results")
        
        # Show decomposition summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Molecules", len(cluster_df))
        with col2:
            st.metric("Successfully Decomposed", len(cluster_df) - len(failed))
        with col3:
            st.metric("Failed", len(failed))
        
        # Display the core with R-groups
        if 'Core' in rgd:
            st.markdown("**Identified Core with R-groups:**")
            core_with_rgroups = rgd['Core'][0]
            col_core1, col_core2 = st.columns([1, 2])
            with col_core1:
                st.image(Draw.MolToImage(core_with_rgroups, size=(300, 300)))
            with col_core2:
                st.info(f"The R-group decomposition identified {len([k for k in rgd.keys() if k != 'Core'])} R-group positions")
        
        # Show failed molecules if any
        if failed:
            with st.expander(f"⚠️ Failed Molecules ({len(failed)})"):
                st.markdown("These molecules did not match the core structure:")
                # Use iloc to get failed molecules by position
                failed_df = cluster_df.iloc[failed].copy()
                
                try:
                    failed_html = mols2grid.display(
                        failed_df,
                        mol_col='mol',
                        subset=['img', 'index'],
                        n_items_per_page=18,
                        size=(150, 150)
                    )._repr_html_()
                    st.components.v1.html(failed_html, height=700, scrolling=True)
                except:
                    st.dataframe(failed_df[[self.smiles_col, 'index']])
        
        # Get R-group names
        r_groups = sorted([x for x in rgd.keys() if x != "Core"])
        
        if not r_groups:
            st.warning("No R-groups found in the decomposition")
            return
        
        st.markdown(f"**Found R-groups:** {', '.join(r_groups)}")
        
        # Add R-group SMILES to dataframe BEFORE removing failed molecules
        # Ensure the R-group data matches the cluster_df length
        for r in r_groups:
            # Pad or truncate R-group data to match cluster_df length
            rgroup_data = rgd[r]
            if len(rgroup_data) < len(cluster_df):
                # Pad with None if needed
                rgroup_data = list(rgroup_data) + [None] * (len(cluster_df) - len(rgroup_data))
            elif len(rgroup_data) > len(cluster_df):
                # Truncate if needed
                rgroup_data = rgroup_data[:len(cluster_df)]
            
            cluster_df.loc[:, r] = rgroup_data
            # Convert to SMILES, handling None values for failed molecules
            cluster_df.loc[:, r] = cluster_df[r].apply(
                lambda x: Chem.MolToSmiles(x) if x is not None else None
            )
        
        # Now remove failed molecules from cluster_df
        if failed:
            # Get the indices we want to keep (all except failed)
            keep_indices = [i for i in range(len(cluster_df)) if i not in failed]
            cluster_df = cluster_df.iloc[keep_indices].reset_index(drop=True)
            # Remove None values from R-group columns
            for r in r_groups:
                cluster_df = cluster_df[cluster_df[r].notna()]
        
        # Display R-group frequency tables
        st.markdown("---")
        st.subheader("📈 R-group Frequencies")
        
        # Create tabs for each R-group
        tabs = st.tabs(r_groups)
        
        for tab, rg in zip(tabs, r_groups):
            with tab:
                # Count frequencies
                value_counts = cluster_df[rg].value_counts().reset_index()
                value_counts.columns = [rg, 'count']
                
                # Add molecule column
                value_counts['mol'] = value_counts[rg].apply(Chem.MolFromSmiles)
                
                st.markdown(f"**{rg} - {len(value_counts)} unique substituents**")
                
                col_table, col_viz = st.columns([1, 2])
                
                with col_table:
                    # Display frequency table
                    display_df = value_counts[[rg, 'count']].copy()
                    st.dataframe(display_df, hide_index=True, height=400)
                
                with col_viz:
                    # Display molecules using mols2grid
                    try:
                        grid_html = mols2grid.display(
                            value_counts,
                            mol_col='mol',
                            smiles_col=rg,
                            subset=['img', 'count'],
                            size=(150, 150),
                            n_items_per_page=12
                        )._repr_html_()
                        st.components.v1.html(grid_html, height=800, scrolling=True)
                    except Exception as e:
                        st.error(f"Error displaying R-group structures: {str(e)}")
        
        # Download option for R-group table
        st.markdown("---")
        st.subheader("💾 Download R-group Table")
        
        # Prepare download dataframe
        download_cols = [self.smiles_col] + r_groups
        if 'Name' in cluster_df.columns:
            download_cols.insert(0, 'Name')
        
        available_download_cols = [col for col in download_cols if col in cluster_df.columns]
        download_df = cluster_df[available_download_cols]
        
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download R-group Table (CSV)",
            data=csv,
            file_name=f"rgroup_table_cluster_{cluster_idx+1}.csv",
            mime="text/csv"
        )
