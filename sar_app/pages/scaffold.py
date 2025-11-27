"""
Scaffold Hunter Page
Identifies and analyzes Murcko scaffolds in molecular datasets.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mols2grid
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from IPython.display import HTML

from sar_app.base import BaseAnalyzer
from sar_app.utils import MoleculeUtils


class ScaffoldAnalyzer(BaseAnalyzer):
    """Scaffold identification and analysis."""
    
    def render(self):
        """Render the scaffold analysis page."""
        st.title("🏗️ Scaffold Finder")
        
        st.markdown("""
        Identify and analyze molecular scaffolds:
        - :red[Identify] Murcko scaffolds in your dataset
        - :red[Count] scaffold frequencies
        - :red[Relate] scaffolds to activity distributions
        - :red[Visualize] scaffolds and members
        """)
        
        uploaded_file = st.file_uploader("Upload CSV with SMILES", type=["csv"], key="scaffold_upload")
        
        if not self._load_data(uploaded_file, self.config.DEFAULT_SCAFFOLD_URL):
            return
        
        if not self.smiles_col:
            st.error("❌ No SMILES column found")
            return
        
        # Preview dataset
        with st.expander("📊 Preview Dataset", expanded=True):
            n_rows = st.slider(
                "Number of rows to display",
                min_value=3,
                max_value=min(100, len(self._df)),
                value=3,
                step=1,
                key="scaffold_preview_rows"
            )
            st.dataframe(self._df.head(n_rows), width='stretch')
            st.info(f"Total rows in dataset: {len(self._df)}")
        
        if st.button("🔍 Find Scaffolds", type="primary"):
            self._analyze_scaffolds()
        
        # Display scaffold analysis if already computed
        if 'scaffold_df' in st.session_state and 'scaffold_counts' in st.session_state:
            df = st.session_state['scaffold_df']
            scaffold_counts = st.session_state['scaffold_counts']
            
            # Display statistics
            st.success(f"✅ Found {len(scaffold_counts)} unique scaffolds from {len(df)} molecules")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Scaffolds", len(scaffold_counts))
            with col2:
                st.metric("Avg Molecules/Scaffold", f"{scaffold_counts['count'].mean():.1f}")
            with col3:
                st.metric("Max Molecules/Scaffold", scaffold_counts['count'].max())
            
            # Plot distribution
            with st.expander("Scaffold Distribution", expanded=False):
                fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
                top_n = min(20, len(scaffold_counts))
                ax.barh(range(top_n), scaffold_counts['count'].head(top_n), color=self.config.PRIMARY_COLOR)
                ax.set_yticks(range(top_n))
                ax.set_yticklabels([f"S{i}" for i in range(top_n)])
                ax.set_xlabel("Number of Molecules")
                ax.set_title(f"Top {top_n} Scaffolds")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Visualize scaffolds
            with st.expander("Scaffold Structures", expanded=True):
                n_display = st.slider("Scaffolds to display", 5, len(scaffold_counts), 20, key="scaffold_display")
                
                display_df = scaffold_counts.head(n_display).copy()
                display_df.loc[:, 'ID'] = [f"S{i}" for i in range(len(display_df))]
                
                # Select legend columns with multiselect
                available_cols = [col for col in display_df.columns if col != 'mol']
                default_legends = ['ID', 'count'] if 'count' in available_cols else available_cols[:min(2, len(available_cols))]
                
                selected_legend_cols = st.multiselect(
                    "Choose columns to display as legends:",
                    options=available_cols,
                    default=default_legends,
                    key="scaffold_legend_cols"
                )
                
                if not selected_legend_cols:
                    st.warning("⚠️ No legend columns selected. Please select at least one column to display.")
                    return
                
                try:
                    # Create tooltip with all available columns (except 'mol')
                    tooltip_cols = [col for col in display_df.columns if col != 'mol']
                    
                    # Build subset list: img first, then selected legend columns
                    subset_list = ["img"] + selected_legend_cols
                    
                    raw_html = mols2grid.display(
                        display_df,
                        subset=subset_list,
                        tooltip=tooltip_cols,
                        mol_col='mol',
                        size=(150, 150)
                    )._repr_html_()
                    
                    # Calculate dynamic height based on number of legend columns
                    base_height = 800
                    height_per_legend = 80
                    dynamic_height = base_height + (len(selected_legend_cols) * height_per_legend - 10)
                    
                    st.components.v1.html(raw_html, height=dynamic_height, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying scaffolds: {str(e)}")
            
            # Activity distribution with boxplots - NOW WITH ACTIVITY COLUMN SELECTOR
            st.markdown("---")
            st.subheader("📊 Relate Activity Distribution with Scaffolds")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    activity_col = st.selectbox(
                        ":green[Select] the activity column from the dataframe",
                        options=numeric_cols,
                        index=0
                    )
                
                with col2:
                    boxplot_color = st.color_picker("Boxplot color", "#1f77b4", key="scaffold_boxplot_color")
                
                if activity_col:
                    self._display_scaffold_activity_distribution(df, scaffold_counts, activity_col, boxplot_color)
            else:
                st.info("No numeric columns found for activity analysis")
            
            # Examine molecules within scaffold
            st.markdown("---")
            self._examine_scaffold_molecules(df, scaffold_counts, numeric_cols[0] if numeric_cols else None)
            
            # Download
            csv = scaffold_counts[['Murcko_Scaffold', 'count']].to_csv(index=False)
            st.download_button("⬇️ Download Scaffold Analysis", csv, "scaffold_analysis.csv", "text/csv")
    
    def _analyze_scaffolds(self):
        """Analyze scaffolds in the dataset."""
        with st.spinner("Extracting scaffolds..."):
            # Add scaffold column
            self._df['Murcko_Scaffold'] = self._df[self.smiles_col].apply(
                lambda x: MoleculeUtils.get_murcko_scaffold(x)
            )
            
            # Remove rows with no scaffold
            df = self._df.dropna(subset=['Murcko_Scaffold']).copy()
            
            if len(df) == 0:
                st.error("No valid scaffolds found")
                return
        
        # Count scaffolds
        scaffold_counts = df.groupby('Murcko_Scaffold').size().reset_index(name='count')
        scaffold_counts = scaffold_counts.sort_values('count', ascending=False).reset_index(drop=True)
        scaffold_counts['mol'] = scaffold_counts['Murcko_Scaffold'].apply(MoleculeUtils.smiles_to_mol)
        
        # Store in session state to persist across reruns
        st.session_state['scaffold_df'] = df
        st.session_state['scaffold_counts'] = scaffold_counts
        
        st.rerun()
    
    def _display_scaffold_activity_distribution(self, df, scaffold_counts, activity_col, boxplot_color="#1f77b4"):
        """Display boxplots showing activity distribution for each scaffold."""
        # Check for NaN values in activity column
        if df[activity_col].isna().sum() > 0:
            st.warning(f"⚠️ Removed {df[activity_col].isna().sum()} molecules with NaN activity values")
            df = df.dropna(subset=[activity_col])
        
        def boxplot_base64_image(dist: np.ndarray, x_lim=None) -> str:
            """Plot a distribution as a seaborn boxplot and save as base64 image.
               The x-axis label is set from the selected activity_col in the outer scope.
            """
            if x_lim is None:
                x_lim = [0, 10]
            plt.figure(dpi=150)
            sns.set(rc={'figure.figsize': (4, 1)})
            sns.set_style('whitegrid')
            ax = sns.boxplot(x=dist, color=boxplot_color)
            ax.set_xlim(x_lim[0], x_lim[1])
            # label x-axis with the chosen activity column
            try:
                ax.set_xlabel(activity_col)
                ax.xaxis.label.set_size(10)
            except Exception:
            # fallback: no label if activity_col not available
                pass
            s = io.BytesIO()
            plt.savefig(s, format='png', bbox_inches="tight")
            plt.close()
            s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            return '<img align="left" src="data:image/png;base64,%s">' % s
        
        def mol_to_base64_image(mol: Chem.Mol) -> str:
            """Convert an RDKit molecule to a base64 encoded image string."""
            plt.figure(dpi=1000)
            drawer = rdMolDraw2D.MolDraw2DCairo(450, 150)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            text = drawer.GetDrawingText()
            im_text64 = base64.b64encode(text).decode('utf8')
            img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
            return img_str
        
        # Create display with boxplots
        rows_to_display = min(scaffold_counts.shape[0], 20)
        tmp_df = scaffold_counts.head(rows_to_display).copy()
        tmp_df.loc[:, 'mol_img'] = tmp_df.mol.apply(mol_to_base64_image)
        
        img_list = []
        for smi in tmp_df['Murcko_Scaffold'].values:
            activity_vals = df.query("Murcko_Scaffold == @smi")[activity_col].values
            x_lim = [df[activity_col].min()*0.98, df[activity_col].max()*1.01]
            img_list.append(boxplot_base64_image(activity_vals, x_lim=x_lim))
        
        tmp_df.loc[:, 'dist_img'] = img_list
        
        with st.expander("Show/Hide Scaffold Activity Distribution", expanded=True):
            st.markdown(HTML(tmp_df[['mol_img', 'count', 'dist_img']].to_html(escape=False)).data, unsafe_allow_html=True)
    
    def _examine_scaffold_molecules(self, df, scaffold_counts, activity_col):
        """Examine molecules with a given scaffold."""
        #st.markdown("---")
        st.subheader("🔍 Examine Molecules with a Given Scaffold")
        
        scaffold_id = st.selectbox(
            "Select a scaffold index",
            options=scaffold_counts.index.tolist(),
            format_func=lambda x: f"Scaffold {x} ({scaffold_counts.loc[x, 'count']} molecules)"
        )
        
        scaffold_smi = scaffold_counts.loc[scaffold_id, 'Murcko_Scaffold']
        scaffold_mol = scaffold_counts.loc[scaffold_id, 'mol']
        
        # Get molecules with this scaffold
        tmp_df = df.query("Murcko_Scaffold == @scaffold_smi").copy()
        
        # Align molecules to scaffold using GenerateDepictionMatching2DStructure
        # First cleanup and prepare the scaffold
        rdDepictor.Compute2DCoords(scaffold_mol)
        rdDepictor.SetPreferCoordGen(True)
        
        # Generate aligned 2D coordinates for all molecules and store them
        aligned_mols = []
        for smiles in tmp_df[self.smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    rdDepictor.Compute2DCoords(mol)
                    AllChem.GenerateDepictionMatching2DStructure(mol, scaffold_mol)
                    aligned_mols.append(mol)
                except:
                    # If alignment fails, use default 2D coordinates
                    rdDepictor.Compute2DCoords(mol)
                    aligned_mols.append(mol)
            else:
                aligned_mols.append(None)
        
        # Store aligned molecules in dataframe
        tmp_df.loc[:, 'mol'] = aligned_mols
        
        # Select legend columns with multiselect
        available_cols = [col for col in tmp_df.columns if col not in ['mol', 'Murcko_Scaffold']]
        default_legends = available_cols[:min(2, len(available_cols))]
        
        selected_legend_cols = st.multiselect(
            "Choose columns to display as legends:",
            options=available_cols,
            default=default_legends,
            key="scaffold_mol_legend_cols"
        )
        
        if not selected_legend_cols:
            st.warning("⚠️ No legend columns selected. Please select at least one column to display.")
            return
        
        # Sort by activity if available
        if activity_col and activity_col in tmp_df.columns:
            tmp_df = tmp_df.sort_values(activity_col, ascending=False)
        
        st.write(f"📌 There are {len(tmp_df)} molecules with scaffold ID {scaffold_id}")
        
        try:
            # Create tooltip with all available columns (except 'mol')
            tooltip_cols = [col for col in tmp_df.columns if col != 'mol']
            
            # Build subset list: img first, then selected legend columns
            subset_list = ["img"] + selected_legend_cols
            
            html_data = mols2grid.display(
                tmp_df,
                mol_col='mol',
                size=(200, 200),
                subset=subset_list,
                tooltip=tooltip_cols,
                n_items_per_page=18,
                fixedBondLength=25,
                clearBackground=False
            ).data
            
            # Calculate dynamic height based on number of legend columns
            base_height = 800
            height_per_legend = 80
            dynamic_height = base_height + (len(selected_legend_cols) * height_per_legend - 10)
            
            st.components.v1.html(html_data, height=dynamic_height, scrolling=True)
        except Exception as e:
            st.error(f"Error displaying molecules: {str(e)}")
