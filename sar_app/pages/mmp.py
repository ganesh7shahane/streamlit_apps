"""
MMP Analysis Page - Matched Molecular Pairs Analysis
Uses exact algorithm from Jupyter notebook
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from itertools import combinations
from operator import itemgetter
from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import seaborn as sns

from ..base import BaseAnalyzer
from ..config import AppConfig


class MMPAnalyzer(BaseAnalyzer):
    """MMP (Matched Molecular Pairs) Analysis Tool"""
    
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self._mmp_df = None
        self._delta_df = None
        self._html_table = None
        
    # ==================== HELPER FUNCTIONS (from notebook) ====================
    
    @staticmethod
    def get_largest_fragment(mol):
        """Get the largest fragment from a molecule (removes salts)"""
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 1:
            return mol
        # Return the fragment with the most atoms
        return max(frags, key=lambda m: m.GetNumAtoms())
    
    @staticmethod
    def remove_map_nums(mol):
        """Remove atom map numbers from a molecule"""
        for atm in mol.GetAtoms():
            atm.SetAtomMapNum(0)
    
    @staticmethod
    def sort_fragments(mol):
        """
        Transform a molecule with multiple fragments into a list of molecules 
        that is sorted by number of atoms from largest to smallest
        """
        frag_list = list(Chem.GetMolFrags(mol, asMols=True))
        [MMPAnalyzer.remove_map_nums(x) for x in frag_list]
        frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_num_atoms_list.sort(key=itemgetter(0), reverse=True)
        return [x[1] for x in frag_num_atoms_list]
    
    @staticmethod
    def rxn_to_base64_image(rxn):
        """Convert an RDKit reaction to a base64 encoded image"""
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 200)
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()
            text = drawer.GetDrawingText()
            im_text64 = base64.b64encode(text).decode('utf8')
            img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
            return img_str
        except:
            return f"<p style='font-size:10px;'>Image error</p>"
    
    @staticmethod
    def stripplot_base64_image(dist, xlabel='Delta', color='#1f77b4'):
        """
        Plot a distribution as a seaborn stripplot and save the 
        resulting image as a base64 image
        """
        try:
            plt.figure(dpi=150)
            sns.set(rc={'figure.figsize': (3, 1)})
            sns.set_style('whitegrid')
            ax = sns.stripplot(x=dist, color=color)
            ax.axvline(0, ls="--", c="red")
            ax.set_xlim(-5, 5)
            ax.set_xlabel(xlabel, fontsize=10)
            s = BytesIO()
            plt.savefig(s, format='png', bbox_inches="tight")
            plt.close()
            s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
            return '<img align="left" src="data:image/png;base64,%s">' % s
        except:
            plt.close('all')
            return "<p>Plot error</p>"
    
    # ==================== MAIN ANALYSIS FUNCTION (from notebook) ====================
    
    def analyze_mmps(self,
                     csv_path: str,
                     smiles_col: str = 'SMILES',
                     id_col: str = 'ID',
                     activity_col: str = 'pIC50',
                     min_transform_occurrence: int = 3,
                     max_cuts: int = 1,
                     rows_to_show: int = 10,
                     ascending: bool = True,
                     remove_salts: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        Analyze Matched Molecular Pairs (MMPs) from a CSV file.
        (Exact implementation from Jupyter notebook)
        """
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading data...")
        df = pd.read_csv(csv_path)
        st.info(f"Loaded {len(df)} molecules")
        progress_bar.progress(0.1)
        
        # Add RDKit molecule column
        status_text.text("Converting SMILES to molecules...")
        df['mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
        df = df.dropna(subset=['mol'])  # Remove invalid SMILES
        progress_bar.progress(0.15)
        
        # Remove salts if requested
        if remove_salts:
            status_text.text("Removing salts...")
            df['mol'] = df['mol'].apply(self.get_largest_fragment)
            progress_bar.progress(0.2)
        
        # Decompose molecules to get scaffolds and side chains
        status_text.text("Fragmenting molecules...")
        row_list = []
        total = len(df)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % max(1, total // 20) == 0:
                progress_bar.progress(0.2 + (idx / total * 0.3))
                status_text.text(f"Fragmenting: {idx}/{total}")
            
            smiles = row[smiles_col]
            name = row[id_col]
            activity = row[activity_col]
            mol = row['mol']
            
            frag_list = FragmentMol(mol, maxCuts=max_cuts)
            for _, frag_mol in frag_list:
                pair_list = self.sort_fragments(frag_mol)
                # Only take first two fragments (Core and R_group)
                # Skip if fragmentation produced unexpected number of fragments
                if len(pair_list) >= 2:
                    tmp_list = [smiles, Chem.MolToSmiles(pair_list[0]), Chem.MolToSmiles(pair_list[1]), name, activity]
                    row_list.append(tmp_list)
        
        row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", activity_col])
        st.info(f"Generated {len(row_df)} fragment pairs")
        progress_bar.progress(0.5)
        
        # Collect pairs with the same scaffold
        status_text.text("Finding matched pairs...")
        delta_list = []
        groups = list(row_df.groupby("Core"))
        total_groups = len(groups)
        
        for gidx, (k, v) in enumerate(groups):
            if gidx % max(1, total_groups // 20) == 0:
                progress_bar.progress(0.5 + (gidx / total_groups * 0.3))
                status_text.text(f"Pairing: {gidx}/{total_groups} cores")
            
            if len(v) > 2:
                for a, b in combinations(range(0, len(v)), 2):
                    reagent_a = v.iloc[a]
                    reagent_b = v.iloc[b]
                    if reagent_a.SMILES == reagent_b.SMILES:
                        continue
                    reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                    delta = reagent_b[activity_col] - reagent_a[activity_col]
                    delta_list.append(
                        list(reagent_a.values) + list(reagent_b.values) +
                        [f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}", delta]
                    )
        
        # Create delta dataframe
        cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", f"{activity_col}_1",
                "SMILES_2", "Core_2", "R_group_2", "Name_2", f"{activity_col}_2",
                "Transform", "Delta"]
        delta_df = pd.DataFrame(delta_list, columns=cols)
        st.info(f"Found {len(delta_df)} molecule pairs")
        progress_bar.progress(0.8)
        
        # Collect frequently occurring pairs
        status_text.text(f"Filtering transforms (min: {min_transform_occurrence})...")
        mmp_list = []
        for k, v in delta_df.groupby("Transform"):
            if len(v) >= min_transform_occurrence:
                mmp_list.append([k, len(v), v.Delta.values])
        
        mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas"])
        st.info(f"Found {len(mmp_df)} frequent transforms")
        progress_bar.progress(0.85)
        
        if len(mmp_df) == 0:
            progress_bar.empty()
            status_text.empty()
            return mmp_df, delta_df, "<p>No transforms found. Try reducing min_transform_occurrence.</p>"
        
        mmp_df['idx'] = range(0, len(mmp_df))
        mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
        mmp_df['rxn_mol'] = mmp_df.Transform.apply(AllChem.ReactionFromSmarts, useSmiles=True)
        
        # Create index linking delta_df and mmp_df
        transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
        delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
        
        # Sort by mean_delta
        mmp_df.sort_values("mean_delta", inplace=True, ascending=ascending)
        mmp_df_display = mmp_df.reset_index(drop=True)
        
        status_text.text("Creating table...")
        progress_bar.progress(0.9)
        
        # Create simple HTML table (no images yet)
        html_table = mmp_df_display[['idx', 'Transform', 'Count', "mean_delta"]].round(2).head(rows_to_show).to_html(escape=False, index=False)
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Complete! Top {rows_to_show} transforms (ascending={ascending})")
        
        return mmp_df, delta_df, html_table
    
    def visualize_mmp_examples(self, query_idx: int, activity_col: str = 'pIC50'):
        """Visualize example molecules for a specific transformation"""
        if self._delta_df is None or self._mmp_df is None:
            st.error("Please run analysis first")
            return
        
        # Get the molecules for this idx
        subset = self._delta_df[self._delta_df['idx'] == query_idx].copy()
        
        if len(subset) == 0:
            st.error(f"No data found for idx {query_idx}")
            return
        
        st.subheader(f"Examples for Transform idx={query_idx}")
        
        # Check which columns actually exist in the subset
        available_cols = [col for col in ['SMILES_1', 'SMILES_2', f'{activity_col}_1', f'{activity_col}_2', 'Delta'] if col in subset.columns]
        st.dataframe(subset[available_cols].head(20))
        
        # Try to use mols2grid if available
        try:
            import mols2grid
            
            mols = []
            labels = []
            smiles_list = []
            names_list = []
            activities_list = []
            
            for _, row in subset.head(10).iterrows():
                # Add first molecule
                mol1 = Chem.MolFromSmiles(row['SMILES_1'])
                if mol1:
                    mols.append(mol1)
                    labels.append(f"{row['Name_1']}: {row[f'{activity_col}_1']:.2f}")
                    smiles_list.append(row['SMILES_1'])
                    names_list.append(row['Name_1'])
                    activities_list.append(row[f'{activity_col}_1'])
                
                # Add second molecule
                mol2 = Chem.MolFromSmiles(row['SMILES_2'])
                if mol2:
                    mols.append(mol2)
                    labels.append(f"{row['Name_2']}: {row[f'{activity_col}_2']:.2f}")
                    smiles_list.append(row['SMILES_2'])
                    names_list.append(row['Name_2'])
                    activities_list.append(row[f'{activity_col}_2'])
            
            grid_df = pd.DataFrame({
                'mol': mols, 
                'label': labels,
                'SMILES': smiles_list,
                'Name': names_list,
                activity_col: activities_list
            })
            
            # Select legend columns with multiselect
            available_mmp_cols = [col for col in grid_df.columns if col != 'mol']
            default_mmp_legends = ['label'] if 'label' in available_mmp_cols else available_mmp_cols[:min(1, len(available_mmp_cols))]
            
            selected_mmp_legend_cols = st.multiselect(
                "Choose columns to display as legends:",
                options=available_mmp_cols,
                default=default_mmp_legends,
                key=f"mmp_legend_cols_{query_idx}"
            )
            
            if not selected_mmp_legend_cols:
                st.warning("⚠️ No legend columns selected. Please select at least one column to display.")
                return
            
            # Create tooltip with all available columns (except 'mol')
            tooltip_cols = [col for col in grid_df.columns if col != 'mol']
            
            # Build subset list: img first, then selected legend columns
            subset_list = ['img'] + selected_mmp_legend_cols
            
            grid = mols2grid.display(grid_df, subset=subset_list, tooltip=tooltip_cols, n_cols=4, size=(200, 200))
            
            # Calculate dynamic height based on number of legend columns
            base_height = 670
            height_per_legend = 80
            dynamic_height = base_height + (len(selected_mmp_legend_cols) * height_per_legend - 10)
            
            st.components.v1.html(grid.data, height=dynamic_height, scrolling=True)
            
        except ImportError:
            st.warning("mols2grid not available - showing table only")
        except Exception as e:
            st.error(f"Error displaying grid: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
    
    # ==================== STREAMLIT UI ====================
    
    def render(self):
        """Render the MMP Analysis page"""
        st.title("⚔️ MMP Analysis")
        st.markdown("**Matched Molecular Pairs** - Identify transformations and activity effects")
        st.markdown("- **:red[Before you use]** this tool, make sure your CSV file contains valid SMILES strings and activity data is on log scale")
        st.markdown("- For this, you may use the **DataFrame Wizard** page to clean and preprocess your data.")
        
        # Initialize session state for results persistence
        if 'mmp_results' not in st.session_state:
            st.session_state.mmp_results = None
        
        # File upload
        st.subheader("📁 Data Input")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        with col2:
            use_default = st.checkbox("Use sample hERG dataset", value=True)
        
        # Load data
        if uploaded_file is not None:
            csv_path = uploaded_file
            self._df = pd.read_csv(csv_path)
        elif use_default:
            csv_path = "data/hERG.csv"
            try:
                self._df = pd.read_csv(csv_path)
            except FileNotFoundError:
                st.error("hERG.csv not found in data/ folder")
                return
        else:
            st.info("Upload CSV or check 'Use hERG data'")
            return
        
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(self._df.head(10))
        
        st.divider()
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            smiles_col = st.selectbox("SMILES Column", self._df.columns, index=0)
        
        with col2:
            id_col = st.selectbox("ID Column", self._df.columns, index=1)
        
        with col3:
            activity_col = st.selectbox("Activity Column", self._df.columns, index=2)
        
        with st.expander("🔧 Advanced"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_transform = st.number_input("Min Occurrence", 1, 20, 3)
            
            with col2:
                max_cuts = st.number_input("Max Cuts", 1, 3, 1)
            
            with col3:
                rows_to_show = st.number_input("Rows to Show", 5, 50, 20)
            
            col4, col5 = st.columns(2)
            
            with col4:
                ascending = st.checkbox("Sort ascending", value=True)
            
            with col5:
                remove_salts = st.checkbox("Remove salts")
        
        st.divider()
        
        # Run analysis
        if st.button("🚀 Run MMP Analysis", type="primary"):
            try:
                # Save temp if uploaded
                if uploaded_file is not None:
                    temp_path = "/tmp/temp_mmp.csv"
                    self._df.to_csv(temp_path, index=False)
                    csv_path = temp_path
                
                self._mmp_df, self._delta_df, self._html_table = self.analyze_mmps(
                    csv_path=csv_path,
                    smiles_col=smiles_col,
                    id_col=id_col,
                    activity_col=activity_col,
                    min_transform_occurrence=min_transform,
                    max_cuts=max_cuts,
                    rows_to_show=rows_to_show,
                    ascending=ascending,
                    remove_salts=remove_salts
                )
                
                # Store results in session state
                st.session_state.mmp_results = {
                    'mmp_df': self._mmp_df,
                    'delta_df': self._delta_df,
                    'html_table': self._html_table,
                    'activity_col': activity_col,
                    'rows_to_show': rows_to_show
                }
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Details"):
                    st.code(traceback.format_exc())
                return
        
        # Load results from session state if available
        if st.session_state.mmp_results is not None:
            self._mmp_df = st.session_state.mmp_results['mmp_df']
            self._delta_df = st.session_state.mmp_results['delta_df']
            self._html_table = st.session_state.mmp_results['html_table']
            activity_col = st.session_state.mmp_results['activity_col']
            rows_to_show = st.session_state.mmp_results['rows_to_show']
        
        # Display results
        if self._html_table and self._delta_df is not None and not self._delta_df.empty:
            st.divider()
            st.subheader("📊 Top Transformations")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                show_images = st.checkbox("Show reaction images (slower)", value=True)
            
            with col2:
                show_all_transforms = st.checkbox("Show all transforms")
            
            with col3:
                sort_by = st.selectbox("Sort by", options=["mean_delta", "Count"], index=0)
            
            with col4:
                sort_order = st.selectbox("Order", options=["Ascending", "Descending"], index=0)
            
            with col5:
                stripplot_color = st.color_picker("Strip plot color", "#1f77b4", key="mmp_stripplot_color")
            
            # Sort the dataframe based on selection
            ascending_order = (sort_order == "Ascending")
            
            if sort_by == "Count":
                display_mmp_df = self._mmp_df.sort_values('Count', ascending=ascending_order).reset_index(drop=True)
            else:
                display_mmp_df = self._mmp_df.sort_values('mean_delta', ascending=ascending_order).reset_index(drop=True)
            
            # Determine how many rows to display
            display_count = len(display_mmp_df) if show_all_transforms else rows_to_show
            
            if show_images:
                with st.spinner("Generating reaction images..."):
                    try:
                        display_df = display_mmp_df.head(display_count).copy()
                        
                        imgs = []
                        plots = []
                        prog = st.progress(0)
                        status = st.empty()
                        
                        for i, row in enumerate(display_df.iterrows()):
                            status.text(f"Generating image {i+1}/{len(display_df)}...")
                            imgs.append(self.rxn_to_base64_image(row[1]['rxn_mol']))
                            plots.append(self.stripplot_base64_image(row[1]['Deltas'], xlabel=f'Δ {activity_col}', color=stripplot_color))
                            prog.progress((i + 1) / len(display_df))
                        
                        prog.empty()
                        status.empty()
                        
                        display_df['MMP Transform'] = imgs
                        display_df['Delta Distribution'] = plots
                        
                        html_with_imgs = display_df[['idx', 'MMP Transform', 'Count', 'mean_delta', 'Delta Distribution']].round(2).to_html(escape=False, index=False)
                        st.markdown(html_with_imgs, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Image generation error: {str(e)}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
                        # Fall back to text table
                        display_df_text = display_mmp_df.head(display_count).copy()
                        html_table_fallback = display_df_text[['idx', 'Transform', 'Count', "mean_delta"]].round(2).to_html(escape=False, index=False)
                        st.markdown(html_table_fallback, unsafe_allow_html=True)
            else:
                # Show simple text table
                display_df_text = display_mmp_df.head(display_count).copy()
                html_table_simple = display_df_text[['idx', 'Transform', 'Count', "mean_delta"]].round(2).to_html(escape=False, index=False)
                st.markdown(html_table_simple, unsafe_allow_html=True)
                st.info("💡 Check 'Show reaction images' above to see 2D structures")
            
            # Download
            st.divider()
            st.subheader("💾 Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv1 = self._mmp_df.drop(columns=['Deltas', 'rxn_mol']).to_csv(index=False)
                st.download_button("📥 Transforms", csv1, "mmp_transforms.csv", "text/csv", key="download_transforms")
            
            with col2:
                csv2 = self._delta_df.to_csv(index=False)
                st.download_button("📥 All Pairs", csv2, "mmp_pairs.csv", "text/csv", key="download_pairs")
            
            # Visualize specific transform
            st.divider()
            st.subheader("🔍 Visualise Transform")
            
            # Initialize session state for visualization persistence
            if 'mmp_query_idx' not in st.session_state:
                st.session_state.mmp_query_idx = None
            
            query_idx = st.number_input(
                "Enter idx from table above",
                min_value=0,
                max_value=len(self._mmp_df)-1 if self._mmp_df is not None else 0,
                value=0
            )
            
            if st.button("Show Examples"):
                st.session_state.mmp_query_idx = query_idx
            
            # Show visualization if query_idx is set in session state
            if st.session_state.mmp_query_idx is not None:
                self.visualize_mmp_examples(st.session_state.mmp_query_idx, activity_col)
