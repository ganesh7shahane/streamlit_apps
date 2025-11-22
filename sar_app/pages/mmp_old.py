"""
MMP Analysis Page - Matched Molecular Pairs Analysis
Identifies structural transformations and their activity effects.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from itertools import combinations
from operator import itemgetter
from typing import Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDraw2D
from rdkit.Chem.rdMMPA import FragmentMol
import rdkit.Chem.MolStandardize.rdMolStandardize as uru

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
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()
            text = drawer.GetDrawingText()
            im_text64 = base64.b64encode(text).decode('utf8')
            img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
            return img_str
        except:
            return f"<p style='font-size:10px;'>Image error</p>"
    
    @staticmethod
    def stripplot_base64_image(dist, xlabel='Delta'):
        """
        Plot a distribution as a seaborn stripplot and save the 
        resulting image as a base64 image
        """
        try:
            plt.figure(dpi=150)
            sns.set(rc={'figure.figsize': (3, 1)})
            sns.set_style('whitegrid')
            ax = sns.stripplot(x=dist)
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
        Analyze Matched Molecular Pairs (MMPs) from a CSV file and display results as HTML table.
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
            df['mol'] = df['mol'].apply(uru.GetLargestFragment)
            progress_bar.progress(0.2)
        
        # Decompose molecules to get scaffolds and side chains
        status_text.text("Fragmenting molecules...")
        row_list = []
        total = len(df)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % max(1, total // 20) == 0:
                progress_bar.progress(0.2 + (idx / total * 0.3))
                status_text.text(f"Fragmenting molecules: {idx}/{total}")
            
            smiles = row[smiles_col]
            name = row[id_col]
            activity = row[activity_col]
            mol = row['mol']
            
            frag_list = FragmentMol(mol, maxCuts=max_cuts)
            for _, frag_mol in frag_list:
                pair_list = self.sort_fragments(frag_mol)
                tmp_list = [smiles] + [Chem.MolToSmiles(x) for x in pair_list] + [name, activity]
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
                status_text.text(f"Finding matched pairs: {gidx}/{total_groups} cores")
            
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
        status_text.text(f"Filtering transforms (min occurrence: {min_transform_occurrence})...")
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
            return mmp_df, delta_df, "<p>No transforms found with the specified minimum occurrence. Try reducing min_transform_occurrence parameter.</p>"
        
        mmp_df['idx'] = range(0, len(mmp_df))
        mmp_df['mean_delta'] = [x.mean() for x in mmp_df.Deltas]
        mmp_df['rxn_mol'] = mmp_df.Transform.apply(AllChem.ReactionFromSmarts, useSmiles=True)
        
        # Create index linking delta_df and mmp_df
        transform_dict = dict([(a, b) for a, b in mmp_df[["Transform", "idx"]].values])
        delta_df['idx'] = [transform_dict.get(x) for x in delta_df.Transform]
        
        # Sort by mean_delta and reset index to show it in table
        mmp_df.sort_values("mean_delta", inplace=True, ascending=ascending)
        mmp_df_display = mmp_df.reset_index(drop=True)
        
        status_text.text("Creating results table...")
        progress_bar.progress(0.9)
        
        # Create simple HTML table without images initially
        html_table = mmp_df_display[['idx', 'Transform', 'Count', "mean_delta"]].round(2).head(rows_to_show).to_html(escape=False, index=False)
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Analysis complete! Showing top {rows_to_show} transforms sorted by mean_delta (ascending={ascending})")
        st.info("Use the idx value from the table to visualize specific transforms")
        
        return mmp_df, delta_df, html_table
        """
        Analyze matched molecular pairs from CSV file.
        
        Args:
            csv_path: Path to CSV file
            smiles_col: Column name for SMILES
            id_col: Column name for molecule IDs
            activity_col: Column name for activity values
            min_transform_occurrence: Minimum times a transform must appear
            max_cuts: Maximum number of cuts (1 recommended)
            rows_to_show: Number of top transforms to display
            ascending: Sort by mean delta ascending (True) or descending (False)
            remove_salts: Whether to remove salts from SMILES
            max_molecules: Maximum number of molecules to process (for performance)
            
        Returns:
            mmp_df: DataFrame of all matched pairs
            delta_df: DataFrame of transformations with statistics
            HTML_table: HTML table for display
        """
        # Load data
        df = pd.read_csv(csv_path)
        df = df[[smiles_col, id_col, activity_col]].dropna()
        
        # Limit dataset size for performance
        if len(df) > max_molecules:
            st.warning(f"⚠️ Dataset has {len(df)} molecules. Using first {max_molecules} for performance. Increase 'Max Molecules' in advanced settings to analyze more.")
            df = df.head(max_molecules)
        
        st.info(f"Processing {len(df)} molecules...")
        
        # Remove salts if requested
        if remove_salts:
            df[smiles_col] = df[smiles_col].apply(
                lambda x: max(x.split('.'), key=len) if '.' in x else x
            )
        
        # Fragment molecules - group by CORE only, store chains separately
        fragments_dict = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Fragmenting molecules...")
        total_mols = len(df)
        
        # Update progress less frequently
        update_frequency = max(1, total_mols // 20)  # Update 20 times max
        
        for idx, row in df.iterrows():
            if idx % update_frequency == 0:
                progress_bar.progress(min(idx / total_mols * 0.3, 0.3))
                status_text.text(f"Fragmenting molecules: {idx}/{total_mols}")
            
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                continue
            
            frags = FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)
            for core, chains in frags:
                core_clean = self.remove_map_nums(core)
                chains_clean = self.remove_map_nums(chains)
                chains_sorted = self.sort_fragments(chains_clean)
                
                # Group by CORE only (not core+chains)
                if core_clean not in fragments_dict:
                    fragments_dict[core_clean] = []
                
                fragments_dict[core_clean].append({
                    'smiles': row[smiles_col],
                    'id': row[id_col],
                    'activity': row[activity_col],
                    'chains': chains_sorted  # Store the chains with each molecule
                })
        
        status_text.text(f"Found {len(fragments_dict)} unique cores. Finding matched pairs...")
        progress_bar.progress(0.3)
        
        # Find matched pairs - optimize by only processing cores with 2+ molecules
        mmp_records = []
        cores_to_process = {core: mols for core, mols in fragments_dict.items() if len(mols) >= 2}
        total_cores = len(cores_to_process)
        
        # Update progress less frequently for better performance
        update_frequency = max(1, total_cores // 20)  # Update 20 times max
        
        for core_idx, (core, molecules) in enumerate(cores_to_process.items()):
            if core_idx % update_frequency == 0:
                progress = 0.3 + (core_idx / total_cores * 0.6)
                progress_bar.progress(min(progress, 0.9))
                status_text.text(f"Analyzing cores: {core_idx}/{total_cores} ({len(mmp_records)} pairs found)")
            
            # Compare all pairs of molecules with the same core
            for mol1, mol2 in combinations(molecules, 2):
                chains1 = mol1['chains'].split('.')
                chains2 = mol2['chains'].split('.')
                
                if len(chains1) != len(chains2):
                    continue
                
                # Find differing fragments
                diff_frags = []
                for c1, c2 in zip(sorted(chains1), sorted(chains2)):
                    if c1 != c2:
                        diff_frags.append((c1, c2))
                
                # Only accept pairs with exactly 1 differing fragment
                if len(diff_frags) == 1:
                    frag1, frag2 = diff_frags[0]
                    transform = f"{frag1}>>{frag2}"
                    delta = mol2['activity'] - mol1['activity']
                    
                    mmp_records.append({
                        'smiles1': mol1['smiles'],
                        'smiles2': mol2['smiles'],
                        'id1': mol1['id'],
                        'id2': mol2['id'],
                        'activity1': mol1['activity'],
                        'activity2': mol2['activity'],
                        'delta': delta,
                        'transform': transform,
                        'core': core
                    })
        
        progress_bar.progress(0.9)
        status_text.text(f"Creating DataFrame from {len(mmp_records)} matched pairs...")
        
        if len(mmp_records) == 0:
            progress_bar.empty()
            status_text.empty()
            return pd.DataFrame(), pd.DataFrame(), "<p>No matched pairs found</p>"
        
        # Check if we have too many pairs
        if len(mmp_records) > 100000:
            st.warning(f"⚠️ Found {len(mmp_records)} pairs - this is a lot! Limiting to 100k for performance.")
            mmp_records = mmp_records[:100000]
        
        mmp_df = pd.DataFrame(mmp_records)
        
        progress_bar.progress(0.92)
        
        if mmp_df.empty:
            progress_bar.empty()
            status_text.empty()
            return mmp_df, pd.DataFrame(), "<p>No matched pairs found</p>"
        
        status_text.text(f"Calculating transform statistics for {len(mmp_df['transform'].unique())} unique transforms...")
        progress_bar.progress(0.94)
        
        # Calculate statistics per transformation
        transform_stats = []
        unique_transforms = mmp_df['transform'].unique()
        
        for tidx, transform in enumerate(unique_transforms):
            if tidx % 100 == 0 and tidx > 0:
                status_text.text(f"Processing transform {tidx}/{len(unique_transforms)}...")
            
            subset = mmp_df[mmp_df['transform'] == transform]
            
            if len(subset) < min_transform_occurrence:
                continue
            
            deltas = subset['delta'].values
            transform_stats.append({
                'transform': transform,
                'count': len(subset),
                'mean_delta': np.mean(deltas),
                'median_delta': np.median(deltas),
                'std_delta': np.std(deltas),
                'deltas': deltas.tolist()
            })
        
        status_text.text("Sorting results...")
        progress_bar.progress(0.96)
        
        delta_df = pd.DataFrame(transform_stats)
        
        if delta_df.empty:
            progress_bar.empty()
            status_text.empty()
            return mmp_df, delta_df, f"<p>Found {len(mmp_df)} matched pairs, but no transforms meet minimum occurrence threshold of {min_transform_occurrence}</p>"
        
        delta_df = delta_df.sort_values('mean_delta', ascending=ascending)
        delta_df.reset_index(drop=True, inplace=True)
        
        status_text.text("Creating results table...")
        progress_bar.progress(0.98)
        
        # Create simple HTML table WITHOUT images first (much faster)
        display_df = delta_df.head(rows_to_show).copy()
        
        # Just show the transformation as text for now
        html_df = display_df[['transform', 'count', 'mean_delta', 'median_delta', 'std_delta']]
        html_df.columns = ['Transformation', 'Count', 'Mean Δ', 'Median Δ', 'Std Δ']
        
        HTML_table = html_df.to_html(escape=False, index=True)
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        # Clean up progress indicators
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return mmp_df, delta_df, HTML_table
    
    def visualize_mmp_examples(self,
                              delta_df: pd.DataFrame,
                              query_idx: int,
                              activity_col: str = 'standard_value',
                              n_cols: int = 4,
                              img_size: Tuple[int, int] = (200, 200)):
        """
        Visualize example molecules for a specific transformation.
        
        Args:
            delta_df: DataFrame from analyze_mmps
            query_idx: Index of transformation to visualize
            activity_col: Name of activity column
            n_cols: Number of columns in grid
            img_size: Size of molecule images
        """
        if query_idx >= len(delta_df):
            st.error(f"Index {query_idx} out of range. Max index: {len(delta_df)-1}")
            return
        
        transform = delta_df.iloc[query_idx]['transform']
        subset = self._mmp_df[self._mmp_df['transform'] == transform]
        
        # Create display data
        display_mols = []
        display_smiles = []
        display_ids = []
        display_activities = []
        display_deltas = []
        
        for _, row in subset.iterrows():
            display_smiles.extend([row['smiles1'], row['smiles2']])
            display_ids.extend([row['id1'], row['id2']])
            display_activities.extend([row['activity1'], row['activity2']])
            display_deltas.extend([0, row['delta']])
        
        # Create molecules
        for smi in display_smiles:
            mol = Chem.MolFromSmiles(smi)
            display_mols.append(mol)
        
        # Create dataframe for mols2grid
        display_df = pd.DataFrame({
            'mol': display_mols,
            'smiles': display_smiles,
            'ID': display_ids,
            activity_col: display_activities,
            'delta': display_deltas
        })
        
        # Display with mols2grid
        try:
            import mols2grid
            
            grid = mols2grid.display(
                display_df,
                subset=[f"ID", activity_col, "delta"],
                tooltip=[f"ID", activity_col, "delta", "smiles"],
                size=img_size,
                n_cols=n_cols,
                use_coords=False
            )
            
            st.components.v1.html(grid.data, height=600, scrolling=True)
            
        except ImportError:
            st.error("mols2grid not installed. Install with: pip install mols2grid")
    
    # ==================== STREAMLIT UI ====================
    
    def render(self):
        """Render the MMP Analysis page"""
        st.title("⚛️ MMP Analysis")
        st.markdown("**Matched Molecular Pairs Analysis** - Identify structural transformations and their activity effects")
        
        # File upload section
        st.subheader("📁 Data Input")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV file with molecular data",
                type=['csv'],
                help="CSV should contain SMILES, ID, and activity columns"
            )
        
        with col2:
            use_default = st.checkbox("Use default data (hERG)", value=True)
        
        # Load data
        if uploaded_file is not None:
            csv_path = uploaded_file
            self._df = pd.read_csv(csv_path)
        elif use_default:
            csv_path = "data/hERG.csv"
            try:
                self._df = pd.read_csv(csv_path)
            except FileNotFoundError:
                st.error("Default hERG.csv not found in data/ folder")
                return
        else:
            st.info("Please upload a CSV file or check 'Use default data'")
            return
        
        # Show data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(self._df.head(10))
        
        st.divider()
        
        # Configuration section
        st.subheader("⚙️ Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            smiles_col = st.selectbox(
                "SMILES Column",
                options=self._df.columns.tolist(),
                index=0 if len(self._df.columns) > 0 else 0
            )
        
        with col2:
            id_col = st.selectbox(
                "ID Column",
                options=self._df.columns.tolist(),
                index=1 if len(self._df.columns) > 1 else 0
            )
        
        with col3:
            activity_col = st.selectbox(
                "Activity Column",
                options=self._df.columns.tolist(),
                index=2 if len(self._df.columns) > 2 else 0
            )
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_transform = st.number_input(
                    "Min Transform Occurrence",
                    min_value=1,
                    max_value=20,
                    value=3,
                    help="Minimum times a transform must appear"
                )
            
            with col2:
                max_cuts = st.number_input(
                    "Max Cuts",
                    min_value=1,
                    max_value=3,
                    value=1,
                    help="Maximum number of cuts (1 recommended)"
                )
            
            with col3:
                rows_to_show = st.number_input(
                    "Rows to Display",
                    min_value=5,
                    max_value=50,
                    value=10,
                    help="Number of top transforms to show"
                )
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                ascending = st.checkbox("Sort ascending (by mean Δ)", value=True)
            
            with col5:
                remove_salts = st.checkbox("Remove salts", value=True)
            
            with col6:
                max_molecules = st.number_input(
                    "Max Molecules",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Maximum molecules to process (for performance)"
                )
        
        st.divider()
        
        # Run analysis button
        if st.button("🚀 Run MMP Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing matched molecular pairs..."):
                try:
                    # Save CSV temporarily if uploaded
                    if uploaded_file is not None:
                        temp_path = "/tmp/temp_mmp.csv"
                        self._df.to_csv(temp_path, index=False)
                        csv_path = temp_path
                    
                    # Run analysis
                    self._mmp_df, self._delta_df, self._html_table = self.analyze_mmps(
                        csv_path=csv_path,
                        smiles_col=smiles_col,
                        id_col=id_col,
                        activity_col=activity_col,
                        min_transform_occurrence=min_transform,
                        max_cuts=max_cuts,
                        rows_to_show=rows_to_show,
                        ascending=ascending,
                        remove_salts=remove_salts,
                        max_molecules=max_molecules
                    )
                    
                    if not self._delta_df.empty:
                        st.success(f"✅ Analysis complete! Found {len(self._mmp_df)} matched pairs, {len(self._delta_df)} unique transformations")
                    else:
                        st.warning(f"⚠️ Found {len(self._mmp_df)} matched pairs but no transforms meet the occurrence threshold")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
                    return
        
        # Display results
        if self._html_table is not None and self._delta_df is not None and not self._delta_df.empty:
            st.divider()
            st.subheader("📊 Top Transformations")
            
            # Display simple table
            st.markdown(self._html_table, unsafe_allow_html=True)
            
            # Add button to generate reaction images
            st.divider()
            if st.button("🎨 Generate Reaction Visualizations", help="Create images for the transformations above (may take time)"):
                with st.spinner("Generating reaction images..."):
                    try:
                        display_df = self._delta_df.head(rows_to_show).copy()
                        
                        # Create images
                        st.write("Creating reaction images...")
                        transform_imgs = []
                        prog = st.progress(0)
                        for idx, row in display_df.iterrows():
                            transform_imgs.append(self.rxn_to_base64_image(row['transform'], size=(400, 150)))
                            prog.progress((idx + 1) / len(display_df))
                        prog.empty()
                        
                        display_df['transform_img'] = transform_imgs
                        
                        # Display with images
                        html_df = display_df[['transform_img', 'count', 'mean_delta', 'median_delta', 'std_delta']]
                        html_df.columns = ['Transformation', 'Count', 'Mean Δ', 'Median Δ', 'Std Δ']
                        st.markdown(html_df.to_html(escape=False, index=True), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating images: {str(e)}")
            
            # Download results
            st.divider()
            st.subheader("💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_buffer = BytesIO()
                self._delta_df.drop(columns=['deltas']).to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Transform Statistics",
                    data=csv_buffer.getvalue(),
                    file_name="mmp_transform_stats.csv",
                    mime="text/csv"
                )
            
            with col2:
                csv_buffer2 = BytesIO()
                self._mmp_df.to_csv(csv_buffer2, index=False)
                st.download_button(
                    label="📥 Download All Matched Pairs",
                    data=csv_buffer2.getvalue(),
                    file_name="mmp_all_pairs.csv",
                    mime="text/csv"
                )
            
            # Visualization section
            st.divider()
            st.subheader("🔍 Visualize Examples")
            
            st.markdown(f"Select a transformation index (0 to {len(self._delta_df)-1}) to view example molecules:")
            
            query_idx = st.number_input(
                "Transformation Index",
                min_value=0,
                max_value=len(self._delta_df)-1,
                value=0,
                help="Index from the table above (first row = 0)"
            )
            
            if st.button("Show Molecules", use_container_width=True):
                with st.spinner("Generating molecule grid..."):
                    self.visualize_mmp_examples(
                        delta_df=self._delta_df,
                        query_idx=query_idx,
                        activity_col=activity_col,
                        n_cols=4,
                        img_size=(200, 200)
                    )
