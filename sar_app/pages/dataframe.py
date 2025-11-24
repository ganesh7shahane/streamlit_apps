"""
DataFrame visualisation Analyzer
Handles CSV analysis, statistics, plotting, and molecular visualisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import mols2grid
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, PandasTools
from io import StringIO

from sar_app.base import BaseAnalyzer
from sar_app.config import AppConfig
from sar_app.utils import MoleculeUtils, DataFrameUtils, PlotUtils


class DataFrameAnalyzer(BaseAnalyzer):
    """DataFrame visualisation and analysis page."""
    
    def render(self):
        """Render the DataFrame visualisation page."""
        st.title("⚛️ Molecule DataFrame Visualisation")
        
        # Introduction
        self._display_intro()
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        # Load data
        if not self._load_data(uploaded_file, self.config.DEFAULT_DATAFRAME_URL):
            return
        
        # Data cleaning options
        self._data_cleaning_section()
        
        # Display data preview
        self._display_data_preview()
        
        # Statistics
        self._display_statistics()
        
        # visualisation
        self._visualisation_section()
        
        # Molecular grid
        if self.smiles_col:
            self._molecular_grid_section()
        
        # Descriptors
        if self.smiles_col:
            self._descriptor_section()
    
    def _display_intro(self):
        """Display page introduction."""
        st.markdown("""
        Clean and analyze CSV files containing SMILES and their properties:
        - :red[View & Clean] the DataFrame
        - :red[Compute] statistics on numerical columns
        - :red[Plot] histograms, barplots, correlation heatmaps
        - :red[Visualise] 2D molecular structures
        - :red[Calculate] physchem descriptors, fingerprints & ADMET properties
        """)
    
    def _data_cleaning_section(self):
        """Handle data cleaning options."""
        st.subheader("Data Cleaning Options")
        
        if not self.smiles_col:
            st.info("No SMILES column found - data cleaning options unavailable")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            remove_invalid = st.checkbox("Remove invalid SMILES not read by RDKit", value=True)
        
        with col2:
            remove_duplicates = st.checkbox("Remove duplicate molecules (InChI)", value=False)
        
        with col3:
            keep_largest_fragment = st.checkbox("Keep largest fragment (remove salts/ions)", value=False)
        
        # New row for activity conversion
        st.markdown("")  # Add some spacing
        col4, col5, col6 = st.columns(3)
        
        with col4:
            convert_activity = st.checkbox("Convert activity (nM) to log scale", value=False)
        
        activity_col_selected = None
        if convert_activity:
            # Get numeric columns for activity selection
            numeric_cols = DataFrameUtils.get_numeric_columns(self._df)
            if numeric_cols:
                with col5:
                    activity_col_selected = st.selectbox(
                        "Select activity column (nM)",
                        options=numeric_cols,
                        key="activity_col_conversion"
                    )
        
        # Apply cleaning operations
        if remove_invalid or remove_duplicates or keep_largest_fragment or convert_activity:
            original_size = len(self._df)
            
            # Keep largest fragment (remove salts)
            if keep_largest_fragment:
                with st.spinner("Extracting largest fragments..."):
                    def get_largest_fragment(smiles):
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is None:
                                return smiles
                            # Get fragments
                            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                            if len(frags) == 1:
                                return smiles
                            # Find largest by heavy atom count
                            largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
                            return Chem.MolToSmiles(largest)
                        except:
                            return smiles
                    
                    self._df.loc[:, self.smiles_col] = self._df[self.smiles_col].apply(get_largest_fragment)
                    st.info("✅ Extracted largest fragments (salts removed)")
            
            # Remove invalid SMILES
            if remove_invalid:
                valid_mask = self._df[self.smiles_col].apply(
                    lambda x: MoleculeUtils.smiles_to_mol(x) is not None
                )
                self._df = self._df[valid_mask].reset_index(drop=True)
                invalid_removed = original_size - len(self._df)
                if invalid_removed > 0:
                    st.warning(f"⚠️ Removed {invalid_removed} invalid SMILES")
                    original_size = len(self._df)  # Update for next operation
            
            # Remove duplicates by InChI
            if remove_duplicates:
                with st.spinner("Generating InChI keys for duplicate detection..."):
                    self._df.loc[:, 'InChI_temp'] = self._df[self.smiles_col].apply(
                        lambda x: Chem.MolToInchi(mol) if (mol := Chem.MolFromSmiles(x)) else None
                    )
                    before_dedup = len(self._df)
                    self._df = self._df.dropna(subset=['InChI_temp'])
                    self._df = self._df.drop_duplicates(subset=['InChI_temp']).reset_index(drop=True)
                    self._df = self._df.drop(columns=['InChI_temp'])
                    duplicates_removed = before_dedup - len(self._df)
                    if duplicates_removed > 0:
                        st.warning(f"⚠️ Removed {duplicates_removed} duplicate molecules")
            
            # Final summary
            total_removed = original_size - len(self._df)
            if total_removed > 0:
                st.success(f"✅ Cleaned dataset: {original_size} → {len(self._df)} molecules ({total_removed} removed)")
        
        # Convert activity to log scale
        if convert_activity and activity_col_selected:
            with st.spinner("Converting activity to log scale..."):
                try:
                    # Create new column name
                    new_col_name = f"pActivity_{activity_col_selected}"
                    
                    # Convert nM to pActivity: pActivity = -log10(Activity_nM * 1e-9)
                    # This is equivalent to: pActivity = 9 - log10(Activity_nM)
                    self._df[new_col_name] = self._df[activity_col_selected].apply(
                        lambda x: 9 - np.log10(x) if pd.notnull(x) and x > 0 else np.nan
                    )
                    
                    # Count NaN values created
                    nan_count = self._df[new_col_name].isna().sum() - self._df[activity_col_selected].isna().sum()
                    
                    if nan_count > 0:
                        st.warning(f"⚠️ {nan_count} invalid/zero values converted to NaN in log scale")
                    
                    st.success(f"✅ Created new column '{new_col_name}' with pActivity values")
                    st.info(f"Formula: pActivity = 9 - log10({activity_col_selected})")
                except Exception as e:
                    st.error(f"❌ Error converting to log scale: {str(e)}")
    
    def _display_data_preview(self):
        """Display data preview."""
        st.subheader("Dataset Preview")
        
        # Get current dataframe length (after any cleaning operations)
        current_df_length = len(self._df)
        max_rows = min(100, current_df_length)
        default_rows = min(5, current_df_length)
        
        rows = st.slider("Rows to display", 3, max_rows, default_rows, key="preview_rows_slider")
        
        # Display the dataframe directly without molecule rendering
        preview_df = self._df.head(rows)
        st.dataframe(preview_df, width='stretch')
        st.info(f"📊 Total: {current_df_length} molecules, {len(self._df.columns)} columns")
    
    def _display_statistics(self):
        """Display statistical analysis."""
        numeric_cols = DataFrameUtils.get_numeric_columns(self._df)
        
        if not numeric_cols:
            st.info("No numeric columns found")
            return
        
        st.subheader("📊 Statistical Analysis")
        
        # NaN counts
        with st.expander("NaN Analysis"):
            nan_df = self._df.isna().sum().reset_index()
            nan_df.columns = ['Column', 'NaN Count']
            st.dataframe(nan_df, width='stretch')
        
        # Descriptive stats
        with st.expander("Descriptive Statistics", expanded=True):
            st.dataframe(self._df[numeric_cols].describe(), width='stretch')
    
    def _visualisation_section(self):
        """Handle all visualisation options."""
        numeric_cols = DataFrameUtils.get_numeric_columns(self._df)
        
        if len(numeric_cols) < 1:
            return
        
        st.markdown("---")
        st.subheader("📈 Visualisations")
        
        # Multiple histograms
        if len(numeric_cols) > 1:
            self._plot_multiple_histograms(numeric_cols)
        
        # Single column plot
        self._plot_single_column(numeric_cols)
        
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            self._plot_correlation_heatmap(numeric_cols)
        
        # Interactive scatter
        if len(numeric_cols) >= 2:
            self._plot_interactive_scatter(numeric_cols)
    
    def _plot_multiple_histograms(self, numeric_cols):
        """Plot multiple histograms."""
        with st.expander("Multi-Histogram Plot"):
            selected_cols = st.multiselect(
                "Select columns", numeric_cols, default=numeric_cols[:min(6, len(numeric_cols))]
            )
            
            if selected_cols:
                col1, col2, col3 = st.columns(3)
                with col1:
                    bins = st.slider("Bins", 5, 100, 30)
                with col2:
                    color = st.color_picker("Color", self.config.HISTOGRAM_COLOR)
                with col3:
                    n_cols = st.slider("Columns", 1, 4, 3)
                
                n_rows = (len(selected_cols) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows), dpi=150)
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                
                for idx, col in enumerate(selected_cols):
                    axes[idx].hist(self._df[col].dropna(), bins=bins, color=color, edgecolor='black')
                    axes[idx].set_title(col)
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                
                # Hide extra subplots
                for idx in range(len(selected_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    def _plot_single_column(self, numeric_cols):
        """Plot single column histogram or barplot."""
        with st.expander("Single Column Plot"):
            col1, col2 = st.columns(2)
            with col1:
                plot_type = st.selectbox("Plot type", ["Histogram", "Bar Plot"])
            with col2:
                selected_col = st.selectbox("Column", numeric_cols)
            
            if plot_type == "Histogram":
                bins = st.slider("Number of bins", 5, 100, 15, key="single_bins")
                color = st.color_picker("Color", self.config.HISTOGRAM_COLOR, key="single_color")
                
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.hist(self._df[selected_col].dropna(), bins=bins, color=color, edgecolor='black')
                ax.set_title(f"Histogram of {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close()
            else:
                col_bin, col_color = st.columns(2)
                with col_bin:
                    bin_width = st.number_input(
                        "Bin width",
                        min_value=0.01,
                        value=float((self._df[selected_col].max() - self._df[selected_col].min()) / 10)
                    )
                with col_color:
                    color = st.color_picker("Color", self.config.BARPLOT_COLOR, key="bar_color")
                
                # Allow user to choose automatic binning by width or manually define bin edges
                bin_mode = st.selectbox("Bin mode", ["Auto (use bin width)", "Manual edges"], key="bin_mode")

                try:
                    min_val = float(self._df[selected_col].min())
                    max_val = float(self._df[selected_col].max())

                    if bin_mode == "Auto (use bin width)":
                        # Ensure bin_width is > 0 and create bins using numpy.arange
                        bin_width = float(max(bin_width, 1e-12))
                        bins = np.arange(min_val, max_val + bin_width, bin_width)
                    else:
                        edges_input = st.text_input(
                            "Enter bin edges (comma-separated), e.g. 0,1,2,3",
                            value=f"{min_val},{max_val}",
                            key="manual_bin_edges"
                        )
                        # Parse edges into floats
                        edges = [float(x.strip()) for x in edges_input.split(",") if x.strip() != ""]
                        if len(edges) < 2:
                            st.error("Please provide at least two numeric edges for manual bins.")
                            return
                        bins = np.array(sorted(edges))

                    # Create binned column and compute counts
                    self._df.loc[:, 'binned'] = pd.cut(self._df[selected_col], bins=bins, include_lowest=True)
                    counts = self._df['binned'].value_counts().sort_index()

                    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                    # Draw bars manually so we can place labels on top
                    bars = ax.bar(range(len(counts)), counts.values, color=color, edgecolor='black')
                    ax.set_title(f"Bar Plot of {selected_col}")
                    ax.set_xlabel("Bins")
                    ax.set_ylabel("Count")
                    ax.set_xticks(range(len(counts)))
                    ax.set_xticklabels([str(i) for i in counts.index], rotation=45)

                    # Annotate counts on top of each bar
                    for rect in bars:
                        height = rect.get_height()
                        ax.annotate(
                            f"{int(height)}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom'
                        )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except ValueError:
                    st.error("Invalid bin specification. Ensure numeric values are provided.")
                except Exception as e:
                    st.error(f"Error creating bar plot: {e}")
                finally:
                    # Clean up temporary column if it exists
                    if 'binned' in self._df.columns:
                        self._df = self._df.drop(columns=['binned'])
    
    def _plot_correlation_heatmap(self, numeric_cols):
        """Plot correlation heatmap with selectable columns."""
        with st.expander("Correlation Heatmap"):
            st.write("Choose which numeric columns to include in the correlation heatmap.")
            # Column selection expander with checkboxes (default: all selected)
            with st.expander("Select / Deselect columns", expanded=False):
                selected_cols = []
                # Layout checkboxes in two columns for compactness
                col_a, col_b = st.columns(2)
                for i, col in enumerate(numeric_cols):
                    target = col_a if (i % 2 == 0) else col_b
                    with target:
                        include = st.checkbox(col, value=True, key=f"corr_include_{col}")
                        if include:
                            selected_cols.append(col)
            
            if not selected_cols:
                st.info("No columns selected. Please select at least two numeric columns to compute a correlation heatmap.")
                return
            
            if len(selected_cols) == 1:
                st.info("Only one column selected. Select at least two numeric columns to compute correlations.")
                return
            
            # Compute correlation and plot
            try:
                corr_df = self._df[selected_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                sns.heatmap(
                    corr_df,
                    annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, cbar_kws={'label': 'Correlation'},
                    vmin=-1, vmax=1
                )
                ax.set_title("Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
    
    def _plot_interactive_scatter(self, numeric_cols):
        """Plot interactive scatter with plotly."""
        with st.expander("Interactive Scatter Plot"):
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="scatter_y")
            with col3:
                color_col = st.selectbox("Color by", numeric_cols, index=min(2, len(numeric_cols)-1), key="scatter_color")
            
            fig = px.scatter(
                self._df, x=x_col, y=y_col, color=color_col,
                hover_data=[self._df.index], trendline="ols"
            )
            st.plotly_chart(fig, width='stretch')
    
    def _molecular_grid_section(self):
        """Display molecular grid with filtering options."""
        st.markdown("---")
        st.subheader("🧬 2D Molecule Visualisation")
        
        numeric_cols = DataFrameUtils.get_numeric_columns(self._df)
        
        # Filter based on numerical columns if available
        if len(numeric_cols) > 0:
            st.write("💡 **Filter molecules by property ranges**")
            #st.info("To visualize all molecules, leave the sliders at their default values.")
            
            # Remove columns with constant values
            numeric_cols_variable = [col for col in numeric_cols if self._df[col].nunique() > 1]
            
            if numeric_cols_variable:
                with st.expander("🎚️ Property Filter Sliders", expanded=True):
                    filters = {}
                    col_ranges = {}  # Store original min/max for each column
                    col1, col2, col3 = st.columns(3, gap="medium")
                    
                    # Create sliders for each numeric column in 3 columns
                    for idx, col in enumerate(numeric_cols_variable):
                        # Distribute sliders across 3 columns
                        target_col = col1 if idx % 3 == 0 else (col2 if idx % 3 == 1 else col3)
                        with target_col:
                            min_val = float(self._df[col].min())
                            max_val = float(self._df[col].max())
                            col_ranges[col] = (min_val, max_val)
                            filters[col] = st.slider(
                                f"{col} range",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"mol_filter_{col}"
                            )
                
                # Apply filters - only filter if slider was moved from default
                filtered_df = self._df.copy()
                for col, (filter_min, filter_max) in filters.items():
                    original_min, original_max = col_ranges[col]
                    # Only apply filter if user changed the slider from default
                    if filter_min != original_min or filter_max != original_max:
                        filtered_df = filtered_df[(filtered_df[col] >= filter_min) & (filtered_df[col] <= filter_max)]
                
                st.write(f"📊 **Remaining molecules after filtering:** {len(filtered_df)} / {len(self._df)}")
            else:
                filtered_df = self._df.copy()
                st.info("All numerical columns have constant values. Showing all molecules.")
        else:
            filtered_df = self._df.copy()
            st.info("No numerical columns found. Showing all molecules without filtering.")
        
        # Display options
        #st.markdown("---")
        display_option = st.radio(
            "Display options:",
            ("Show filtered molecules", "Show all molecules (ignore filters)"),
            index=0,
            key="mol_display_option"
        )
        
        # Select which dataframe to display
        if display_option == "Show all molecules (ignore filters)":
            display_df = self._df.copy()
            st.info(f"Displaying all {len(display_df)} molecules from the dataset")
        else:
            display_df = filtered_df.copy()
            st.info(f"Displaying {len(display_df)} filtered molecules")
        
        # Select legend columns with multiselect
        available_cols = [col for col in display_df.columns if col != 'mol']
        
        # Default to first two columns if available
        default_legends = available_cols[:min(2, len(available_cols))]
        
        selected_legend_cols = st.multiselect(
            "Choose columns to display as legends:",
            options=available_cols,
            default=default_legends,
            key="mol_legend_cols"
        )
        
        if not selected_legend_cols:
            st.warning("⚠️ No legend columns selected. Please select at least one column to display.")
            return
        
        # Generate molecular grid
        with st.spinner("Generating molecular structures..."):
            display_df.loc[:, 'mol'] = display_df[self.smiles_col].apply(
            lambda x: MoleculeUtils.smiles_to_mol(x)
            )
            
            # Do not limit molecules — display all
            display_df_limited = display_df
        
        try:
            # Format numeric legend columns to 2 decimal places for display
            df_for_display = display_df_limited.copy()
            for col in selected_legend_cols:
                if col in df_for_display.columns and pd.api.types.is_numeric_dtype(df_for_display[col]):
                    df_for_display.loc[:, col] = df_for_display[col].apply(
                        lambda v: f"{v:.2f}" if pd.notnull(v) else ""
                    )

            # Create tooltip with all available columns (except 'mol' and 'img')
            tooltip_cols = [col for col in df_for_display.columns if col not in ['mol', 'img']]
            
            # Build subset list: img first, then selected legend columns
            subset_list = ["img"] + selected_legend_cols
            
            raw_html = mols2grid.display(
                df_for_display,
                subset=subset_list,
                tooltip=tooltip_cols,
                mol_col='mol',
                size=self.config.MOL_GRID_SIZE,
                n_items_per_page=self.config.MOL_ITEMS_PER_PAGE,
                fixedBondLength=25,
                clearBackground=False
            )._repr_html_()
            
            # Calculate dynamic height based on number of legend columns
            # Base height + extra height per legend column
            base_height = 670
            height_per_legend = 80  # pixels per legend column
            dynamic_height = base_height + (len(selected_legend_cols) * height_per_legend - 10)
            
            st.components.v1.html(raw_html, height=dynamic_height, scrolling=True)
        except Exception as e:
            st.error(f"Error displaying molecules: {str(e)}")
    
    def _calculate_admet_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADMET properties using Admetica CLI."""
        try:
            import subprocess
            import tempfile
            import os
            
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
                temp_input_path = temp_input.name
                df[[self.smiles_col]].to_csv(temp_input_path, index=False)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Progress indicator
                progress_bar = st.progress(0)
                st.info("Running Admetica predictions... This may take a few minutes.")
                
                # Run admetica_predict CLI command for absorption properties
                # Available absorption properties: Caco2, Lipophilicity, Solubility, Pgp-Inhibitor, Pgp-Substrate
                absorption_props = "Caco2,Solubility,Lipophilicity,Pgp-Inhibitor,Pgp-Substrate"
                
                cmd = [
                    'admetica_predict',
                    '--dataset-path', temp_input_path,
                    '--smiles-column', self.smiles_col,
                    '--properties', absorption_props,
                    '--save-path', temp_output_path
                ]
                
                progress_bar.progress(0.3)
                
                # Run the command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                progress_bar.progress(0.7)
                
                if result.returncode == 0:
                    # Read the results
                    result_df = pd.read_csv(temp_output_path)
                    
                    # Merge results with original dataframe
                    # The output will have predictions for the specified properties
                    for col in result_df.columns:
                        if col != self.smiles_col:
                            df[col] = result_df[col]
                    
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    st.success(f"✅ Calculated ADMET absorption properties for {len(df)} molecules")
                else:
                    progress_bar.empty()
                    st.error(f"❌ Admetica command failed: {result.stderr}")
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            st.error("❌ Admetica prediction timed out. Try with a smaller dataset.")
        except FileNotFoundError:
            st.error("❌ Admetica CLI not found. Please install it using: pip install admetica==1.4.1")
        except Exception as e:
            st.error(f"❌ Error calculating ADMET properties: {str(e)}")
        
        return df
    
    def _calculate_metabolism_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metabolism-related ADMET properties using Admetica CLI."""
        try:
            import subprocess
            import tempfile
            import os
            
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
                temp_input_path = temp_input.name
                df[[self.smiles_col]].to_csv(temp_input_path, index=False)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Progress indicator
                progress_bar = st.progress(0)
                st.info("Running Admetica metabolism predictions... This may take a few minutes.")
                
                # Run admetica_predict CLI command for metabolism properties
                # CYP enzymes: CYP1A2-Inhibitor, CYP1A2-Substrate, CYP2C19-Inhibitor, CYP2C19-Substrate,
                #              CYP2C9-Inhibitor, CYP2C9-Substrate, CYP2D6-Inhibitor, CYP2D6-Substrate,
                #              CYP3A4-Inhibitor, CYP3A4-Substrate
                metabolism_props = "CYP1A2-Inhibitor,CYP1A2-Substrate,CYP2C19-Inhibitor,CYP2C19-Substrate,CYP2C9-Inhibitor,CYP2C9-Substrate,CYP2D6-Inhibitor,CYP2D6-Substrate,CYP3A4-Inhibitor,CYP3A4-Substrate"
                
                cmd = [
                    'admetica_predict',
                    '--dataset-path', temp_input_path,
                    '--smiles-column', self.smiles_col,
                    '--properties', metabolism_props,
                    '--save-path', temp_output_path
                ]
                
                progress_bar.progress(0.3)
                
                # Run the command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                progress_bar.progress(0.7)
                
                if result.returncode == 0:
                    # Read the results
                    result_df = pd.read_csv(temp_output_path)
                    
                    # Merge results with original dataframe
                    # The output will have predictions for the specified properties
                    for col in result_df.columns:
                        if col != self.smiles_col:
                            df[col] = result_df[col]
                    
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    st.success(f"✅ Calculated ADMET metabolism properties for {len(df)} molecules")
                else:
                    progress_bar.empty()
                    st.error(f"❌ Admetica command failed: {result.stderr}")
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            st.error("❌ Admetica prediction timed out. Try with a smaller dataset.")
        except FileNotFoundError:
            st.error("❌ Admetica CLI not found. Please install it using: pip install admetica==1.4.1")
        except Exception as e:
            st.error(f"❌ Error calculating metabolism properties: {str(e)}")
        
        return df
    
    def _calculate_toxicity_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate toxicity-related ADMET properties using Admetica CLI."""
        try:
            import subprocess
            import tempfile
            import os
            
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
                temp_input_path = temp_input.name
                df[[self.smiles_col]].to_csv(temp_input_path, index=False)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
                temp_output_path = temp_output.name
            
            try:
                # Progress indicator
                progress_bar = st.progress(0)
                st.info("Running Admetica toxicity predictions... This may take a few minutes.")
                
                # Run admetica_predict CLI command for toxicity properties
                # Toxicity properties: hERG and LD50
                toxicity_props = "hERG,LD50"
                
                cmd = [
                    'admetica_predict',
                    '--dataset-path', temp_input_path,
                    '--smiles-column', self.smiles_col,
                    '--properties', toxicity_props,
                    '--save-path', temp_output_path
                ]
                
                progress_bar.progress(0.3)
                
                # Run the command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                progress_bar.progress(0.7)
                
                if result.returncode == 0:
                    # Read the results
                    result_df = pd.read_csv(temp_output_path)
                    
                    # Merge results with original dataframe
                    # The output will have predictions for the specified properties
                    for col in result_df.columns:
                        if col != self.smiles_col:
                            df[col] = result_df[col]
                    
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    st.success(f"✅ Calculated ADMET toxicity properties for {len(df)} molecules")
                else:
                    progress_bar.empty()
                    st.error(f"❌ Admetica command failed: {result.stderr}")
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            st.error("❌ Admetica prediction timed out. Try with a smaller dataset.")
        except FileNotFoundError:
            st.error("❌ Admetica CLI not found. Please install it using: pip install admetica==1.4.1")
        except Exception as e:
            st.error(f"❌ Error calculating toxicity properties: {str(e)}")
        
        return df
    
    def _descriptor_section(self):
        """Calculate molecular descriptors."""
        st.markdown("---")
        st.subheader("⚗️ Calculate Descriptors")
        st.info("Select one or more descriptor types and click 'Calculate'.")
        
        # First row: Basic descriptors and fingerprints
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            calc_basic = st.checkbox("Basic 2D Descriptors", help="Suffix _1 is added to avoid overwriting existing columns")
        with col2:
            calc_all = st.checkbox("All RDKit 2D Descriptors (200+)", help="Suffix _2 is added to avoid overwriting existing columns")
        with col3:
            calc_fp = st.checkbox("Morgan Fingerprints", help="Radius=2, Bits=2048")
        with col4:
            calc_admet = st.checkbox("ADMET: Absorption", help="Caco2, Solubility, Lipophilicity, Pgp-Inhibitor, Pgp-Substrate")
        
        # Second row: Metabolism and Toxicity ADMET properties
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            calc_metabolism = st.checkbox("ADMET: Metabolism", help="CYP enzymes (1A2, 2C9, 2C19, 2D6, 3A4) substrate and inhibitor predictions")
        with col6:
            calc_toxicity = st.checkbox("ADMET: Toxicity", help="hERG and LD50 predictions")
        
        if st.button("✅ Calculate", type="primary"):
            with st.spinner("Calculating descriptors..."):
                result_df = self._df.copy()
                
                if calc_basic:
                    for desc_name in self.config.BASIC_DESCRIPTORS:
                        result_df[f"{desc_name}_1"] = result_df[self.smiles_col].apply(
                            lambda x: MoleculeUtils.calculate_descriptors(x, [desc_name]).get(desc_name)
                        )
                    
                    # Add SA Score and QED
                    result_df['SAScore_1'] = result_df[self.smiles_col].apply(
                        lambda x: MoleculeUtils.calculate_sa_score(x)
                    )
                    result_df['QED_1'] = result_df[self.smiles_col].apply(
                        lambda x: MoleculeUtils.calculate_qed(x)
                    )
                
                if calc_all:
                    st.info("Computing all 200+ RDKit descriptors... This may take a moment.")
                    result_df = MoleculeUtils.calculate_all_descriptors(result_df, self.smiles_col)
                
                if calc_fp:
                    result_df['MorganFP'] = result_df[self.smiles_col].apply(
                        lambda x: MoleculeUtils.generate_morgan_fp(
                            x, self.config.FP_RADIUS, self.config.FP_BITS
                        )
                    )
                
                if calc_admet:
                    st.info("Computing ADMET properties (Absorption)... This may take a moment.")
                    result_df = self._calculate_admet_properties(result_df)
                
                if calc_metabolism:
                    st.info("Computing ADMET properties (Metabolism)... This may take a moment.")
                    result_df = self._calculate_metabolism_properties(result_df)
                
                if calc_toxicity:
                    st.info("Computing ADMET properties (Toxicity)... This may take a moment.")
                    result_df = self._calculate_toxicity_properties(result_df)
                
                st.success(f"✅ Calculated! Shape: {result_df.shape}")
                st.dataframe(result_df, width='stretch')
                
                # Download button
                sdf_buffer = StringIO()
                sdf_copy = result_df.copy()
                #create a mol column from smiles if not already present
                if 'mol' not in sdf_copy.columns:
                    sdf_copy.loc[:, 'mol'] = sdf_copy[self.smiles_col].apply(
                        lambda x: MoleculeUtils.smiles_to_mol(x)
                    )
                PandasTools.WriteSDF(sdf_copy, sdf_buffer, molColName='mol', properties=list(sdf_copy.columns), idName=None)
                sdf_data = sdf_buffer.getvalue()
                sdf_buffer.close()
                
                #st.info("Click the button below to download the SDF file. If descriptors were calculated, they will be included as properties in the SDF file. Otherwise, only the orginal columns will be included.")
                st.download_button(
                    label="⬇️ Download SDF",
                    data=sdf_data,
                    file_name="molecules.sdf",
                    mime="chemical/x-mdl-sdfile",
                    type="primary"
                )
