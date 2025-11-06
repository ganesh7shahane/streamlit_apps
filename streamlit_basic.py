from pyparsing import itemgetter
import rdkit
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw, rdRGroupDecomposition, Descriptors, FilterCatalog
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from rdkit.Chem import rdMMPA
import useful_rdkit_utils as uru
from tqdm.auto import tqdm
from itertools import chain
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from rdkit.Chem import AllChem
from scaffold_finder import FragmentMol
from rdkit.Chem.rdDepictor import Compute2DCoords
import mols2grid
import io
from io import StringIO
from IPython.display import HTML
import base64
import requests
from rdkit import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from streamlit_ketcher import st_ketcher

#adjust the width of the page
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:55rem;
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        [data-testid="stSidebar"] .stMarkdown {
            padding: 0.5rem 0;
        }
    </style>
    """
)

############################################################################
#
# Define Sidebar with Chemical Sketcher
#
############################################################################

with st.sidebar:
    st.title("🧪 Chemical Sketcher")
    st.markdown("Draw a molecule using Ketcher below:")
    
    # Launch the Ketcher chemical sketcher
    molecule_smiles = st_ketcher()
    
    # Display the SMILES string if a molecule is drawn
    if molecule_smiles:
        st.subheader("Generated SMILES:")
        st.code(molecule_smiles, language="text")
        
        # Option to copy SMILES
        if st.button("📋 Copy SMILES to Clipboard"):
            st.write("SMILES copied!")
            st.toast("SMILES copied to clipboard!", icon="✅")
        
        # Add button to analyze in SMILES Analysis page
        st.info("💡 Go to 'SMILES Analysis' page to see detailed analysis of this molecule!")
        
        # Display molecular properties if SMILES is valid
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol:
                st.subheader("Quick Properties:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MW", f"{Descriptors.MolWt(mol):.2f}")
                    st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                with col2:
                    st.metric("HBD", Descriptors.NumHDonors(mol))
                    st.metric("HBA", Descriptors.NumHAcceptors(mol))
        except:
            st.warning("Invalid SMILES structure")
    else:
        st.info("Draw a molecule above to see its SMILES and properties")
    
    st.markdown("---")
    st.markdown("### 📚 Quick Links")
    st.markdown("- [Ketcher Documentation](https://github.com/epam/ketcher)")
    st.markdown("- [RDKit Blog](https://greglandrum.github.io/rdkit-blog/)")
    st.markdown("- My Github: [ganesh7shahane](https://github.com/ganesh7shahane)")

# Store molecule_smiles in session state for use across pages
if molecule_smiles:
    st.session_state['sketched_smiles'] = molecule_smiles
    st.session_state['sketched_mol'] = Chem.MolFromSmiles(molecule_smiles) if Chem.MolFromSmiles(molecule_smiles) else None
else:
    st.session_state['sketched_smiles'] = None
    st.session_state['sketched_mol'] = None

############################################################################
#
# Define SideBar Menu for different pages
#
############################################################################



############################################################################
#
#   Define option menu
#
############################################################################

selected = option_menu(
    menu_title="Structure-Activity Relationship (SAR) Analysis Tools",
    menu_icon="bar-chart-fill",
    options=["DataFrame Viz", "Analyse Scaffolds", "SMILES Analysis", "Taylor-Butina Clustering"],
    icons=["0-square", "1-square", "2-square", "3-square"],
    default_index=0,
    orientation="horizontal"
)

if selected == "DataFrame Viz":
    st.title("⚛️ Molecule DataFrame Visualisation")

    st.markdown("Sometimes we have a CSV file containing SMILES and their properties, and we just wish to visualise it and analyse some of its properties. This web app is precisely for that.")
    st.markdown("Specifically, this page let's you:")
    st.markdown("- :red[View & Clean] the CSV file as a Pandas DataFrame")
    st.markdown("- :red[Compute] stats on the numerical columns: NaN count, mean, count, std, min, max etc.")
    st.markdown("- :red[Plot] histograms, barplots, correlation heatmaps of the numerical columns")
    st.markdown("- :red[Visualise] the 2D structures of the molecules")
    st.markdown("- :red[Calculate] RDKit descriptors: 2D Phys-chem descriptors, Morgan fingerprints or both")
    st.subheader("Upload the CSV file")

    # Load default CSV file (local or URL)
    default_file_path = 'https://raw.githubusercontent.com/ganesh7shahane/useful_cheminformatics/refs/heads/main/data/FINE_TUNING_pi3k-mtor_objectives.csv'  # adjust path or use URL
    default_df = pd.read_csv(default_file_path, index_col=False)
    uploaded_file = st.file_uploader("", type=["csv"])

    ############################################################################
    #
    #   Do all analysis on the CSV
    #
    ############################################################################

    if uploaded_file is not None:
        # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file, index_col=False)
        st.success(f"Analyzing uploaded file: {uploaded_file.name}")
    else:
        df = default_df
        st.info("Using default CSV file for sample SAR analysis")
        
    # Display the top 5 rows
    st.subheader("Let's see how the dataset looks like")
    rows = st.slider("Choose rows to display", 3, len(df))
    col1,col2 = st.columns(2, gap="large")
    with col1:
        #add a checkbox to remove duplcate molecules based on inchi computed from SMILES
        remove_invalid_smiles = st.checkbox("Remove invalid SMILES that RDKit cannot read", value=True)
        if remove_invalid_smiles:
            # Find the SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == "smiles":
                    smiles_col = col
                    break
            if smiles_col:
                valid_smiles = []
                invalid_smiles = []
                for smi in df[smiles_col]:
                    try:
                        Chem.CanonSmiles(smi)
                        valid_smiles.append(smi)
                    except:
                        invalid_smiles.append(smi)
                before_count = df.shape[0]
                df = df[df[smiles_col].isin(valid_smiles)].reset_index(drop=True)
                after_count = df.shape[0]
                st.success(f"Removed {before_count - after_count} invalid SMILES.")
            else:
                st.error("No SMILES column found (case-insensitive). Cannot remove invalid SMILES.")
    with col2:
        remove_duplicates = st.checkbox("Remove duplicate molecules based on InChI", value=False)
        if remove_duplicates:
            # Find the SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == "smiles":
                    smiles_col = col
                    break
            if smiles_col:
                df.loc[:, 'InChI_'] = df[smiles_col].apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if x else None)
                before_count = df.shape[0]
                df = df.drop_duplicates(subset=['InChI_']).reset_index(drop=True)
                after_count = df.shape[0]
                st.success(f"Removed {before_count - after_count} duplicate molecules based on InChI.")
            else:
                st.error("No SMILES column found (case-insensitive). Cannot remove duplicates.")
    st.dataframe(df.head(rows))
    
    # Show the total number of rows
    st.info(f"It appears there are {df.shape[0]} molecules in the dataset.")
    
    # Find the SMILES column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break
    if smiles_col:
        st.success(f"Identified SMILES column: {smiles_col}")
    else:
        st.error("No SMILES column found (case-insensitive). Some functionalities will be limited.")
        
    ############################################################################
    #
    #   Statistics and Histograms
    #
    ############################################################################
    
    st.subheader("Check for NaN in all columns")
    #check Nan in all columns and rename the count column to 'NaN count'
    nan_counts = df.isna().sum().reset_index()
    nan_counts.columns = ['Column', 'NaN Count']
    st.dataframe(nan_counts)
    #st.write(df.isna().sum())
            
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_cols = []
            
    if len(numeric_cols) > 0:
        if len(numeric_cols) > 1:
            st.subheader("Compute some column statistics")
            st.write("**count, min, max, etc:**")
            st.dataframe(df[numeric_cols].describe())

            st.write("\n")
            st.subheader("Histograms of Selected Columns")
            
            #Allow user to select columns through expander that lists all columns with checkboxes
            with st.expander("Show and select columns"):
                for col in numeric_cols:
                    st.checkbox(col, value=True, key=col)

            #Get the selected columns
            selected_cols = [col for col in numeric_cols if st.session_state[col]]
            
            #The dataframe will be filtered to only selected columns
            df_selected = df[selected_cols]
                                            
            #Give user options to change the default values
            st.info("You can change the default values of number of bins, color and figure size below.") 
            
            #shift the slider to the left column
            col1, col2, col3 = st.columns(3, gap="large")
            with col1:
                default_bins = st.slider("Number of bins", min_value=2, max_value=100, value=30, step=5)
            with col2:
                default_color = st.color_picker("Pick a color for the histogram", value='#EE82EE')
            with col3:
                default_size = st.slider("Figure size (width, height)", min_value=5, max_value=25, value=(14, 15), step=1)

            fig = df_selected.hist(
                bins=default_bins,
                color=default_color,
                edgecolor='black',
                layout=(len(numeric_cols) // 2 + 1, 3),
                figsize=default_size,
                grid=True
            )

            plt.tight_layout()      # Prevents overlapping text
            plt.gcf().set_dpi(300) # Set DPI for high resolution
            with st.expander("Show/Hide Histograms", expanded=True):
                st.pyplot(plt.gcf())  

        #Put up a divider
        st.markdown("---")
        
        st.subheader("**Draw a single column histogram or bar plot**")
        #select whether to draw a histogram or barplot, put these side-by-side
        col1, col2 = st.columns(2)
        with col1:
            plot_type = st.selectbox("Select plot type", options=["Histogram", "Bar Plot"])
        with col2:
            selected_column = st.selectbox("Select a column", options=numeric_cols)

        # Only one column should be selected; if more are selected, take the last
        if plot_type == "Histogram" and selected_column:
            st.write(f"{plot_type} for: {selected_column}")

            # Prepare histogram
            col1_1, col2_1, col3_1 = st.columns(3, gap="large")
            with col1_1:
                default_bins_1 = st.slider("Bins", min_value=2, max_value=100, value=15, step=2)
            with col2_1:
                default_color_1 = st.color_picker("Pick a color", value="#82EEDE")
            with col3_1:
                default_size_1 = st.slider("Figure size - (width, height)", min_value=5, max_value=25, value=(6, 7), step=1)

            fig, ax = plt.subplots(figsize=default_size_1, dpi=200)  # High resolution
            df[selected_column].hist(ax=ax, bins=default_bins_1, color=default_color_1, edgecolor="black")
            ax.set_title(f"Histogram of {selected_column}")
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Frequency")
            with st.expander("Show/Hide Histogram", expanded=True):
                st.pyplot(fig)

        elif plot_type == "Bar Plot" and selected_column:
            st.write(f"{plot_type} for: {selected_column}")

            # Let user select which column to plot
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Automatically suggest bin width
            col_min, col_max = df[selected_column].min(), df[selected_column].max()
            default_bin_width = (col_max - col_min) / 10  # default ~10 bins
            

            col1_2, col2_2, col3_2 = st.columns(3)
            with col1_2:
                st.write(f"**Value range:** {col_min:.2f} to {col_max:.2f}")
            with col2_2:
                bin_mode = st.radio("How do you want to define bins?", ["Auto", "Custom"], horizontal=True)
            with col3_2:
                default_color_2 = st.color_picker("Pick a color", value="#125AF7")

            if bin_mode == "Custom":
                bin_width = st.number_input("Enter bin width:", min_value=0.1, value=float(default_bin_width))
                
            else:
                bin_width = default_bin_width
            bins = np.arange(col_min, col_max + bin_width, bin_width)
            df.loc[:, "binned"] = pd.cut(df[selected_column], bins=bins, include_lowest=True)

            # Count frequency per bin
            bin_counts = df["binned"].value_counts().sort_index()

            # Plot bar chart
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            ax.bar(bin_counts.index.astype(str), bin_counts.values, color=default_color_2, edgecolor="black")
            #plot count on top of each bar
            for i, v in enumerate(bin_counts.values):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {selected_column}")
            plt.xticks(rotation=45, ha="right")
            with st.expander("Show/Hide Bar Plot", expanded=True):
                st.pyplot(fig)

        ##put up a divider
        st.markdown("---")
        
        #Put the correlation heatmap under an expander
        
        st.subheader("Correlation Heatmap")
        st.markdown("A correlation heatmap helps to visualize the correlation coefficients between multiple numerical columns in a dataset. It provides insights into how different variables relate to each other, which can be useful for feature selection and understanding relationships in the data.")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            with st.expander("Show/Hide Correlation Heatmap", expanded=True):
                st.pyplot(fig)
        else:
            st.error("Not enough numerical columns to compute correlation heatmap.")
            
        ##put up a divider
        st.markdown("---")
        
        st.subheader("Interactive Regression Plot")
        if len(numeric_cols) >= 2:
            #set col1 and col2 side-by-side
            col1, col2, col3 = st.columns(3)
            with col1:
                col1 = st.selectbox("Select X-axis column", options=numeric_cols, index=0)
            with col2:
                col2 = st.selectbox("Select Y-axis column", options=numeric_cols, index=1)
            if col1 and col2:                    
                #Use plotly to make an interactive regression plot. When hovering over a point, show the 2D structure and index of the molecule
                #colour slider
                with col3:
                    color_col = st.selectbox("Select color column", options=numeric_cols, index=2)
                fig = px.scatter(df, x=col1, y=col2, hover_data=[df.index,smiles_col], size_max=50, color=color_col)
                #add regression line and correlation coefficient to the title
                corr_coef = df[[col1, col2]].corr().iloc[0, 1]
                fig.add_traces(px.scatter(df, x=col1, y=col2, trendline="ols").data[1])
                fig.update_layout(title=f"{col1} vs {col2} (Correlation Coefficient: {corr_coef:.2f})", title_font_size=20)
                st.plotly_chart(fig)
        else:
            st.error("Not enough numerical columns to plot regression.")
            
        if len(numeric_cols)>=3:
            st.subheader("Interactive 3D Scatter Plot")
            #put the three selectboxes side-by-side
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                col_x = st.selectbox("Select X-axis column", options=numeric_cols, index=0, key="x_axis")
            with col_y:
                col_y = st.selectbox("Select Y-axis column", options=numeric_cols, index=1, key="y_axis")
            with col_z:
                col_z = st.selectbox("Select Z-axis column", options=numeric_cols, index=2, key="z_axis")
            if col_x and col_y and col_z:
                fig_3d = px.scatter_3d(df, x=col_x, y=col_y, z=col_z, hover_data=[df.index,smiles_col], size_max=50)
                fig_3d.update_layout(title=f"3D Scatter Plot: {col_x} vs {col_y} vs {col_z}", title_font_size=20)
                st.plotly_chart(fig_3d)
        else:
            st.error("Not enough numerical columns to plot 3D scatter plot.")

    ############################################################################
    #
    #   2D Visualisation
    #
    ############################################################################
    
    # Double-ended sliders for each numerical column
    st.subheader("Visualise 2D structures of molecules")
    st.write(f"Currently, there are {len(df)} molecules in the dataset.")
    
    @st.cache_data
    def smi_to_png(smi: str) -> str:
        """Returns molecular image as data URI."""
        mol = rdkit.Chem.MolFromSmiles(smi)
        pil_image = rdkit.Chem.Draw.MolToImage(mol)

        with io.BytesIO() as buffer:
            pil_image.save(buffer, "png")
            data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{data}"

    df_smiles = df[smiles_col]
    ok_smiles = []
    delete_smiles = []
    i=0
    
    ####################################################################################
    #
    #   Check if all SMILES can be read by RDKit
    ####################################################################################

    st.write("Checking if all SMILES strings can be read by RDKit...")
    for ds in df_smiles:
        try:
            cs = Chem.CanonSmiles(ds)
            ok_smiles.append(cs)
            i=i+1
        except:
            print('Invalid SMILES:', ds)
            delete_smiles.append(ds)
    
    if len(delete_smiles) == 0:
        st.success("All SMILES strings are valid and can be read by RDKit.")
    else:   
        st.error(f"There are {len(delete_smiles)} invalid SMILES strings.")
        st.write(f"These invalid SMILES strings are:")
        st.write(delete_smiles)
        st.write("Removing these invalid SMILES strings from the dataset and proceeding...")
        df = df[~df[smiles_col].isin(delete_smiles)].reset_index(drop=True)
    
    ####################################################################################
    #
    #   Filter the dataframe based on the sliders
    #
    ####################################################################################
    if len(numeric_cols) > 0:
        st.write("To visualise 2D structures of all molecules, leave the sliders at their default values.")
        filters = {}
        # Create two columns for layout
        col1, col2 = st.columns(2, gap="large")
        #remove columns from numeric_cols that have constant values
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 1]
        for idx, col in enumerate(numeric_cols):
            with col1 if idx % 2 == 0 else col2:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                filters[col] = st.slider(
                f"{col} range",
                min_value=float(min_val),
                max_value=float(max_val),
                value=(float(min_val), float(max_val))
        )

        filtered_df = df.copy()
        for col, (min_val, max_val) in filters.items():
            new_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

        st.write(f"Remaining molecules: {len(new_df)}")

        # Generate 2D structures and property legends
        st.subheader("2D Structures of Filtered Molecules")
        mols = []
        legends = []
        new_df.loc[:, 'mol_2'] = new_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
        
        #Add options to either display top 10 or all molecules
        display_option = st.radio("Display options:", ("All filtered molecules", "Display all molecules in the dataset (no filtering)"), index=0)
        
        #choose legend from dropdown, list of columns, not mol column
        legend_option = st.selectbox("Choose legend to display:", options=[col for col in new_df.columns if col != 'mol_2'], index=0)

        if display_option == "All filtered molecules":
            display_count = len(new_df)
            html_data = mols2grid.display(new_df.head(display_count), mol_col='mol_2', size=(200, 200), 
                                        subset=["img",legend_option],
                                        n_items_per_page=16, 
                                        fixedBondLength=25, clearBackground=False,
                                        ).data
            st.components.v1.html(html_data, height=1100, scrolling=True)
        
        elif display_option == "Display all molecules in the dataset (no filtering)":
            no_restricted_df = df.copy()
            display_count = len(df)
            no_restricted_df.loc[:, 'mol_2'] = no_restricted_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
            #AllChem.Compute2DCoords(no_restricted_df['mol_2'][0])
            #[AllChem.GenerateDepictionMatching2DStructure(m,no_restricted_df['mol_2'][0]) for m in no_restricted_df['mol_2']]
            html_data = mols2grid.display(no_restricted_df.head(display_count), mol_col='mol_2', size=(200, 200), subset=["img",legend_option],n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
            st.components.v1.html(html_data, height=1100, scrolling=True)
    
    elif len(numeric_cols) == 0:
        st.info("No numerical columns found in the dataset. Cannot apply filters.")
        if smiles_col:
            st.write("Displaying 2D structures of all molecules in the dataset.")
            df.loc[:, 'mol_2'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
            display_count = len(df)
            html_data = mols2grid.display(df.head(display_count), mol_col='mol_2', size=(200, 200), subset=["img",legend_option], n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
            st.components.v1.html(html_data, height=1100, scrolling=True)
        else:
            st.error("No SMILES column found (case-insensitive). Cannot visualise 2D structures.")
    
    ############################################################################
    #
    #   Calculate descriptors
    #
    ############################################################################
    
    st.subheader("Calculate descriptors")
    st.markdown("Here, you can prepare your dataset for training machine learning models or some other analysis.") 
    st.info("Select one or more descriptors and click 'Calculate'.")
    # Define descriptor options
    basic_descs = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']

    desc_basic = st.checkbox("Basic 2D descriptors")
    desc_all = st.checkbox("All RDKit 2D descriptors (200+)")
    desc_fp = st.checkbox("Morgan fingerprints (radius=2, nBits=2048)")
    calculate = st.button("✅ Calculate", type="primary")

    if calculate:
        # Prepare molecule objects
        mols = [Chem.MolFromSmiles(s) for s in df[smiles_col]]
        result = pd.DataFrame({'SMILES': df[smiles_col]})
        
        if desc_basic:
            for d in basic_descs:
                func = getattr(Descriptors, d)
                result[d] = [func(m) if m else None for m in mols]
            #add rotatable bonds, TPSA and SA score
            result['NumRotatableBonds_1'] = [Descriptors.NumRotatableBonds(m) if m else None for m in mols]
            result['TPSA_1'] = [Descriptors.TPSA(m) if m else None for m in mols]
            result['SAScore_1'] = [sascorer.calculateScore(m) if m else None for m in mols]
            result['QED_1'] = [Chem.QED.qed(m) if m else None for m in mols]
            #add suffix _1 to all the columns
            result.rename(columns={d: d + '_1' for d in basic_descs}, inplace=True)
        
        if desc_all:
            for dname, func in Descriptors._descList:
                result[dname] = [func(m) if m else None for m in mols]
            #add suffix _2 to the new columns
            result.rename(columns={dname: dname + '_2' for dname, func in Descriptors._descList}, inplace=True)
        
        if desc_fp:
            nBits = 2048
            result['MorganFP'] = [
                AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits).ToBitString() if mol else None
                for mol in mols
            ]
        
        st.subheader("Calculated Results")
        #Concatenate the original dataframe with the result dataframe
        final_df = pd.concat([df, result.drop(columns=['SMILES'])], axis=1)
        st.write(f"The resulting dataframe has {final_df.shape[0]} rows and {final_df.shape[1]} columns.")
        st.dataframe(final_df, use_container_width=True)
        df_to_download = final_df.copy()
    else:
        df_to_download = df.copy()
        
    ###########################################################################################
    #
    #   Download the dataframe as SDF
    #
    ###########################################################################################
    st.subheader("Download the DataFrame as SDF")
    #st.markdown("You can download the DataFrame as an SDF file, which is a common format for storing molecular structures along with their associated data.")
    
    #check if there is a mol column in the dataframe
    if 'mol' not in df_to_download.columns:
        if smiles_col:
            df_to_download.loc[:, 'mol'] = df_to_download[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
        else:
            st.error("No SMILES column found (case-insensitive). Cannot generate SDF file.")
    # Use PandasTools to convert DataFrame to SDF
    from rdkit.Chem import PandasTools
    
    #Create a button to download the SDF file
    sdf_buffer = StringIO()
    PandasTools.WriteSDF(df_to_download, sdf_buffer, molColName='mol', properties=list(df_to_download.columns), idName=None)
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

##########################################################################
#   Scaffold Identification
###########################################################################

if selected == "Analyse Scaffolds":
    st.title("📎 Identifying Scaffolds from a chemical series")    
    
    st.markdown("Identifying Bemis Murcko (BM) scaffolds—the core structural frameworks that recur among active compounds in SAR datasets—is a critical step because it helps link structure to biological function.")

    st.markdown("- BM scaffolds identify the chemical “core” responsible for a compound’s biological activity, enabling chemists to understand which parts of the structure can be modified without losing potency.")

    st.markdown("- Once scaffolds associated with high activity are known, chemists can perform scaffold decoration and substituent optimization more effectively to improve pharmacological properties such as potency, selectivity, and solubility.")

    st.markdown("- Recognizing key scaffolds allows researchers to explore novel chemotypes via scaffold hopping—designing structurally distinct molecules that retain activity—thus improving IP diversity and discovering new drugs for existing targets.") 

    from rdkit.Chem.Scaffolds import MurckoScaffold
    import useful_rdkit_utils as uru
    
    lib_file = requests.get("https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/main/sar_analysis/scaffold_finder.py")
    ofs = open("scaffold_finder.py","w")
    # Fix the UndefinedVariableError in find_scaffolds
    fixed_code = lib_file.text.replace("len(df_in)\n    scaffold_df =", "num_df_rows = len(df_in)\n    scaffold_df =")
    print(lib_file.text,file=ofs)
    ofs.close()
    from scaffold_finder import generate_fragments, find_scaffolds, get_molecules_with_scaffold, cleanup_fragment

    st.subheader("Upload the CSV file")
    default_file_path = 'https://raw.githubusercontent.com/ganesh7shahane/streamlit_apps/refs/heads/main/data/chembl208.csv'  # adjust path or use URL
    default_df = pd.read_csv(default_file_path, index_col=False)
    uploaded_file = st.file_uploader("", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file, index_col=False)
        st.success(f"Analyzing uploaded file: {uploaded_file.name}")
    else:
        df = default_df
        st.info("Using default CSV file for sample SAR analysis")
        
    # Display the top 5 rows
    st.subheader("Let's see how the dataset looks like")
    
    rows = st.slider("Choose rows to display", 3, len(df))
    
    col1,col2 = st.columns(2, gap="large")
    with col1:
        #add a checkbox to remove duplcate molecules based on inchi computed from SMILES
        remove_invalid_smiles = st.checkbox("Remove invalid SMILES that RDKit cannot read", value=True)
        if remove_invalid_smiles:
            # Find the SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == "smiles":
                    smiles_col = col
                    break
            if smiles_col:
                valid_smiles = []
                invalid_smiles = []
                for smi in df[smiles_col]:
                    try:
                        Chem.CanonSmiles(smi)
                        valid_smiles.append(smi)
                    except:
                        invalid_smiles.append(smi)
                before_count = df.shape[0]
                df = df[df[smiles_col].isin(valid_smiles)].reset_index(drop=True)
                after_count = df.shape[0]
                st.success(f"Removed {before_count - after_count} invalid SMILES.")
            else:
                st.error("No SMILES column found (case-insensitive). Cannot remove invalid SMILES.")
    with col2:  
        remove_duplicates = st.checkbox("Remove duplicate molecules based on InChI", value=False)
        if remove_duplicates:
            # Find the SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == "smiles":
                    smiles_col = col
                    break
            if smiles_col:
                df.loc[:, 'InChI_'] = df[smiles_col].apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if x else None)
                before_count = df.shape[0]
                df = df.drop_duplicates(subset=['InChI_']).reset_index(drop=True)
                after_count = df.shape[0]
                st.success(f"Removed {before_count - after_count} duplicate molecules based on InChI.")
            else:
                st.error("No SMILES column found (case-insensitive). Cannot remove duplicates.")
    st.dataframe(df.head(rows))
    
    # Show the total number of rows
    st.info(f"It appears there are {df.shape[0]} molecules in the dataset.")
    
    # Find the SMILES column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break
    if smiles_col:
        st.success(f"Identified SMILES column: {smiles_col}")
    else:
        st.error("No SMILES column found (case-insensitive). Some functionalities will be limited.")
    
    #identify scaffolds
    st.subheader("Identify Scaffolds")
    st.markdown("This might take a few seconds...")
    
    #identify the smiles column in the uploaded dataframe, if any of the columns has the regex 'smiles' in any pattern
    if smiles_col:
        st.write(f"Identified SMILES column: {smiles_col}")
        
    #Calculate Murcko_Scaffold for each molecule
    df.loc[:, 'Murcko_Scaffold'] = df[smiles_col].apply(lambda x: Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(x) if x else None)
    
    # Now create a new dataframe with the unique Bemis-Murcko_Scaffolds and the number of molecules with the 
    # scaffold in the initial dataset.
    scaffold_df = uru.value_counts_df(df,"Murcko_Scaffold")
    
    # Visualise all scaffolds and their counts
    st.write(f"There are {scaffold_df.shape[0]} unique scaffolds in the dataset and following are their counts")
    st.dataframe(scaffold_df)
    
    #Plot a barplot of the counts of the scaffolds, with X-axis as index as the scaffold ID and count as the y-axis
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    #only horizontal and not vertical grid lines
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    
    scaffold_df['count'][:20].plot(kind='bar', ax=ax, color="skyblue", edgecolor="black")
    ax.set_title(f"Barplot of top 20 Scaffold Counts")
    ax.set_xlabel("Scaffold ID")
    ax.set_ylabel("Number of molecules with the scaffold")
    st.pyplot(fig)

    st.subheader(f"Let's visualise all the scaffolds")
    #Visualise the scaffolds
    mols = []
    legends = []
    
    temp_scaffold_df = scaffold_df.copy()
    temp_scaffold_df.loc[:, 'mol_2'] = temp_scaffold_df['Murcko_Scaffold'].apply(lambda x: Chem.MolFromSmiles(x))
    html_data = mols2grid.display(temp_scaffold_df, mol_col='mol_2', size=(200, 200), n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
    st.components.v1.html(html_data, height=1000, scrolling=True)
        
    ###############################################################################
    #
    #   Relate activity distribution with scaffolds
    #
    ###############################################################################
    st.subheader("Let's relate activity distribution with scaffolds")

    #first, we need a mol column in scaffold_df
    scaffold_df.loc[:, 'mol'] = scaffold_df['Murcko_Scaffold'].apply(lambda x: Chem.MolFromSmiles(x))

    #Now we define the functions to plot boxplots and convert mol to base64 image
    def boxplot_base64_image(dist: np.ndarray, x_lim: list[int] = None) -> str:
        """
        Plot a distribution as a seaborn boxplot and save the resulting image as a base64 image.

        Parameters:
        dist (np.ndarray): The distribution data to plot.
        x_lim (list[int]): The x-axis limits for the boxplot.

        Returns:
        str: The base64 encoded image string.
        """
        if x_lim is None:
            x_lim = [0, 10]
        plt.figure(dpi=150)  # Set the figure size and resolution
        sns.set(rc={'figure.figsize': (4, 1)})
        sns.set_style('whitegrid')
        ax = sns.boxplot(x=dist)
        #set x_lim as per the input
        ax.set_xlim(x_lim[0], x_lim[1])
        s = io.BytesIO()
        plt.savefig(s, format='png', bbox_inches="tight")
        plt.close()
        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        return '<img align="left" src="data:image/png;base64,%s">' % s

    def mol_to_base64_image(mol: Chem.Mol) -> str:
        """
        Convert an RDKit molecule to a base64 encoded image string.

        Parameters:
        mol (Chem.Mol): The RDKit molecule to convert.

        Returns:
        str: The base64 encoded image string.
        """
        plt.figure(dpi=400)
        drawer = rdMolDraw2D.MolDraw2DCairo(350, 150)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        text = drawer.GetDrawingText()
        im_text64 = base64.b64encode(text).decode('utf8')
        img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
        return img_str
    
    #drop down menu to select the activity column
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    activity_col = st.selectbox(":green[Select] the activity column from the dataframe.",options=numeric_cols, index=0)
    #put the selected column in a variable
    st.write(f"Selected activity column: {activity_col}")
    
    #if activity_col contains any NaN values, delete the molecules and display an error message
    if df[activity_col].isna().sum() > 0:
        st.error(f"The selected activity column {activity_col} contains {df[activity_col].isna().sum()} NaN values. Removing such molecules and proceeding...")
        df = df.dropna(subset=[activity_col])

    if activity_col is not None:
        rows_to_display = scaffold_df.shape[0]
        tmp_df = scaffold_df.head(rows_to_display).copy()
        tmp_df.loc[:, 'mol_img'] = tmp_df.mol.apply(mol_to_base64_image)
        img_list = []
        for smi in tmp_df['Murcko_Scaffold'].values:
            img_list.append(boxplot_base64_image(df.query("Murcko_Scaffold == @smi")[activity_col].values, x_lim=[df[activity_col].min()*0.98, df[activity_col].max()*1.01]))
        tmp_df.loc[:, 'dist_img'] = img_list
        #display the dataframe with the images
        with st.expander("Show/Hide Scaffold Activity Distribution", expanded=True):
            st.markdown(HTML(tmp_df[['mol_img','count','dist_img']].to_html(escape=False)).data, unsafe_allow_html=True)
    else:
        st.error("No activity column selected.")
    ###############################################################################
    #
    #   Examine molecules with a given scaffold
    #
    ###############################################################################
    st.subheader("Examine molecules with a given scaffold")
    scaffold_id = st.selectbox("Select a scaffold index", options=scaffold_df.index.tolist())
    scaffold_smi = scaffold_df.Murcko_Scaffold.values[scaffold_id]
    tmp_df = df.query("Murcko_Scaffold == @scaffold_smi").copy() #put molecules with the scaffold in a new dataframe
    tmp_df.loc[:, 'mol'] = tmp_df[smiles_col].apply(Chem.MolFromSmiles)
    scaffold_mol = scaffold_df.mol.values[scaffold_id] #get mol object for the scaffold into scaffold_mol
    AllChem.Compute2DCoords(scaffold_mol)
    #[AllChem.GenerateDepictionMatching2DStructure(m,scaffold_mol) for m in tmp_df.mol]
    #st.write("hello")

    #concatenate legends
    legend_ = st.selectbox("Select legend", options=[col for col in tmp_df.columns if col != 'mol'], index=0)
    tmp_df.loc[:, 'legend'] = tmp_df[legend_].astype(str)
    #sort tmp_df as per descending order
    tmp_df_sort = tmp_df.sort_values(activity_col, ascending=False)
    #Draw molecules sorted as per ROCK2_log

    st.write(f"There are {len(tmp_df_sort)} molecules with scaffold ID {scaffold_id}")
    html_data = mols2grid.display(tmp_df_sort, mol_col='mol', size=(200, 200), subset=["img","legend"], n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
    st.components.v1.html(html_data, height=1100, scrolling=True)

############################################################################################
#   SMILES Analysis
############################################################################################
if selected == "SMILES Analysis":
    st.title("🔬 SMILES Analysis")
    st.write("This page lets you input a SMILES string, visualise the 2D structure, compute some common 2D phys-chem descriptors, and alerts.")
    
    # Check if user has drawn a molecule in sidebar
    if st.session_state.get('sketched_smiles'):
        st.success(f"✅ Molecule detected from sidebar sketcher!")
        use_sidebar_mol = st.checkbox("Use molecule from sidebar sketcher", value=True)
        if use_sidebar_mol:
            smiles_input = st.session_state['sketched_smiles']
            st.info(f"Using SMILES: `{smiles_input}`")
        else:
            st.subheader("Input the SMILES string")
            smiles_input = st.text_input("Enter the SMILES string here", value="C=CC(=O)N1CCN([C@H](C1)C)c1nc(OC[C@@H]2CCCN2C)nc2c1cc(Cl)c(c2F)c1nc(N)cc(c1C(F)(F)F)C")
    else:
        st.subheader("Input the SMILES string")
        st.info("💡 Tip: You can also draw a molecule using the Ketcher sketcher in the sidebar!")
        smiles_input = st.text_input("Enter the SMILES string here", value="C=CC(=O)N1CCN([C@H](C1)C)c1nc(OC[C@@H]2CCCN2C)nc2c1cc(Cl)c(c2F)c1nc(N)cc(c1C(F)(F)F)C")
    
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            #Display image and table side-by-side
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.subheader("2D Structure")
                #make the image sharper
                img = Draw.MolToImage(mol, size=(850, 700), kekulize=True)
                st.image(img, caption="2D structure of the input SMILES", use_container_width=False)
            with col2:
                st.subheader("Properties")
                #Compute some common 2D phys-chem descriptors
                descriptors = {
                    "Molecular Weight": Descriptors.MolWt,
                    "cLogP": Descriptors.MolLogP,
                    "Number of H-bond Donors": Descriptors.NumHDonors,
                    "Number of H-bond Acceptors": Descriptors.NumHAcceptors,
                    "Topological Polar Surface Area (TPSA)": Descriptors.TPSA,
                    "QED": Chem.QED.qed,
                    "Number of Rotatable Bonds": Descriptors.NumRotatableBonds,
                    "Number of Aromatic Rings": Descriptors.NumAromaticRings,
                    "Number of Aliphatic Rings": Descriptors.NumAliphaticRings,
                    "Number of Rings": Descriptors.RingCount,
                    "Fraction of Csp3 Carbons": Descriptors.FractionCSP3,
                    "SA Score": sascorer.calculateScore
                }
                # Create a DataFrame to display the descriptors
                desc_df = pd.DataFrame({desc_name: func(mol) for desc_name, func in descriptors.items()}, index=[0])
                #rename column to Descriptor and Value
                desc_df_T=desc_df.T
                desc_df_T.columns = ['Value'] #rename column to Value

                #st.dataframe(desc_df_T, use_container_width=True)
                st.dataframe(desc_df_T, width='content')
    
    # Problematic-substructure detection and highlighting
    st.markdown("### Automated structural alerts and highlights")
    if not smiles_input or mol is None:
        st.error("Provide a valid SMILES string to run the structural alerts.")
    else:
        alerts = []  # list of dicts: {"name":..., "desc":..., "atom_idxs": [...]}

        # 1) Built-in RDKit filter catalogs (PAINS, REACTIVE) if available
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
            #params.AddCatalog(FilterCatalogParams.FilterCatalogs.REACTIVE)
            catalog = FilterCatalog(params)
            matches = catalog.GetMatches(mol)
            for match in matches:
                # best-effort extraction of atom indices for highlighting
                atom_idxs = []
                try:
                    atom_idxs = list(match.GetAtomIds())  # FilterMatch method if present
                except Exception:
                    # fallback: try to get the SMARTS/pattern molecule and re-match
                    try:
                        patt = match.GetPatternMol()
                        #st.write(f"{patt}")
                        smatches = mol.GetSubstructMatches(patt)
                        #st.write(f"{smatches}")
                        if smatches:
                            atom_idxs = list(smatches[0])
                    except Exception:
                        atom_idxs = []
                alert_name = match.GetDescription() if hasattr(match, "GetDescription") else "RDKit Filter Match"
                alerts.append({"name": f"{alert_name}", "desc": alert_name, "atom_idxs": atom_idxs})
                #st.write(f"{atom_idxs}")
        except Exception as e:
            st.info("RDKit FilterCatalog unavailable or raised an error; skipping built-in filters.")
            # continue without failing

        # 2) Custom SMARTS-based structural alerts (common problematic groups)
        custom_alerts = [
            ("Nitro", "[NX3](=O)=O", "Possible nitro group"),
            ("Aldehyde", "[CX3H1](=O)[#6]", "Aliphatic aldehyde"),
            ("Acyl halide", "[CX3](=O)[Cl,Br,I,F]", "Potentially reactive acyl halide"),
            ("Epoxide (3-membered oxygen ring)", "C1OC1", "Small strained epoxide"),
            ("Michael acceptor (alpha,beta-unsaturated carbonyl)", "C=CC(=O)[#6]", "Electrophilic Michael acceptor"),
            ("Free thiol", "[SH]", "Free thiol / sulfhydryl"),
            ("Peroxide", "OO", "Peroxide moiety"),
            ("Isocyanate", "N=C=O", "Isocyanate functional group"),
        ]
        for name, smarts, desc in custom_alerts:
            try:
                patt = Chem.MolFromSmarts(smarts)
                if patt:
                    matches = mol.GetSubstructMatches(patt)
                    for m in matches:
                        atom_idxs = list(m)
                        alerts.append({"name": name, "desc": desc + f" ({smarts})", "atom_idxs": atom_idxs})
            except Exception:
                # ignore malformed SMARTS or matching errors
                continue

        # 3) Simple property-based flags (Rule-of-5 style)
        ro5_violations = []
        try:
            mw = Descriptors.MolWt(mol)
            clogp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            if mw > 500:
                ro5_violations.append(f"Molecular weight = {mw:.1f} (>500)")
            if clogp > 5:
                ro5_violations.append(f"cLogP = {clogp:.2f} (>5)")
            if hbd > 5:
                ro5_violations.append(f"H-bond donors = {hbd} (>5)")
            if hba > 10:
                ro5_violations.append(f"H-bond acceptors = {hba} (>10)")
        except Exception:
            ro5_violations = []

        # Display summary
        if not alerts and not ro5_violations:
            st.success("No built-in-filter or custom SMARTS alerts found; basic Ro5 metrics are OK.")
        else:
            if alerts:
                st.warning(f"Found {len(alerts)} structural alert(s). Expand below to inspect and visualize them.")
            if ro5_violations:
                st.error("Rule-of-5 style violations detected:")
                for v in ro5_violations:
                    st.write(f" - {v}")

        # Render each alert with highlighted structure
        if alerts:
            with st.expander("Show detected structural alerts and highlighted substructures", expanded=True):
                for idx, a in enumerate(alerts, 1):
                    title = f"{idx}. {a['name']}"
                    st.markdown(f"**{title}** — {a.get('desc','')}")
                    atom_idxs = a.get("atom_idxs", []) or []
                    if atom_idxs:
                        # determine bonds between highlighted atoms for nicer rendering
                        bond_idxs = []
                        for b in mol.GetBonds():
                            a1 = b.GetBeginAtomIdx()
                            a2 = b.GetEndAtomIdx()
                            if a1 in atom_idxs and a2 in atom_idxs:
                                bond_idxs.append(b.GetIdx())
                        # Draw highlighted molecule
                        try:
                            img = Draw.MolToImage(mol, size=(420, 200), highlightAtoms=atom_idxs, highlightBonds=bond_idxs)
                            st.image(img, use_column_width=False)
                        except Exception:
                            # fallback drawing with rdMolDraw2D
                            try:
                                drawer = rdMolDraw2D.MolDraw2DCairo(420, 200)
                                opts = drawer.drawOptions()
                                rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=atom_idxs, highlightBonds=bond_idxs)
                                drawer.FinishDrawing()
                                img_bytes = drawer.GetDrawingText()
                                st.image(img_bytes)
                            except Exception:
                                st.info("Could not render highlighted image for this alert.")
                    # else:
                    #     st.info("Alert triggered, but no specific atom indices were returned to highlight.")

#############################################################################
#   R-group Analysis
#############################################################################
if selected == "Taylor-Butina Clustering":
    st.title("🔬 Taylor-Butina Clustering")
    st.markdown("Taylor-Butina clustering is a widely used algorithm in cheminformatics for clustering chemical compounds based on their structural similarity. It is particularly useful for organizing large chemical libraries into smaller, more manageable groups of similar molecules.")
    st.markdown("""
    In this section, you can:

    - Read the input data from a CSV file
    - Cluster the input data to identify similar molecules
    - View the largest cluster and identify representative molecules from each cluster
                """)
    
    # Load default CSV file (local or URL)
    default_file_path = 'https://raw.githubusercontent.com/ganesh7shahane/streamlit_apps/refs/heads/main/data/chembl1075104.csv'  # adjust path or use URL
    default_df = pd.read_csv(default_file_path, index_col=False)
    st.subheader("Upload the CSV file")
    uploaded_file = st.file_uploader("", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file, index_col=False)
    else:
        df = default_df
        st.info("Using default CSV file for sample SAR analysis")
    
    st.subheader("Let's see how the dataset looks like")
    rows = st.slider("Choose rows to display",1,len(df))
    
    st.dataframe(df.head(rows), width='stretch')
    # Show the total number of rows
    st.info(f"It appears there are {df.shape[0]} molecules in the dataset.")
    
    # Find the SMILES column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break
    if smiles_col:
        st.write(f"Identified SMILES column: {smiles_col}")
    else:
        st.error("No SMILES column found (case-insensitive). Some functionalities will be limited.")
    
    st.subheader("Perform butina clustering")
    st.write("First, we calculate molecular fingerprints and then perform butina clustering based on them. This helps identify diverse subsets or clusters of similar compounds within a dataset.")
    #Compute mol
    df.loc[:, 'mol'] = df[smiles_col].apply(Chem.MolFromSmiles)
    #Compute fingerprints and perform butina clustering
    # Let the user choose fingerprint parameters
    col1, col2, col3 = st.columns(3)
    nBits = col1.selectbox("Fingerprint size (bits)", options=[1024, 2048], index=1, help="Number of bits for the Morgan fingerprint, better to keep it 2048")
    radius = col2.selectbox("Morgan radius", options=[1, 2, 3], index=1, help="Radius for the Morgan fingerprint, better to keep it 2")
    sim_cutoff = col3.slider("Similarity cutoff", min_value=0.1, max_value=0.9, value=0.2, step=0.05, help="Similarity cutoff for butina clustering, lower values result in more clusters. A cutoff of 0.2 means molecules with Tanimoto similarity >= 0.8 will be clustered together. A cutoff of 0.2 is often a good starting point for diverse clustering.")
    st.info(f"Computing Morgan fingerprints (radius={radius}, nBits={nBits})...")

    # Compute Morgan fingerprints (store RDKit ExplicitBitVect objects)
    df.loc[:, 'fp'] = df.mol.apply(lambda m: AllChem.GetMorganFingerprintAsBitVect(m, int(radius), nBits) if m is not None else None)
    # perform Taylor-Butina clustering using the user-selected similarity cutoff
    df.loc[:, 'cluster'] = uru.taylor_butina_clustering(df.fp.values, cutoff=float(sim_cutoff))
    
    st.write(f"Total clusters identified: {df.cluster.nunique()}")
        
    #display custer 
    st.dataframe(df.cluster.value_counts(), width='content')
     #display selectboxes side by side
     
    st.subheader("Visualise molecules in a cluster")

    col1, col2, col3 = st.columns(3)
    with col1:  
        cluster_ids = sorted(df.cluster.unique())
        cluster_to_visualize = st.selectbox("Select a cluster", options=cluster_ids, index=0)
    with col2:
        #Choose legend from dropdown, list of columns, not mol column
        legend_option = st.selectbox("Choose legend to display:", options=[col for col in df.columns if col not in ['mol','fp','cluster']], index=0)
    with col3:
        legend_2_option = st.selectbox("Choose another legend to display (optional):", options=[col for col in df.columns if col not in ['mol','fp','cluster',legend_option]] + ["None"], index=len([col for col in df.columns if col not in ['mol','fp','cluster',legend_option]]))
    df.loc[:, 'legend'] = df[legend_option].astype(str)
    df.loc[:, 'legend_2'] = df[legend_2_option].astype(str) if legend_2_option != "None" else ""
    #df.loc[:, 'legend'] = df['legend'] + " " + df['legend_2']

    html_data = mols2grid.display(df.query(f"cluster == {cluster_to_visualize}"), mol_col="mol",subset=["img","legend","legend_2"], size=(200, 200), n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
        
    num_molecules = df.query(f"cluster == {cluster_to_visualize}").shape[0]
    if num_molecules >8 and num_molecules < 13:
        st.components.v1.html(html_data, height=950, scrolling=True)
    elif num_molecules <=4:
        st.components.v1.html(html_data, height=500, scrolling=True)
    elif num_molecules >4 and num_molecules <=8:
        st.components.v1.html(html_data, height=700, scrolling=True)
    else:
        st.components.v1.html(html_data, height= 1200, scrolling=True)
    
    st.write("Let's look at the dataframe with the cluster assignments")
    st.dataframe(df)

    #Put a streamlit download button to download the dataframe with cluster assignments as CSV
    compute = st.button("⬇️ Prepare DataFrame with cluster assignments as SDF",
        type="primary"
    )
    if compute:

        # Create a button to download the SDF file
        sdf_buffer = StringIO()
        PandasTools.WriteSDF(df, sdf_buffer, molColName='mol', properties=list(df.columns), idName=None)
        sdf_data = sdf_buffer.getvalue()
        sdf_buffer.close()
        
        st.download_button(
            label="⬇️ Download SDF",
            data=sdf_data,
            file_name="molecules_with_clusters.sdf",
            mime="chemical/x-mdl-sdfile",
            type="primary"
        )
        
    
    # #Text input to enter mol file
    # st.subheader("Define core or scaffold")
    # #Enter SMILES
    # smiles_inp = st.text_input("Enter SMILES of the scaffold", value="[*]C(=O)c1ccc(N[*])c(O[*])c1")
    # smiles_inp_mol = Chem.MolFromSmiles(smiles_inp)
    # AllChem.Compute2DCoords(smiles_inp_mol)
    # # Convert to MOL block string
    # mol_block = Chem.MolToMolBlock(smiles_inp_mol)
    # #Convert the mol block to mol object
    # core = Chem.MolFromMolBlock(mol_block)

    # #Display the core molecule
    # if core:
    #     st.write("Here's the core molecule")
    #     img = Draw.MolToImage(core, size=(300, 300), kekulize=True)
    #     st.image(img, caption="Scaffold structure")

    #     st.write(f"Now let's create a new Pandas dataframe from the molecules in cluster {cluster_to_visualize}.")
    #     #Now let's create a new Pandas dataframe from the molecules in cluster 0. 
    #     df_0 = df.query(f"cluster == 0").copy()

    #     #we'll add and index column to keep track of things
    #     df_0.loc[:, 'index'] = range(0,len(df_0))
    #     st.write(f"There are {len(df_0)} molecules in cluster {cluster_to_visualize}.")
    #     st.dataframe(df_0, width='stretch')
        
        
    #     st.subheader("Perform R-group decomposition on molecules")
    #     st.markdown(
    #     """
    #     As mentioned above, we're using the function rdRGroupDecomposition.RGroupDecompose from the RDKit. 
        
    #     Note that RDKit returns two values from this function.
        
    #     rgd - a dictionary containing the results of the R-group decomposition. This dictionary has keys containing the core with a key "Core", and the R-groups in keys named "R1", "R2", etc. Each dictionary key links to a list of cores or R-groups corresponding input molecules that matched the core (didn't fail).
        
    #     failed - a list containing the indices of molecules that did not match the core.
    #     """
    #     )
    #     #
        
    #     rgd,failed = rdRGroupDecomposition.RGroupDecompose([core],df_0.mol.values,asRows=False)

    #     #rgd_core = mols2grid.display(pd.DataFrame(rgd), mol_col="Core", size=(200, 200), n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
    #     #st.components.v1.html(rgd_core, height=1100, scrolling=True)
    #     st.write("Let's look at the first core, all the rest should be the same. ")
    #     st.image(Draw.MolToImage(rgd['Core'][0], size=(300, 300), kekulize=True), caption="Core used for R-group decomposition")