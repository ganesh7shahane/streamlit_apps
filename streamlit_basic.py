import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem import FilterCatalog
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from rdkit.Chem import AllChem
from rdkit.Chem.rdDepictor import Compute2DCoords
import mols2grid
import io
from io import StringIO
from IPython.display import HTML
import base64
import requests
from rdkit import RDConfig
import os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

#adjust the width of the page
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:55rem;
        }
    </style>
    """
)
#Put a temporary slider
st.sidebar.title("Menu")

############################################################################
#
#   Define sidebar menu
#
############################################################################

selected = option_menu(
    menu_title="Structure-Activity Relationship (SAR) Analysis Tools",
    menu_icon="bar-chart-fill",
    options=["DataFrame Viz", "Identifying Scaffolds", "SMILES Analysis", "MMP Analysis"],
    icons=["0-square", "1-square", "2-square", "3-square"],
    default_index=0,
    orientation="horizontal"
)
if selected == "DataFrame Viz":
    st.title("Molecule DataFrame Visualisation")

    #st.header("Molecule DataFrame Analysis and Visualisation", divider='rainbow')

    st.markdown("Sometimes we have a CSV file containing SMILES and their properties, and we just wish to visualise it and analyse some of its properties. This web app is precisely for that.")
    st.markdown("Specifically, this page let's you:")
    st.markdown("- :red[View] the CSV file as a Pandas DataFrame")
    st.markdown("- :red[Compute] stats on the numerical columns: NaN count, mean, count, std, min, max etc.")
    st.markdown("- :red[Plot] histograms, barplots, correlation heatmaps of the numerical columns")
    st.markdown("- :red[Visualise] the 2D structures of the molecules")
    st.markdown("- :red[Calculate] RDKit descriptors: 2D Phys-chem descriptors, Morgan fingerprints or both")
    st.subheader("Upload the CSV file")

    uploaded_file = st.file_uploader("", type=["csv"])

    ############################################################################
    #
    #   Do all analysis on the CSV
    #
    ############################################################################

    if uploaded_file is not None:
        # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display the top 5 rows
        st.subheader("Let's see how the dataset looks like")
        rows = st.slider("Choose rows to display",1,len(df))
        st.dataframe(df.head(rows), use_container_width=True)
        
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
                st.markdown("First, let's look at the histograms of all numerical columns.")
                                
                #Give user options to change the default values
                st.info("You can change the default values of number of bins, color and figure size below.") 
                
                #shift the slider to the left column
                col1, col2, col3 = st.columns(3)
                with col1:
                    default_bins = st.slider("Number of bins", min_value=2, max_value=100, value=30, step=5)
                with col2:
                    default_color = st.color_picker("Pick a color for the histogram", value='#EE82EE')
                with col3:
                    default_size = st.slider("Figure size (width, height)", min_value=5, max_value=25, value=(14, 15), step=1)

                fig = df.hist(
                    bins=default_bins,
                    color=default_color,
                    edgecolor='black',
                    layout=(len(numeric_cols) // 2 + 1, 3),
                    figsize=default_size,
                    grid=True
                )

                plt.tight_layout()      # Prevents overlapping text
                plt.gcf().set_dpi(300) # Set DPI for high resolution

                st.pyplot(plt.gcf())    # Show the whole figure in Streamlit

            st.subheader("**Draw a single histogram or bar plot**")
            #select whether to draw a histogram or barplot
            plot_type = st.selectbox("Select plot type", options=["Histogram", "Bar Plot"])
            selected_column = st.selectbox("Select a column", options=numeric_cols)

            # Only one column should be selected; if more are selected, take the last
            if plot_type == "Histogram" and selected_column:
                st.write(f"{plot_type} for: {selected_column}")

                # Prepare histogram
                col1_1, col2_1, col3_1 = st.columns(3)
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
                st.pyplot(fig)
            
            elif plot_type == "Bar Plot" and selected_column:
                st.write(f"{plot_type} for: {selected_column}")

                # Let user select which column to plot
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                #selected_col = st.selectbox("Select a column to plot:", numeric_cols)

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
                    #also put custom ranges separated by comma
                    #custom_bins = st.text_input("Enter custom bin ranges (comma-separated):", value="")
                else:
                    bin_width = default_bin_width

                # Create bins
                bins = np.arange(col_min, col_max + bin_width, bin_width)
                df["binned"] = pd.cut(df[selected_column], bins=bins, include_lowest=True)

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

                st.pyplot(fig)

            st.subheader("Correlation Heatmap")
            st.markdown("A correlation heatmap helps to visualize the correlation coefficients between multiple numerical columns in a dataset. It provides insights into how different variables relate to each other, which can be useful for feature selection and understanding relationships in the data.")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
                sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)
            else:
                st.error("Not enough numerical columns to compute correlation heatmap.")
                
            st.subheader("Regression Plot")
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select X-axis column", options=numeric_cols, index=0)
                col2 = st.selectbox("Select Y-axis column", options=numeric_cols, index=1)
                if col1 and col2:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    sns.regplot(x=df[col1], y=df[col2], ax=ax, scatter_kws={'s': 10}, line_kws={'color': 'red'})
                    ax.set_title(f"Regression Plot: {col1} vs {col2}")
                    #Also put correlation coefficient in the title
                    corr_coef = df[[col1, col2]].corr().iloc[0, 1]
                    ax.set_title(f"Regression Plot: {col1} vs {col2} (Correlation Coefficient: {corr_coef:.2f})")
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    st.pyplot(fig)
            else:
                st.error("Not enough numerical columns to plot regression.")

        ############################################################################
        #
        #   2D Visualisation
        #
        ############################################################################
        
        # Double-ended sliders for each numerical column
        st.subheader("Visualise 2D structures of molecules")
        st.write(f"Currently, there are {len(df)} molecules in the dataset.")

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
            st.error(f"There are {len(delete_smiles)} invalid SMILES strings that RDKit cannot read.")
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
            new_df['mol_2'] = new_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
            
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
                no_restricted_df['mol_2'] = no_restricted_df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
                #AllChem.Compute2DCoords(no_restricted_df['mol_2'][0])
                #[AllChem.GenerateDepictionMatching2DStructure(m,no_restricted_df['mol_2'][0]) for m in no_restricted_df['mol_2']]
                html_data = mols2grid.display(no_restricted_df.head(display_count), mol_col='mol_2', size=(200, 200), subset=["img",legend_option],n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
                st.components.v1.html(html_data, height=1100, scrolling=True)
        
        elif len(numeric_cols) == 0:
            st.info("No numerical columns found in the dataset. Cannot apply filters.")
            if smiles_col:
                st.write("Displaying 2D structures of all molecules in the dataset.")
                df['mol_2'] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
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
        calculate = st.button("Calculate")

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
                df_to_download['mol'] = df_to_download[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
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
            label="Download SDF",
            data=sdf_data,
            file_name="molecules.sdf",
            mime="chemical/x-mdl-sdfile"
        )

##########################################################################
#   Scaffold Identification
###########################################################################

if selected == "Identifying Scaffolds":
    st.title("Identifying Scaffolds from a chemical series")    
    
    st.markdown('''
    In Cheminformatics, we frequently run into cases where we want to understand structure-activity relationships in a set of molecules. In order to do this, we typically separate the molecules based on common scaffolds, then create an R-group table to explore the substituents that have been attached to the scaffold. 
    In addition, identifying scaffolds allows us to perform tasks like aligning molecules that will enable us to more easily compare the molecules. In this notebook we'll use a method inspired by a 2019 paper by [Naveja, Vogt, Stumpfe, Medina-Franco, and Bajorath](https://pubs.acs.org/doi/full/10.1021/acsomega.8b03390). The method operates by carrying out three steps.
    
    :red[**Step 1**:]
    Decompose each molecule into a set of fragments. In this case we'll use the [FragmentMol](https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html) function from the Matched Molecular Pair Analysis (MMPA) functions in the RDKit decompose a molecule into a set of fragments. The RDKit MMPA methods use the Hussain and Rea methodology, which fragments a molecule by successively disconnecting acyclic single bonds.

    :red[**Step 2**:]
    Remove any fragments where the number of atoms is less than 2/3 the number of atoms in the original molecule.

    :red[**Step 3**:]
    Collect fragments and records their frequency. We can do this easily using the Pandas groupby method.
    ''')
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
    uploaded_file = st.file_uploader("", type=["csv"])
    
    if uploaded_file is not None:
    # Read the uploaded CSV into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display the top 5 rows
        st.subheader("Let's see how the dataset looks like")
        rows = st.slider("Choose rows to display",1,len(df))
        st.dataframe(df.head(rows), use_container_width=True)
        
        # Show the total number of rows
        st.write(f"Total rows: {df.shape[0]}")
        
        # Find the SMILES column (case-insensitive)
        smiles_col = None
        for col in df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
        
        #generate the mol objects
        if smiles_col:
            df["mol"] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x))
        else:
            st.error("No SMILES column found (case-insensitive).")
        
        #identify scaffolds
        st.subheader("Identify Scaffolds")
        st.markdown("This might take a few seconds...")
        
        #identify the smiles column in the uploaded dataframe, if any of the columns has the regex 'smiles' in any pattern
        if smiles_col:
            st.write(f"Identified SMILES column: {smiles_col}")
            
        #Calculate Murcko_Scaffold for each molecule
        df['Murcko_Scaffold'] = df[smiles_col].apply(lambda x: Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(x) if x else None)
        
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
        temp_scaffold_df['mol_2'] = temp_scaffold_df['Murcko_Scaffold'].apply(lambda x: Chem.MolFromSmiles(x))
        html_data = mols2grid.display(temp_scaffold_df, mol_col='mol_2', size=(200, 200), n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
        st.components.v1.html(html_data, height=1000, scrolling=True)
        
    ###############################################################################
    #
    #   Relate activity distribution with scaffolds
    #
    ###############################################################################
        st.subheader("Let's relate activity distribution with scaffolds")
    
        #first, we need a mol column in scaffold_df
        scaffold_df['mol'] = scaffold_df['Murcko_Scaffold'].apply(lambda x: Chem.MolFromSmiles(x))
    
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
    
        #With the functions above we can define this table with a few lines of code.
        
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
            tmp_df['mol_img'] = tmp_df.mol.apply(mol_to_base64_image)
            #st.dataframe(tmp_df)
            img_list = []
            for smi in tmp_df['Murcko_Scaffold'].values:
                img_list.append(boxplot_base64_image(df.query("Murcko_Scaffold == @smi")[activity_col].values, x_lim=[df[activity_col].min()*0.98, df[activity_col].max()*1.01]))
            tmp_df['dist_img'] = img_list
            #display the dataframe with the images
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
        tmp_df['mol'] = tmp_df[smiles_col].apply(Chem.MolFromSmiles) 
        scaffold_mol = scaffold_df.mol.values[scaffold_id] #get mol object for the scaffold into scaffold_mol
        AllChem.Compute2DCoords(scaffold_mol)
        [AllChem.GenerateDepictionMatching2DStructure(m,scaffold_mol) for m in tmp_df.mol]

        #concatenate legends
        legend_ = st.selectbox("Select legend", options=[col for col in tmp_df.columns if col != 'mol'], index=0)
        tmp_df['legend'] = tmp_df[legend_].astype(str)
        #tmp_df['legend'] = tmp_df['GV'] + ' | ' + tmp_df['ROCK2_log'].astype(str)

        #sort tmp_df as per ROCK2_log in descending order
        tmp_df_sort = tmp_df.sort_values(activity_col, ascending=False)
        #Draw molecules sorted as per ROCK2_log

        st.write(f"There are {len(tmp_df_sort)} molecules with scaffold ID {scaffold_id}")
        html_data = mols2grid.display(tmp_df_sort, mol_col='mol', size=(200, 200), subset=["img","legend"], n_items_per_page=16, fixedBondLength=25, clearBackground=False).data
        st.components.v1.html(html_data, height=1100, scrolling=True)

############################################################################################
#   SMILES Analysis
############################################################################################
if selected == "SMILES Analysis":
    #st.subheader("2D sketcher")
    from streamlit_ketcher import st_ketcher

    st.title("Chemical Sketcher")

    # Launch the sketcher widget
    smiles = st_ketcher()
    
    st.title("SMILES Analysis")
    st.write("This page lets you input a SMILES string, visualise the 2D structure, compute some common 2D phys-chem descriptors, and alerts.")
    st.subheader("Input the SMILES string")
    smiles_input = st.text_input("Enter the SMILES string here", value="Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C")
    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            st.subheader("2D Structure")
            img = Draw.MolToImage(mol, size=(1200, 400))
            st.image(img, caption="2D structure of the input SMILES", use_container_width=False)
            
            st.subheader("Calculate 2D Phys-chem Descriptors")
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

            st.dataframe(desc_df_T, use_container_width=True)

            st.subheader("Check for Alerts")
            # initialize filter
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog(params)
            

#############################################################################
#   Matched Molecular Pair Analysis
#############################################################################
if selected == "Matched Molecular Pair Analysis":
    st.title("Matched Molecular Pair Analysis (MMPA) of a chemical series")