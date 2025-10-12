import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


############################################################################
#
#   Define sidebar menu
#
############################################################################


selected = option_menu(
    menu_title=None,
    options=["Home", "Identifying Scaffolds", "R-group Analysis", "Matched Molecular Pair Analysis"],
    default_index=0,
    orientation="horizontal"
)
    
if selected == "Identifying Scaffolds":
    st.title("Identifying Scaffolds from a chemical series")

if selected == "R-group Analysis":
    st.title("R-group Analysis of a chemical series")
    
if selected == "Matched Molecular Pair Analysis":
    st.title("Matched Molecular Pair Analysis (MMPA) of a chemical series")