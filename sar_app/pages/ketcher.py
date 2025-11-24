"""
Ketcher Chemical Sketcher Page
Provides an interactive molecular drawing interface.
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from streamlit_ketcher import st_ketcher

from sar_app.base import BaseAnalyzer
from sar_app.config import AppConfig
from sar_app.utils import MoleculeUtils


class KetcherAnalyzer(BaseAnalyzer):
    """Ketcher chemical sketcher page."""
    
    def render(self):
        """Render the Ketcher sketcher page."""
        st.title("🧪 Chemical Sketcher")
        
        # Introduction
        self._display_intro()
        
        # Ketcher sketcher
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Draw Your Molecule")
            molecule_smiles = st_ketcher()
        
        with col2:
            if molecule_smiles:
                self._display_molecule_info(molecule_smiles)
            else:
                pass
        
        # Store in session state
        if molecule_smiles:
            st.session_state['sketched_smiles'] = molecule_smiles
            try:
                st.session_state['sketched_mol'] = Chem.MolFromSmiles(molecule_smiles)
            except:
                st.session_state['sketched_mol'] = None
        
        # Display detailed analysis if molecule is valid
        if molecule_smiles:
            self._display_detailed_analysis(molecule_smiles)
    
    def _display_intro(self):
        """Display page introduction."""
        st.markdown("""
        Use the Ketcher molecular editor to:
        - :red[Draw] chemical structures interactively
        - :red[View] SMILES representation
        - :red[Analyze] molecular properties
        - :red[Export] to other analysis pages
        
        💡 **Tip:** After drawing, go to 'SMILES Analysis' for detailed analysis!
        """)
        st.markdown("---")
    
    def _display_molecule_info(self, smiles: str):
        """Display basic molecule information."""
        st.subheader("Generated SMILES:")
        st.code(smiles, language="text")
        
        if st.button("📋 Copy SMILES", width='stretch'):
            st.toast("SMILES copied to clipboard!", icon="✅")
        
        # Quick properties
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.markdown("---")
                st.subheader("Quick Properties:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mol. Weight", f"{Descriptors.MolWt(mol):.2f}")
                    st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                    st.metric("H-Bond Donors", Descriptors.NumHDonors(mol))
                
                with col2:
                    st.metric("TPSA", f"{Descriptors.TPSA(mol):.2f}")
                    st.metric("Rot. Bonds", Descriptors.NumRotatableBonds(mol))
                    st.metric("H-Bond Acceptors", Descriptors.NumHAcceptors(mol))
        except:
            st.warning("⚠️ Invalid SMILES structure")
    
    def _display_detailed_analysis(self, smiles: str):
        """Display detailed molecular analysis."""
        st.markdown("---")
        st.subheader("📊 Detailed Analysis")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("Invalid SMILES - cannot generate molecule")
                return
            
            # Molecular formula and structure
            with st.expander("Molecular Formula & Structure", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Molecular Formula:**")
                    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
                    st.code(formula, language="text")
                    
                    st.write("**InChI:**")
                    inchi = Chem.MolToInchi(mol)
                    st.code(inchi, language="text")
                    
                    st.write("**InChIKey:**")
                    inchikey = Chem.MolToInchiKey(mol)
                    st.code(inchikey, language="text")
                
                with col2:
                    st.write("**2D Structure:**")
                    img = MoleculeUtils.mol_to_image(mol, size=(1000, 1000))
                    if img:
                        st.image(img, width='stretch')
            
            # Lipinski's Rule of Five
            with st.expander("Lipinski's Rule of Five"):
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                violations = 0
                checks = []
                
                if mw > 500:
                    violations += 1
                    checks.append(("❌", "Molecular Weight", f"{mw:.2f}", "> 500"))
                else:
                    checks.append(("✅", "Molecular Weight", f"{mw:.2f}", "≤ 500"))
                
                if logp > 5:
                    violations += 1
                    checks.append(("❌", "LogP", f"{logp:.2f}", "> 5"))
                else:
                    checks.append(("✅", "LogP", f"{logp:.2f}", "≤ 5"))
                
                if hbd > 5:
                    violations += 1
                    checks.append(("❌", "H-Bond Donors", str(hbd), "> 5"))
                else:
                    checks.append(("✅", "H-Bond Donors", str(hbd), "≤ 5"))
                
                if hba > 10:
                    violations += 1
                    checks.append(("❌", "H-Bond Acceptors", str(hba), "> 10"))
                else:
                    checks.append(("✅", "H-Bond Acceptors", str(hba), "≤ 10"))
                
                # Display results
                if violations == 0:
                    st.success("✅ **Passes Lipinski's Rule of Five!**")
                elif violations == 1:
                    st.warning(f"⚠️ **1 violation of Lipinski's Rule**")
                else:
                    st.error(f"❌ **{violations} violations of Lipinski's Rule**")
                
                # Display table
                st.markdown("| Status | Property | Value | Threshold |")
                st.markdown("|--------|----------|-------|-----------|")
                for check in checks:
                    st.markdown(f"| {check[0]} | {check[1]} | {check[2]} | {check[3]} |")
            
            # Additional descriptors
            with st.expander("Additional Descriptors"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Heavy Atoms", Descriptors.HeavyAtomCount(mol))
                    st.metric("Aromatic Rings", Descriptors.NumAromaticRings(mol))
                    st.metric("Aliphatic Rings", Descriptors.NumAliphaticRings(mol))
                
                with col2:
                    st.metric("Heteroatoms", Descriptors.NumHeteroatoms(mol))
                    st.metric("Sp3 Carbons", Descriptors.NumAliphaticCarbocycles(mol))
                    st.metric("Stereocenters", len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
                
                with col3:
                    st.metric("Rings", Descriptors.RingCount(mol))
                    st.metric("Saturated Rings", Descriptors.NumSaturatedRings(mol))
                    try:
                        from rdkit.Chem import Lipinski
                        st.metric("Fraction Csp3", f"{Lipinski.FractionCSP3(mol):.3f}")
                    except:
                        st.metric("Fraction Csp3", "N/A")
            
        except Exception as e:
            st.error(f"Error analyzing molecule: {str(e)}")
        
        # Back button at the bottom
        st.markdown("---")
        if st.button("← Back to Main", type="secondary", width='stretch'):
            st.session_state.current_page = "DataFrame Wizard"
            st.rerun()
