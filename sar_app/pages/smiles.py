"""
SMILES Analysis Page
Individual molecule analysis with property calculation and drug-likeness assessment.
"""

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, BRICS, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import mols2grid
import sys
import os
from rdkit import RDConfig

from sar_app.base import BaseAnalyzer
from sar_app.utils import MoleculeUtils

# Try to import sascorer (optional)
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORER = True
except:
    HAS_SASCORER = False


class SMILESAnalyzer(BaseAnalyzer):
    """Individual SMILES analysis and property calculation."""
    
    def render(self):
        """Render the SMILES analysis page."""
        st.title("🧪 SMILES Analysis")
        
        st.markdown("""
        Analyze individual molecules:
        - :red[Draw] 2D structure
        - :red[Calculate] molecular properties
        - :red[Assess] drug-likeness (Lipinski, QED, SA Score)
        - :red[Fragment] molecules (BRICS)
        """)
        
        # Get SMILES input
        smiles = self._get_smiles_input()
        
        if not smiles:
            return
        
        # Validate
        mol = MoleculeUtils.smiles_to_mol(smiles)
        if not mol:
            st.error("❌ Invalid SMILES")
            return
        
        # Canonical SMILES
        canonical = MoleculeUtils.mol_to_smiles(mol)
        if canonical != smiles:
            st.info(f"📌 Canonical SMILES: `{canonical}`")
        
        # Display sections
        self._display_structure(mol)
        self._display_properties(mol)
        self._display_druglikeness(mol)
        self._display_admet_properties(smiles)
        self._display_substructure_alerts(smiles, mol)
        self._display_fragments(smiles)
        #self._display_fingerprints(mol)
    
    def _get_smiles_input(self) -> str:
        """Get SMILES from input or sidebar."""
        # Check sidebar
        sidebar_smiles = None
        if 'sketched_smiles' in st.session_state and st.session_state.sketched_smiles:
            if st.checkbox("Use molecule from sidebar", value=True):
                sidebar_smiles = st.session_state.sketched_smiles
                st.success(f"✅ Using: `{sidebar_smiles}`")
        
        # Manual input
        smiles = st.text_input(
            "Enter SMILES",
            value="[C@H]12CN(C[C@H](CC1)N2)c1c2c(nc(n1)OC[C@@]13CCCN1C[C@@H](C3)F)c(c(nc2)c1cc(cc2ccc(c(c12)C#C)F)O)F",
            help="KRAS inhibitor by default"
        )
        
        # Auto-run Analyse on first page load by storing the SMILES in session state
        if 'current_smiles' not in st.session_state:
            st.session_state.current_smiles = sidebar_smiles if sidebar_smiles else smiles
            st.success("🔍 Analysis started automatically")
            return st.session_state.current_smiles

        # Manual re-run button
        if st.button("🔍 Analyse"):
            st.session_state.current_smiles = sidebar_smiles if sidebar_smiles else smiles
            return st.session_state.current_smiles
        
        # Return the previously analyzed SMILES if it exists
        if 'current_smiles' in st.session_state:
            return st.session_state.current_smiles
        
        return ""
    
    def _display_structure(self, mol):
        """Display 2D structure."""
        st.subheader("🧬 2D Structure")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            img = Draw.MolToImage(mol, size=(600, 600))
            st.image(img)
        
        with col2:
            st.markdown("### Lipinski Descriptors")
            st.metric("MW", f"{Descriptors.MolWt(mol):.2f} g/mol")
            st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")
            st.metric("TPSA", f"{Descriptors.TPSA(mol):.2f} Å²")
            st.metric("HBA", Descriptors.NumHAcceptors(mol))
            st.metric("HBD", Descriptors.NumHDonors(mol))
            st.metric("Rotatable Bonds", Descriptors.NumRotatableBonds(mol))
    
    def _display_properties(self, mol):
        """Display molecular properties."""
        st.markdown("---")
        st.subheader("📊 Other Properties")
        
        props = {
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Aliphatic Rings': Descriptors.NumAliphaticRings(mol),
            'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
            'Fraction Csp3': Lipinski.FractionCSP3(mol),
            'Molar Refractivity': Descriptors.MolMR(mol)
        }
        
        col1, col2 = st.columns(2)
        items = list(props.items())
        mid = len(items) // 2
        
        with col1:
            for name, value in items[:mid]:
                if isinstance(value, float):
                    st.metric(name, f"{value:.3f}")
                else:
                    st.metric(name, value)
        
        with col2:
            for name, value in items[mid:]:
                if isinstance(value, float):
                    st.metric(name, f"{value:.3f}")
                else:
                    st.metric(name, value)
    
    def _display_druglikeness(self, mol):
        """Display drug-likeness assessment."""
        st.markdown("---")
        st.subheader("💊 Drug-likeness Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # QED
            try:
                from rdkit.Chem import QED
                qed = QED.qed(mol)
                st.metric("QED Score", f"{qed:.3f}", help="0=not drug-like, 1=very drug-like")
                if qed >= 0.67:
                    st.success("✅ Good drug-likeness")
                elif qed >= 0.4:
                    st.info("⚠️ Moderate drug-likeness")
                else:
                    st.warning("❌ Poor drug-likeness")
            except:
                st.info("ℹ️ QED Score unavailable")
        
        with col2:
            # SA Score
            if HAS_SASCORER:
                try:
                    sa = sascorer.calculateScore(mol)
                    st.metric("SA Score", f"{sa:.2f}", help="1=easy, 10=difficult to synthesize")
                    if sa <= 3:
                        st.success("✅ Easy synthesis")
                    elif sa <= 6:
                        st.info("⚠️ Synthesis with intermediate difficulty")
                    else:
                        st.warning("❌ Difficult synthesis")
                except:
                    st.info("ℹ️ SA Score calculation failed")
            else:
                st.info("ℹ️ SA Score unavailable (sascorer not found)")
    
    def _display_admet_properties(self, smiles: str):
        """Display ADMET properties using Admetica (automatically calculated)."""
        import subprocess
        import tempfile
        import os
        import sys
        
        st.markdown("---")
        st.subheader("☘️ ADMET Properties")
        
        with st.spinner("Calculating ADMET properties..."):
            try:
                # Get the admetica_predict command path
                python_dir = os.path.dirname(sys.executable)
                admetica_cmd = os.path.join(python_dir, 'admetica_predict')
                
                # Check if command exists
                if not os.path.exists(admetica_cmd):
                    st.error("❌ Admetica CLI not found. Please install: pip install admetica==1.4.1")
                    return
                
                # Create temporary files
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
                    temp_input_path = temp_input.name
                    pd.DataFrame({'SMILES': [smiles]}).to_csv(temp_input_path, index=False)
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
                    temp_output_path = temp_output.name
                
                results = {}
                
                # Calculate absorption properties
                try:
                    absorption_props = "Caco2,Solubility,Lipophilicity,Pgp-Inhibitor,Pgp-Substrate"
                    cmd = [
                        admetica_cmd,
                        '--dataset-path', temp_input_path,
                        '--smiles-column', 'SMILES',
                        '--properties', absorption_props,
                        '--save-path', temp_output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        result_df = pd.read_csv(temp_output_path)
                        for col in result_df.columns:
                            if col != 'SMILES':
                                results[col] = result_df[col].iloc[0]
                except subprocess.TimeoutExpired:
                    st.warning("⚠️ Absorption calculation timed out")
                except Exception as e:
                    st.warning(f"⚠️ Absorption calculation failed: {str(e)}")
                
                # Calculate toxicity properties
                try:
                    toxicity_props = "hERG,LD50"
                    cmd = [
                        admetica_cmd,
                        '--dataset-path', temp_input_path,
                        '--smiles-column', 'SMILES',
                        '--properties', toxicity_props,
                        '--save-path', temp_output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        result_df = pd.read_csv(temp_output_path)
                        for col in result_df.columns:
                            if col != 'SMILES':
                                results[col] = result_df[col].iloc[0]
                except subprocess.TimeoutExpired:
                    st.warning("⚠️ Toxicity calculation timed out")
                except Exception as e:
                    st.warning(f"⚠️ Toxicity calculation failed: {str(e)}")
                
                # Display results
                if results:
                    col1, col2 = st.columns(2)
                    
                    # Heuristically separate toxicity-related keys from other ADMET outputs
                    toxicity_indicators = ['herg', 'ld50', 'tox', 'toxic']
                    absorption_keys = [k for k in results.keys() if not any(t in k.lower() for t in toxicity_indicators)]
                    toxicity_keys = [k for k in results.keys() if any(t in k.lower() for t in toxicity_indicators)]
                    
                    with col1:
                        st.markdown("#### 💧 Absorption")
                        
                        if not absorption_keys:
                            st.write("No absorption properties returned.")
                        
                        for key in absorption_keys:
                            value = results[key]
                            try:
                                # Format numeric values
                                numeric = float(value)
                                st.metric(key, f"{numeric:.3f}")
                            except Exception:
                                st.metric(key, str(value))
                    
                    with col2:
                        st.markdown("#### 🔴 Toxicity")
                        
                        if not toxicity_keys:
                            st.write("No toxicity properties returned.")
                        
                        for key in toxicity_keys:
                            value = results[key]
                            kl = key.lower()
                            
                            if 'herg' in kl:
                                try:
                                    herg_float = float(value)
                                    st.metric("hERG (Cardiac)", f"{herg_float:.3f}")
                                    if herg_float > 0.7:
                                        st.error("⚠️ High cardiac risk")
                                    elif herg_float > 0.3:
                                        st.warning("⚠️ Moderate risk")
                                    else:
                                        st.success("✅ Low risk")
                                except Exception:
                                    st.metric("hERG (Cardiac)", str(value))
                            
                            elif 'ld50' in kl:
                                try:
                                    ld50_float = float(value)
                                    st.metric("LD50 (Acute)", f"{ld50_float:.3f}")
                                    if ld50_float > 3.0:
                                        st.error("⚠️ High toxicity")
                                    elif ld50_float > 2.0:
                                        st.warning("⚠️ Moderate toxicity")
                                    else:
                                        st.success("✅ Low toxicity")
                                except Exception:
                                    st.metric("LD50 (Acute)", str(value))
                            
                            else:
                                # Generic toxicity/other properties
                                try:
                                    numeric = float(value)
                                    st.metric(key, f"{numeric:.3f}")
                                except Exception:
                                    st.metric(key, str(value))
                        
                        if 'LD50' in results:
                            ld50_val = results['LD50']
                            try:
                                ld50_float = float(ld50_val)
                                st.metric("LD50 (Acute)", f"{ld50_float:.3f}")
                                if ld50_float > 3.0:
                                    st.error("⚠️ High toxicity")
                                elif ld50_float > 2.0:
                                    st.warning("⚠️ Moderate toxicity")
                                else:
                                    st.success("✅ Low toxicity")
                            except:
                                st.metric("LD50 (Acute)", str(ld50_val))
                    
                    # Link to ADMET Documentation
                    if st.button("📖 View ADMET Documentation", type="secondary", use_container_width=True):
                        st.session_state.current_page = "ADMET Documentation"
                        st.rerun()
                else:
                    st.error("❌ Failed to calculate ADMET properties")
                
                # Clean up
                try:
                    os.unlink(temp_input_path)
                    os.unlink(temp_output_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def _display_fragments(self, smiles):
        """Display molecular fragments using mols2grid."""
        st.markdown("---")
        st.subheader("🔬 BRICS Fragments")
        
        try:
            mol = MoleculeUtils.smiles_to_mol(smiles)
            frags = list(BRICS.BRICSDecompose(mol))
            
            if frags:
                st.write(f"Found **{len(frags)} fragments**")
                
                # Create DataFrame for fragments
                frag_data = []
                for i, frag_smiles in enumerate(frags, 1):
                    try:
                        frag_mol = Chem.MolFromSmiles(frag_smiles)
                        if frag_mol:
                            frag_data.append({
                                'Fragment_ID': i,
                                'SMILES': frag_smiles,
                                'mol': frag_mol,
                                'Formula': Chem.rdMolDescriptors.CalcMolFormula(frag_mol),
                                'MW': round(Descriptors.MolWt(frag_mol), 2),
                                'Atoms': frag_mol.GetNumAtoms(),
                                'Bonds': frag_mol.GetNumBonds()
                            })
                    except Exception as e:
                        st.warning(f"Could not process fragment {i}: {frag_smiles}")
                        continue
                
                if frag_data:
                    df_frags = pd.DataFrame(frag_data)
                    
                    # Display using mols2grid                    
                    try:
                        html_data = mols2grid.display(
                            df_frags,
                            mol_col='mol',
                            subset=["img", "Fragment_ID", "SMILES"],
                            size=(200, 200),
                            n_items_per_page=12,
                            fixedBondLength=25,
                            clearBackground=False
                        )._repr_html_()
                        
                        # Adjust height based on number of fragments
                        if len(df_frags) <= 4:
                            height = 500
                        elif len(df_frags) <= 8:
                            height = 700
                        elif len(df_frags) <= 12:
                            height = 900
                        else:
                            height = 1100
                        
                        st.components.v1.html(html_data, height=height, scrolling=True)
                    except Exception as e:
                        st.error(f"Error displaying fragments grid: {str(e)}")
                        st.write("Falling back to table view:")
                        st.dataframe(df_frags.drop(columns=['mol']), use_container_width=True)
                else:
                    st.warning("No valid fragments could be generated")
            else:
                st.info("No fragments found")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    def _display_fingerprints(self, mol):
        """Display molecular fingerprints."""
        st.markdown("---")
        st.subheader("🔑 Molecular Fingerprints")
        
        fp_types = st.multiselect(
            "Select fingerprint types",
            ["Morgan", "RDKit", "MACCS"],
            default=["Morgan"]
        )
        
        for fp_type in fp_types:
            with st.expander(f"{fp_type} Fingerprint"):
                try:
                    if fp_type == "Morgan":
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    elif fp_type == "RDKit":
                        fp = Chem.RDKFingerprint(mol)
                    else:  # MACCS
                        from rdkit.Chem import MACCSkeys
                        fp = MACCSkeys.GenMACCSKeys(mol)
                    
                    fp_str = fp.ToBitString()
                    bits_on = fp_str.count('1')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Bits", len(fp_str))
                    with col2:
                        st.metric("Bits ON", bits_on)
                    with col3:
                        st.metric("Density", f"{bits_on/len(fp_str):.3f}")
                    
                    st.code(fp_str[:200] + "..." if len(fp_str) > 200 else fp_str)
                    
                    st.download_button(
                        f"⬇️ Download",
                        fp_str,
                        f"{fp_type}_fp.txt",
                        key=f"download_{fp_type}"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    def _display_substructure_alerts(self, smiles, mol):
        """Display structural alerts and problematic substructures."""
        st.markdown("---")
        st.subheader("⚠️ Automated Structural Alerts")
        
        alerts = []  # list of dicts: {"name":..., "desc":..., "atom_idxs": [...]}
        
        # 1) Built-in RDKit filter catalogs (PAINS, REACTIVE, etc.)
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
            catalog = FilterCatalog(params)
            matches = catalog.GetMatches(mol)
            
            for match in matches:
                # Extract atom indices for highlighting
                atom_idxs = []
                try:
                    atom_idxs = list(match.GetAtomIds())
                except Exception:
                    # Fallback: try to get the pattern molecule and re-match
                    try:
                        patt = match.GetPatternMol()
                        smatches = mol.GetSubstructMatches(patt)
                        if smatches:
                            atom_idxs = list(smatches[0])
                    except Exception:
                        atom_idxs = []
                
                alert_name = match.GetDescription() if hasattr(match, "GetDescription") else "RDKit Filter Match"
                alerts.append({"name": f"{alert_name}", "desc": alert_name, "atom_idxs": atom_idxs})
        except Exception as e:
            st.info("RDKit FilterCatalog unavailable or raised an error; skipping built-in filters.")
        
        # 2) Custom SMARTS-based structural alerts
        custom_alerts = [
            ("Nitro", "[NX3](=O)=O", "Possible nitro group"),
            ("Aldehyde", "[CX3H1](=O)[#6]", "Aliphatic aldehyde"),
            ("Acyl halide", "[CX3](=O)[Cl,Br,I,F]", "Potentially reactive acyl halide"),
            ("Epoxide", "C1OC1", "Small strained epoxide"),
            ("Michael acceptor", "C=CC(=O)[#6]", "Electrophilic Michael acceptor"),
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
                continue
        
        # 3) Property-based flags (Rule-of-5 style)
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
            st.success("✅ No structural alerts or Rule-of-5 violations found!")
        else:
            if alerts:
                st.warning(f"Found {len(alerts)} structural alert(s)")
            if ro5_violations:
                st.error("Rule-of-5 violations detected:")
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
                        # Determine bonds between highlighted atoms
                        bond_idxs = []
                        for b in mol.GetBonds():
                            a1 = b.GetBeginAtomIdx()
                            a2 = b.GetEndAtomIdx()
                            if a1 in atom_idxs and a2 in atom_idxs:
                                bond_idxs.append(b.GetIdx())
                        
                        # Draw highlighted molecule
                        try:
                            img = Draw.MolToImage(
                                mol,
                                size=(420, 200),
                                highlightAtoms=atom_idxs,
                                highlightBonds=bond_idxs
                            )
                            st.image(img, use_column_width=False)
                        except Exception:
                            # Fallback drawing
                            try:
                                drawer = rdMolDraw2D.MolDraw2DCairo(420, 200)
                                rdMolDraw2D.PrepareAndDrawMolecule(
                                    drawer, mol,
                                    highlightAtoms=atom_idxs,
                                    highlightBonds=bond_idxs
                                )
                                drawer.FinishDrawing()
                                img_bytes = drawer.GetDrawingText()
                                st.image(img_bytes)
                            except Exception:
                                st.info("Could not render highlighted image for this alert.")
                    
                    st.markdown("---")
