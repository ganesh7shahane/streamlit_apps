"""
ADMET Properties Documentation
Information about Admetica models, properties, and their interpretation.
"""

import streamlit as st
from sar_app.base import BaseAnalyzer


class ADMETDocsAnalyzer(BaseAnalyzer):
    """ADMET properties documentation page."""
    
    def render(self):
        """Render the ADMET documentation page."""
        st.title("📖 ADMET Properties Documentation")
        
        st.markdown("""
        This page provides information about the ADMET (Absorption, Metabolism, and Toxicity) 
        properties predicted by Admetica, how to interpret them, and the model performance metrics.
        
        All models are built using **Chemprop** (message passing neural networks) and trained on curated datasets 
        from various sources including PubChem, ChEMBL, and academic publications.
        """)
        
        # Table of Contents
        st.markdown("---")
        st.markdown("### 📑 Table of Contents")
        st.markdown("""
        - [Absorption Properties](#absorption-properties)
        - [Metabolism Properties](#metabolism-properties)
        - [Toxicity Properties](#toxicity-properties)
        - [Model Performance & Evaluation](#model-performance-evaluation)
        - [References](#references)
        """)
        
        # Absorption Properties
        st.markdown("---")
        st.header("🔵 Absorption Properties", anchor="absorption-properties")
        
        st.markdown("""
        Absorption is the process by which drugs enter the bloodstream after administration. These properties 
        help predict bioavailability and drug delivery efficiency.
        """)
        
        # Caco-2
        with st.expander("**Caco2** - Intestinal Permeability"):
            st.markdown("""
            **Description**: The human colon epithelial cancer cell line (Caco-2) is used as an in vitro model 
            to simulate human intestinal tissue. Measures the rate at which a drug passes through intestinal walls.
            
            **Unit**: cm/s (log scale)
            
            **Interpretation**:
            - **High permeability** (> -5.0): Good intestinal absorption
            - **Moderate** (-6.0 to -5.0): Acceptable absorption
            - **Low** (< -6.0): Poor absorption, may require formulation optimization
            
            **Model Performance**:
            - Dataset size: 910 compounds
            - MAE: 0.317 | RMSE: 0.415 | R²: 0.701 | Spearman: 0.832
            - Task: Regression
            
            **Training Data**: Wang et al., curated from experimental measurements
            """)
        
        # Solubility
        with st.expander("**Solubility** - Aqueous Solubility"):
            st.markdown("""
            **Description**: Measures a drug's ability to dissolve in water. Poor water solubility can lead to 
            slow drug absorption, inadequate bioavailability, and potential toxicity issues.
            
            **Unit**: log mol/L
            
            **Interpretation**:
            - **High** (> -2): Highly soluble
            - **Moderate** (-4 to -2): Acceptable solubility
            - **Low** (< -4): Poorly soluble, formulation challenges expected
            - Note: More than 40% of new chemical entities have solubility issues
            
            **Model Performance**:
            - Dataset size: 9,982 compounds
            - MAE: 0.714 | RMSE: 1.089 | R²: 0.788 | Spearman: 0.897
            - Task: Regression
            
            **Training Data**: Sorkun et al., aggregated from multiple sources
            """)
        
        # Lipophilicity
        with st.expander("**Lipophilicity** - Octanol-Water Partition"):
            st.markdown("""
            **Description**: Measures the ability of a drug to dissolve in lipid environments (fats, oils). 
            Critical for membrane permeability and distribution.
            
            **Unit**: LogD at pH 7.4 (log of distribution coefficient)
            
            **Interpretation**:
            - **Too low** (< 0): May have poor membrane permeability
            - **Optimal** (1-3): Good balance for oral drugs
            - **Too high** (> 5): Risk of poor solubility, high metabolism, toxicity
            
            **Model Performance**:
            - Dataset size: 4,200 compounds
            - MAE: 0.399 | RMSE: 0.596 | R²: 0.748 | Spearman: 0.881
            - Task: Regression
            
            **Training Data**: AstraZeneca, Wu et al., from MoleculeNet
            """)
        
        # P-gp Inhibitor
        with st.expander("**Pgp-Inhibitor** - P-glycoprotein Inhibition"):
            st.markdown("""
            **Description**: P-glycoprotein (P-gp) is a membrane transport protein. Inhibiting P-gp can affect 
            drug-drug interactions and alter the pharmacokinetics of co-administered drugs.
            
            **Output**: Probability (0-1)
            
            **Interpretation**:
            - **High** (> 0.7): Likely P-gp inhibitor → potential for drug-drug interactions
            - **Moderate** (0.3-0.7): Uncertain, further testing recommended
            - **Low** (< 0.3): Unlikely to inhibit P-gp
            
            **Model Performance**:
            - Dataset size: 1,275 compounds (666 inhibitors, 609 non-inhibitors)
            - Specificity: 0.916 | Sensitivity: 0.863 | Accuracy: 0.888 | Balanced Acc: 0.889
            - Task: Binary classification
            
            **Training Data**: Combined from two sources (F, B., et al., Tingjun Hou)
            """)
        
        # P-gp Substrate
        with st.expander("**Pgp-Substrate** - P-glycoprotein Substrate"):
            st.markdown("""
            **Description**: Probability of being a substrate of P-glycoprotein. Substrates are actively 
            transported out of cells, leading to reduced cellular permeability.
            
            **Output**: Probability (0-1)
            
            **Interpretation**:
            - **High** (> 0.7): Likely P-gp substrate → may have reduced bioavailability
            - **Moderate** (0.3-0.7): Uncertain
            - **Low** (< 0.3): Unlikely to be effluxed by P-gp
            - Compounds with high molecular mass and many polar atoms are more likely to be substrates
            
            **Model Performance**:
            - Dataset size: 332 compounds (126 substrates, 206 non-substrates)
            - Task: Binary classification
            
            **Training Data**: Wang et al.
            """)
        
        # Metabolism Properties
        st.markdown("---")
        st.header("🟢 Metabolism Properties", anchor="metabolism-properties")
        
        st.markdown("""
        Metabolism properties predict how drugs interact with cytochrome P450 (CYP) enzymes, which are responsible 
        for metabolizing ~75% of all drugs. Understanding these interactions is crucial for:
        - Predicting drug clearance rates
        - Identifying potential drug-drug interactions
        - Optimizing dosing regimens
        """)
        
        # CYP Overview
        st.info("""
        **CYP Enzyme Overview**:
        - **CYP1A2**: Metabolizes ~5% of drugs (caffeine, theophylline)
        - **CYP2C9**: Metabolizes ~10% of drugs (warfarin, NSAIDs)
        - **CYP2C19**: Metabolizes ~5% of drugs (proton pump inhibitors, antidepressants)
        - **CYP2D6**: Metabolizes ~25% of drugs (antidepressants, antipsychotics, opioids)
        - **CYP3A4**: Metabolizes ~50% of drugs (statins, benzodiazepines, immunosuppressants)
        """)
        
        # CYP1A2
        with st.expander("**CYP1A2-Inhibitor** & **CYP1A2-Substrate**"):
            st.markdown("""
            **Inhibitor Description**: Probability of inhibiting CYP1A2 enzyme.
            
            **Interpretation**:
            - **Inhibitor = 1**: Likely inhibits CYP1A2 → may slow metabolism of co-administered CYP1A2 substrates
            - **Inhibitor = 0**: Unlikely to inhibit
            - **Substrate = 1**: Metabolized by CYP1A2 → metabolism affected by CYP1A2 inhibitors
            - **Substrate = 0**: Not metabolized by this enzyme
            
            **Model Performance (Inhibitor)**:
            - Dataset size: 13,239 compounds (5,997 inhibitors, 7,242 non-inhibitors)
            - Specificity: 0.873 | Sensitivity: 0.866 | Accuracy: 0.87 | Balanced Acc: 0.869
            - Source: PubChem AID 1851
            """)
        
        # CYP2C9
        with st.expander("**CYP2C9-Inhibitor** & **CYP2C9-Substrate**"):
            st.markdown("""
            **Inhibitor Description**: Probability of inhibiting CYP2C9 enzyme, which metabolizes warfarin 
            and many NSAIDs.
            
            **Interpretation**:
            - **Inhibitor = 1**: May cause increased exposure to drugs like warfarin (bleeding risk)
            - **Substrate = 1**: Metabolism affected by CYP2C9 inhibitors
            
            **Model Performance (Inhibitor)**:
            - Dataset size: 12,881 compounds
            - Specificity: 0.830 | Sensitivity: 0.819 | Accuracy: 0.826 | Balanced Acc: 0.824
            - Task: Binary classification (AUPRC metric)
            - Source: PubChem
            
            **Model Performance (Substrate)**:
            - Dataset size: 899 compounds
            - Specificity: 0.728 | Sensitivity: 0.757 | Accuracy: 0.738 | Balanced Acc: 0.742
            """)
        
        # CYP2C19
        with st.expander("**CYP2C19-Inhibitor** & **CYP2C19-Substrate**"):
            st.markdown("""
            **Inhibitor Description**: Probability of inhibiting CYP2C19, which metabolizes proton pump 
            inhibitors and some antidepressants.
            
            **Interpretation**:
            - **Inhibitor = 1**: May affect metabolism of PPIs and clopidogrel
            - **Substrate = 1**: Metabolism varies with CYP2C19 genotype (poor vs. rapid metabolizers)
            
            **Model Performance (Inhibitor)**:
            - Dataset size: 13,427 compounds
            - Specificity: 0.819 | Sensitivity: 0.830 | Accuracy: 0.824 | Balanced Acc: 0.825
            - Source: PubChem
            """)
        
        # CYP2D6
        with st.expander("**CYP2D6-Inhibitor** & **CYP2D6-Substrate**"):
            st.markdown("""
            **Inhibitor Description**: Probability of inhibiting CYP2D6, one of the most important drug-metabolizing 
            enzymes with high genetic polymorphism.
            
            **Interpretation**:
            - **Inhibitor = 1**: Major DDI concern; affects ~25% of drugs
            - **Substrate = 1**: Highly variable metabolism in population due to genetic variants
            
            **Model Performance (Inhibitor)**:
            - Dataset size: 11,127 compounds
            - Specificity: 0.866 | Sensitivity: 0.751 | Accuracy: 0.843 | Balanced Acc: 0.808
            - Source: PubChem
            
            **Model Performance (Substrate)**:
            - Dataset size: 941 compounds
            - Specificity: 0.749 | Sensitivity: 0.769 | Accuracy: 0.753 | Balanced Acc: 0.759
            """)
        
        # CYP3A4
        with st.expander("**CYP3A4-Inhibitor** & **CYP3A4-Substrate**"):
            st.markdown("""
            **Inhibitor Description**: Probability of inhibiting CYP3A4, the most abundant CYP enzyme that 
            metabolizes ~50% of all drugs.
            
            **Interpretation**:
            - **Inhibitor = 1**: High DDI risk; most clinically significant CYP enzyme
            - **Substrate = 1**: Metabolism affected by many common drugs (grapefruit juice, ketoconazole)
            
            **Clinical Significance**: 
            - Inhibition can lead to toxic drug levels (e.g., statins + azole antifungals)
            - Induction can lead to therapeutic failure
            
            **Model Performance (Inhibitor)**:
            - Dataset size: 12,997 compounds
            - Specificity: 0.815 | Sensitivity: 0.842 | Accuracy: 0.826 | Balanced Acc: 0.829
            - Task: Binary classification (AUPRC metric)
            - Enhanced with Novartis data for improved performance
            
            **Model Performance (Substrate)**:
            - Dataset size: 1,149 compounds
            - Specificity: 0.569 | Sensitivity: 0.779 | Accuracy: 0.718 | Balanced Acc: 0.674
            """)
        
        # Toxicity Properties
        st.markdown("---")
        st.header("🔴 Toxicity Properties", anchor="toxicity-properties")
        
        st.markdown("""
        Toxicity predictions help identify compounds with potential safety concerns early in drug development.
        """)
        
        # hERG
        with st.expander("**hERG** - Cardiac Toxicity"):
            st.markdown("""
            **Description**: Probability of blocking the hERG (human Ether-à-go-go-Related Gene) potassium channel, 
            which can cause QT interval prolongation and potentially fatal cardiac arrhythmias (Torsades de Pointes).
            
            **Output**: Probability (0-1) of being a hERG blocker
            
            **Interpretation**:
            - **High** (> 0.7): Likely hERG blocker → **HIGH CARDIAC RISK**
            - **Moderate** (0.3-0.7): Uncertain, requires experimental validation
            - **Low** (< 0.3): Unlikely to cause cardiac toxicity via hERG
            
            **Clinical Significance**:
            - hERG inhibition is a major cause of drug withdrawal from market
            - Critical liability in drug development
            - False positives are acceptable; false negatives are dangerous
            
            **Model Performance**:
            - Dataset size: 22,249 compounds
            - Specificity: 0.811 | Sensitivity: 0.897 | Accuracy: 0.885 | Balanced Acc: 0.854
            - Task: Binary classification
            
            **Training Data**: Curated from multiple sources including ChEMBL and PubChem
            """)
        
        # LD50
        with st.expander("**LD50** - Acute Toxicity"):
            st.markdown("""
            **Description**: LD50 (Lethal Dose 50%) is the amount of substance that kills 50% of test animals. 
            Predicts acute oral toxicity in rats.
            
            **Unit**: log(1/(mol/kg)) - higher values = more toxic
            
            **Interpretation**:
            - **High LD50** (> 3.0): Highly toxic
            - **Moderate** (2.0-3.0): Moderately toxic
            - **Low** (< 2.0): Relatively safe
            - Negative values indicate low toxicity (large doses needed for lethality)
            
            **Toxicity Classes** (GHS):
            - **Category 1**: ≤ 5 mg/kg (fatal if swallowed)
            - **Category 2**: 5-50 mg/kg (fatal)
            - **Category 3**: 50-300 mg/kg (toxic)
            - **Category 4**: 300-2000 mg/kg (harmful)
            - **Category 5**: 2000-5000 mg/kg (may be harmful)
            
            **Model Performance**:
            - Dataset size: 7,282 compounds
            - MAE: 0.437 | RMSE: 0.609 | R²: 0.596 | Spearman: 0.745
            - Task: Regression
            
            **Training Data**: Zhu et al., from experimental rat studies
            """)
        
        # Model Performance & Evaluation
        st.markdown("---")
        st.header("📊 Model Performance & Evaluation", anchor="model-performance-evaluation")
        
        st.markdown("""
        All Admetica models have been rigorously evaluated and compared against existing tools.
        """)
        
        with st.expander("**Understanding Performance Metrics**"):
            st.markdown("""
            **For Classification Models**:
            - **Specificity**: True negative rate (correctly identifying non-active compounds)
            - **Sensitivity (Recall)**: True positive rate (correctly identifying active compounds)
            - **Accuracy**: Overall correct predictions
            - **Balanced Accuracy**: Average of sensitivity and specificity (better for imbalanced datasets)
            - **ROC AUC**: Area under receiver operating characteristic curve (0.5 = random, 1.0 = perfect)
            
            **For Regression Models**:
            - **MAE** (Mean Absolute Error): Average prediction error
            - **RMSE** (Root Mean Square Error): Penalizes large errors more
            - **R²** (R-squared): Proportion of variance explained (0 = poor, 1 = perfect)
            - **Spearman**: Rank correlation (measures monotonic relationship)
            
            **Interpretation Guidelines**:
            - **Excellent**: R² > 0.8, Accuracy > 0.9, ROC AUC > 0.9
            - **Good**: R² > 0.6, Accuracy > 0.8, ROC AUC > 0.8
            - **Acceptable**: R² > 0.4, Accuracy > 0.7, ROC AUC > 0.7
            """)
        
        with st.expander("**Comparison with Other Tools**"):
            st.markdown("""
            Admetica has been benchmarked against commercial and free tools including:
            - **ADMETLab 3.0**
            - **admetSAR**
            - **SwissADME**
            - **pkCSM**
            - **preADMET**
            - **Novartis models** (published in Nature 2024)
            
            **Key Findings**:
            - Admetica models are **competitive** with commercial tools
            - Some models **enhanced** using Novartis surrogate data:
              * CYP3A4-Inhibitor: +21.6% sensitivity improvement
              * Caco-2: -31.9% MSE reduction (better predictions)
              * CYP2C9-Inhibitor: +69.3% specificity improvement
            - **Advantages**: Open-source, reproducible, customizable, free
            """)
        
        with st.expander("**Model Enhancement & Training**"):
            st.markdown("""
            **Training Methodology**:
            - **Architecture**: Chemprop (Message Passing Neural Networks)
            - **Features**: Molecular graphs (atoms and bonds)
            - **No hand-crafted descriptors** required
            - **Cross-validation**: Rigorous splitting strategies to prevent overfitting
            
            **Data Sources**:
            - PubChem Bioassays
            - ChEMBL database
            - Academic publications (peer-reviewed)
            - Some models enhanced with Novartis predictions (Nature 2024)
            
            **Continuous Improvement**:
            - Models regularly updated with new data
            - Community contributions welcome (MIT license)
            - Benchmarking against latest literature
            """)
        
        # References
        st.markdown("---")
        st.header("📚 References", anchor="references")
        
        st.markdown("""
        **Primary Source**:
        - Admetica GitHub Repository: [datagrok-ai/admetica](https://github.com/datagrok-ai/admetica)
        - License: MIT (open-source)
        
        **Key Publications**:
        1. **Novartis Comparison**: Nature Communications (2024) - "ADMET predictions from deep learning"
        2. **Evaluation of ADMET Tools**: Molecules (2023) - "Evaluation of Free Online ADMET Tools"
        3. **Chemprop**: J. Chem. Inf. Model. (2019) - "Analyzing Learned Molecular Representations for Property Prediction"
        
        **Dataset References**:
        - **PubChem**: AID 1851 (CYP enzymes), various bioassays
        - **ChEMBL**: Curated bioactivity database (EMBL-EBI)
        - **TDC** (Therapeutics Data Commons): Benchmark datasets
        - **Academic**: Hou et al., Wang et al., Zhu et al., Sorkun et al.
        
        **Model Files & Code**:
        - Pre-trained models: Available in Admetica repository
        - Training scripts: Jupyter notebooks included
        - CLI tool: `admetica_predict` (PyPI package)
        
        **Related Tools**:
        - Datagrok Admetica Plugin: [GitHub](https://github.com/datagrok-ai/public/tree/master/packages/Admetica)
        - Integration examples and tutorials in repository
        """)
        
        # Footer
        st.markdown("---")
        st.info("""
        **Note**: These predictions are computational estimates based on machine learning models. 
        While they are useful for prioritization and hypothesis generation, experimental validation 
        is always required for regulatory submissions and clinical development.
        
        **Disclaimer**: Admetica predictions should not be used as the sole basis for safety or efficacy 
        decisions. Always consult with qualified professionals and conduct appropriate experimental validation.
        """)
        
        st.success("""
        💡 **For more information**: Visit the [Admetica GitHub repository](https://github.com/datagrok-ai/admetica) 
        for detailed documentation, training notebooks, and the latest model updates.
        """)
        
        # Back button at the bottom
        st.markdown("---")
        if st.button("← Back to Main", type="secondary", use_container_width=True):
            st.session_state.current_page = "DataFrame Wizard"
            st.rerun()
