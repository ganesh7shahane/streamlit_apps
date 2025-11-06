"""
Sidebar component with links and information.
"""

import streamlit as st
from sar_app.config import AppConfig


class SidebarComponent:
    """Manages the sidebar with links and information."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize sidebar component.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    def render(self):
        """Render the sidebar with links and information."""
        with st.sidebar:
            st.title("Additional Tools & Docs")
            
            # Chemical Sketcher Link
            st.markdown("### ✏️ Draw Molecules interactively")
            if st.button("🖌️ Open Chemical Sketcher", use_container_width=True, type="primary"):
                st.session_state.current_page = "Chemical Sketcher"
                st.rerun()
            
            # Quick Links
            st.markdown("---")
            st.markdown("### 📚 Quick Links")
            
            # ADMET Documentation Link
            if st.button("📖 ADMET Properties Guide", use_container_width=True):
                st.session_state.current_page = "ADMET Documentation"
                st.rerun()
            
            st.markdown("- [Ketcher Documentation](https://github.com/epam/ketcher)")
            st.markdown("- [RDKit Documentation](https://www.rdkit.org/docs/)")
            st.markdown("- [RDKit Blog](https://greglandrum.github.io/rdkit-blog/)")
            st.markdown("- [Streamlit Docs](https://docs.streamlit.io)")
            
            # Developer Info
            st.markdown("---")
            st.markdown("### �‍💻 Developer")
            st.markdown("**Ganesh Shahane**")
            st.markdown("- [GitHub](https://github.com/ganesh7shahane)")
            st.markdown("- [LinkedIn](https://www.linkedin.com/in/ganesh7shahane)")
            
            # Tips
            st.markdown("---")
            st.markdown("### 💡 Quick Tips")
            st.info("""
            - Draw molecules in **Chemical Sketcher**
            - Analyze individual molecules in **SMILES Analysis**
            - Visualize datasets in **DataFrame Wiz**
            - Find scaffolds in **Analyse Scaffolds**
            - Cluster molecules in **Taylor-Butina Clustering**
            """)
