"""
SAR Application - Main Entry Point
Object-oriented Streamlit app for Structure-Activity Relationship analysis.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from sar_app.config import AppConfig
from sar_app.sidebar import SidebarComponent
from sar_app.pages import (
    DataFrameAnalyzer,
    ScaffoldAnalyzer,
    SMILESAnalyzer,
    ClusteringAnalyzer,
    KetcherAnalyzer,
    ADMETDocsAnalyzer,
    MMPAnalyzer
)
from sar_app.utils import MemoryUtils


class SARApplication:
    """Main application orchestrator."""
    
    def __init__(self):
        self.config = AppConfig()
        self.sidebar = SidebarComponent(self.config)
        self._configure_page()
        self._initialize_session()
    
    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout=self.config.LAYOUT,
            initial_sidebar_state=self.config.SIDEBAR_STATE
        )
    
    def _initialize_session(self):
        """Initialize session state."""
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "DataFrame Wizard"
    
    def run(self):
        """Run the application."""
        # Header
        self._render_header()
        
        # Sidebar
        self.sidebar.render()
        
        # Check if we're on a sidebar-only page (like Chemical Sketcher or ADMET Documentation)
        if st.session_state.current_page in ["Chemical Sketcher", "ADMET Documentation"]:
            # Render sidebar-only pages directly without showing in menu
            self._render_page(st.session_state.current_page)
        else:
            # Navigation menu for main pages
            selected = self._render_menu()
            
            # Render selected page
            self._render_page(selected)
        
        # Footer with memory usage
        self._render_footer()
    
    def _render_header(self):
        """Render application header."""
        st.markdown(
            f"<h1 style='text-align: center;'>{self.config.PAGE_TITLE}</h1>",
            unsafe_allow_html=True
        )
        st.markdown("---")
    
    def _render_menu(self) -> str:
        """Render navigation menu."""
        # Find the index of the current page for proper menu highlighting
        try:
            current_index = self.config.MENU_OPTIONS.index(st.session_state.current_page)
        except (ValueError, KeyError):
            current_index = 0  # Default to DataFrame Wizard
        
        selected = option_menu(
            menu_title=None,
            options=self.config.MENU_OPTIONS,
            icons=self.config.MENU_ICONS,
            menu_icon="cast",
            default_index=current_index,
            orientation="horizontal",
            styles={
            "container": {"padding": "0!important", "background-color": "#f8fae1"},
            "icon": {"color": "red", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "color": "#000000",  # Font color for unselected items
                "--hover-color": "#eee",
            },
            "nav-link-selected": {
                "background-color": "darkgreen",
                "color": "#ffffff"  # Font color for selected item
            },
            }
        )
        
        st.session_state.current_page = selected
        return selected
    
    def _render_page(self, selected: str):
        """Render the selected page."""
        try:
            analyzer = self._create_analyzer(selected)
            if analyzer:
                analyzer.render()
                analyzer.cleanup()
        except Exception as e:
            st.error(f"Error rendering page: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    def _create_analyzer(self, page: str):
        """Factory method to create analyzer instance."""
        analyzers = {
            "Chemical Sketcher": KetcherAnalyzer,
            "DataFrame Wizard": DataFrameAnalyzer,
            "Scaffold Hunter": ScaffoldAnalyzer,
            "SMILES Analysis": SMILESAnalyzer,
            "Butina Clustering": ClusteringAnalyzer,
            "MMP Analysis": MMPAnalyzer,
            "ADMET Documentation": ADMETDocsAnalyzer
        }
        
        analyzer_class = analyzers.get(page)
        if analyzer_class:
            return analyzer_class(self.config)
        
        st.warning(f"Page '{page}' not implemented")
        return None
    
    def _render_footer(self):
        """Render footer with memory usage."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(
                "<small>Built with Streamlit & RDKit | By Ganesh Shahane</small>",
                unsafe_allow_html=True
            )
        
        with col2:
            mem_usage = MemoryUtils.get_memory_usage()
            st.markdown(
                f"<small>💾 Memory: {mem_usage['used_gb']:.2f} / {mem_usage['total_gb']:.2f} GB</small>",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"<small>📄 Page: {st.session_state.current_page}</small>",
                unsafe_allow_html=True
            )


def main():
    """Application entry point."""
    app = SARApplication()
    app.run()


if __name__ == "__main__":
    main()
