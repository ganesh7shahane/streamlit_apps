# SAR Application - OOP Refactor

A complete object-oriented rewrite of the Streamlit SAR (Structure-Activity Relationship) analysis application.

## 🏗️ Architecture

### Package Structure
```
sar_app/
├── __init__.py          # Package initialization
├── config.py            # Immutable configuration (frozen dataclass)
├── utils.py             # Utility classes (Molecule, DataFrame, Memory, Plot)
├── base.py              # Abstract base class for analyzers
├── sidebar.py           # Ketcher sidebar component
└── pages/
    ├── __init__.py      # Page exports
    ├── dataframe.py     # DataFrame visualization & analysis
    ├── scaffold.py      # Scaffold identification & analysis
    ├── smiles.py        # Individual SMILES analysis
    └── clustering.py    # Taylor-Butina clustering
```

### Design Principles

**1. Object-Oriented Programming**
- Abstract base classes define common interfaces
- Inheritance for shared functionality
- Composition for component reusability
- Factory pattern for analyzer creation

**2. Clean Code**
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- Meaningful names and clear documentation
- Small, focused methods

**3. Memory Efficiency**
- Streamlit caching with `@st.cache_data`
- Dtype optimization for DataFrames
- Batch processing for large datasets
- Memory monitoring with psutil

**4. Immutability**
- Frozen dataclass for configuration
- Prevents accidental state mutation
- Thread-safe by design

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Usage
1. Upload a CSV file with molecular data
2. Select SMILES column
3. Choose an analysis page from the menu:
   - **DataFrame Viz**: Data exploration, statistics, visualizations
   - **Analyse Scaffolds**: Murcko scaffold identification
   - **SMILES Analysis**: Individual molecule property calculation
   - **Taylor-Butina Clustering**: Structural clustering

## 📦 Components

### Core Classes

**AppConfig** (`config.py`)
- Frozen dataclass with all application settings
- Immutable configuration prevents runtime modification
- Single source of truth for constants

**BaseAnalyzer** (`base.py`)
- Abstract base class for all page analyzers
- Defines common interface: `render()`, `cleanup()`, `_load_data()`
- Enforces consistent page structure

**Utility Classes** (`utils.py`)
- `MoleculeUtils`: SMILES/molecule conversions, image generation
- `DataFrameUtils`: Data cleaning, optimization, statistics
- `MemoryUtils`: Memory monitoring and reporting
- `PlotUtils`: Consistent styling for visualizations

**SidebarComponent** (`sidebar.py`)
- Ketcher molecule sketcher integration
- Manages molecule drawing and session state
- Calculates molecular properties

### Page Analyzers

**DataFrameAnalyzer** (`pages/dataframe.py`)
- Data cleaning and preprocessing
- Statistical summaries
- Multiple visualization types (histograms, scatter, heatmaps)
- Molecular grid display
- Descriptor calculation

**ScaffoldAnalyzer** (`pages/scaffold.py`)
- Murcko scaffold identification
- Scaffold counting and distribution
- Activity analysis per scaffold
- Interactive molecular grids

**SMILESAnalyzer** (`pages/smiles.py`)
- Individual molecule analysis
- 2D structure display
- Property calculation (MW, LogP, TPSA, etc.)
- Drug-likeness assessment (Lipinski, QED, SA Score)
- BRICS fragmentation
- Fingerprint generation

**ClusteringAnalyzer** (`pages/clustering.py`)
- Taylor-Butina clustering algorithm
- Configurable fingerprints (Morgan, RDKit, MACCS)
- Distance matrix calculation
- Cluster visualization and statistics
- Representative molecule selection

## 🔧 Key Features

### Caching Strategy
- Expensive operations cached with `@st.cache_data`
- Molecular conversions, descriptor calculations cached
- Memory-efficient DataFrame operations

### Error Handling
- Try-except blocks for RDKit operations
- Graceful fallbacks for invalid molecules
- User-friendly error messages

### Type Safety
- Type hints throughout codebase
- Dataclass validation
- Runtime type checking where appropriate

### Session State Management
- Centralized state in `st.session_state`
- Sidebar molecule shared across pages
- Current page tracking

## 📊 Data Requirements

### Input Format
CSV file with:
- SMILES column (required)
- Optional: Activity values (pchembl_value, IC50, etc.)
- Optional: Molecular descriptors (MW, LogP, etc.)

### Example
```csv
SMILES,pchembl_value,MW
CCO,5.2,46.07
c1ccccc1,4.8,78.11
```

## 🎨 Customization

### Styling
Modify `config.py`:
```python
@dataclass(frozen=True)
class AppConfig:
    PAGE_TITLE: str = "Your App Name"
    PAGE_ICON: str = "🧪"
    COLORS: Tuple[str, ...] = ("#FF6B6B", "#4ECDC4", ...)
```

### Adding Pages
1. Create new analyzer in `sar_app/pages/`
2. Inherit from `BaseAnalyzer`
3. Implement `render()` method
4. Add to `pages/__init__.py`
5. Register in `app.py` factory

## 🐛 Troubleshooting

### Memory Issues
- Reduce dataset size
- Use data sampling for previews
- Check memory usage in footer

### RDKit Errors
- Validate SMILES with `MoleculeUtils.smiles_to_mol()`
- Check for None returns
- Use try-except for RDKit operations

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify RDKit installation

## 📝 Development

### Running Tests
```bash
# Validate setup
python -c "from sar_app import AppConfig; print('OK')"

# Check imports
python -c "from sar_app.pages import *; print('OK')"
```

### Code Style
- Follow PEP 8
- Use type hints
- Document with docstrings
- Keep methods under 50 lines

## 📄 License

Same as original streamlit_basic.py

## 🙏 Credits

Built with:
- Streamlit
- RDKit
- Pandas/NumPy
- Matplotlib/Seaborn/Plotly
