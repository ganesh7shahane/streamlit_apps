# Quick Start Guide - OOP SAR Application

## ✅ What's Been Done

Your Streamlit app has been completely refactored using **object-oriented programming** and **clean code principles**:

### 📁 New Structure
```
sar_app/                    # Main package
├── __init__.py            # Package initialization  
├── config.py              # Frozen dataclass configuration
├── base.py                # Abstract base class
├── utils.py               # 4 utility classes
├── sidebar.py             # Ketcher component
└── pages/                 # Analyzer pages
    ├── __init__.py
    ├── dataframe.py       # DataFrame analysis
    ├── scaffold.py        # Scaffold identification
    ├── smiles.py          # Individual SMILES analysis
    └── clustering.py      # Taylor-Butina clustering

app.py                     # Main entry point
requirements.txt           # Dependencies
README_OOP.md             # Full documentation
```

### 🎯 Key Features

**OOP Principles Applied:**
- ✅ Abstract base classes (`BaseAnalyzer`)
- ✅ Inheritance (all pages inherit from base)
- ✅ Composition (utilities as separate classes)
- ✅ Encapsulation (frozen config, private methods)
- ✅ Factory pattern (analyzer creation)

**Clean Code:**
- ✅ Single Responsibility Principle
- ✅ DRY (no code duplication)
- ✅ Clear naming conventions
- ✅ Type hints throughout
- ✅ Comprehensive docstrings

**Memory Efficiency:**
- ✅ `@st.cache_data` for expensive operations
- ✅ Dtype optimization
- ✅ Memory monitoring with psutil
- ✅ Batch processing

## 🚀 Running the App

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

### Step 3: Use the App
1. Upload CSV with molecular data
2. Select SMILES column
3. Navigate between pages:
   - **DataFrame Viz**: Data exploration
   - **Analyse Scaffolds**: Murcko scaffolds
   - **SMILES Analysis**: Individual molecules
   - **Taylor-Butina Clustering**: Structural clustering

## 🔍 What Changed

### Before (streamlit_basic.py)
- ❌ 1223 lines in single file
- ❌ Procedural programming
- ❌ Code duplication
- ❌ Mixed responsibilities
- ❌ Hard to test & maintain

### After (sar_app/)
- ✅ Modular package structure
- ✅ Object-oriented design
- ✅ Reusable components
- ✅ Clear separation of concerns
- ✅ Easy to extend & test

## 📦 Core Components

### SARApplication (app.py)
Main orchestrator that:
- Configures page settings
- Renders navigation menu
- Routes to analyzers
- Displays memory usage

### AppConfig (config.py)
Immutable configuration:
```python
@dataclass(frozen=True)
class AppConfig:
    PAGE_TITLE = "Structure Activity Relationships"
    PAGES = ["DataFrame Viz", "Analyse Scaffolds", ...]
    COLORS = ("#3498db", "#e74c3c", ...)
```

### BaseAnalyzer (base.py)
Abstract base for all pages:
```python
class BaseAnalyzer(ABC):
    @abstractmethod
    def render(self):
        """Must be implemented by subclasses"""
        pass
```

### Utilities (utils.py)
Four utility classes:
- `MoleculeUtils`: SMILES/molecule operations
- `DataFrameUtils`: Data cleaning & stats
- `MemoryUtils`: Memory monitoring
- `PlotUtils`: Consistent styling

## 🧪 Testing

### Verify Installation
```bash
# Test imports
python -c "from sar_app import AppConfig; print('✅ Config OK')"
python -c "from sar_app.pages import *; print('✅ Pages OK')"

# Check structure
python -c "from sar_app import SARApplication; print('✅ App OK')"
```

### Run the App
```bash
streamlit run app.py
```

## 🎨 Customization

### Change Colors
Edit `sar_app/config.py`:
```python
COLORS: Tuple[str, ...] = (
    "#YOUR_COLOR_1",
    "#YOUR_COLOR_2",
    ...
)
```

### Add New Page
1. Create `sar_app/pages/mypage.py`
2. Inherit from `BaseAnalyzer`
3. Implement `render()` method
4. Add to `pages/__init__.py`
5. Register in `app.py` factory

## 📊 Memory Efficiency

The app monitors and optimizes memory:
- Footer shows real-time memory usage
- DataFrames use optimized dtypes
- Cached operations prevent recomputation
- Cleanup methods in analyzers

## 🐛 Known Issues

### sascorer Import
The SA Score calculation requires RDKit contrib. The app handles this gracefully:
- If available: calculates SA score
- If not: shows info message

### Large Datasets
For datasets >10k molecules:
- Use sampling for previews
- Enable batch processing
- Monitor memory in footer

## 📝 Next Steps

1. **Run the app**: `streamlit run app.py`
2. **Upload test data**: Use `data/chembl*.csv`
3. **Explore features**: Try all 4 pages
4. **Customize**: Modify config as needed

## 🎓 Learning Resources

- **README_OOP.md**: Full architecture documentation
- **Code comments**: Detailed inline documentation
- **Type hints**: Self-documenting code

## ✨ Benefits Over Original

1. **Maintainability**: Easy to find and fix bugs
2. **Extensibility**: Simple to add new features
3. **Reusability**: Components work independently
4. **Testability**: Each class can be tested in isolation
5. **Readability**: Clear structure and naming
6. **Performance**: Optimized caching and memory

---

**Original**: `streamlit_basic.py` (still available, untouched)  
**New OOP Version**: `app.py` + `sar_app/` package

Enjoy your refactored application! 🚀
