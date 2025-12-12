# Documentation Index

This folder contains all documentation for the STRATHEX Woodchopping Handicap System.

## 📚 Documentation Files

### Getting Started
- **ReadMe.md** - Main project documentation and function reference
- **HANDICAP_SYSTEM_EXPLAINED.md** - Comprehensive user guide (judges/competitors)

### Development Documentation
- **REFACTORING_COMPLETE.md** - Complete refactoring summary and new architecture
- **NewFeatures.md** - Planned features and enhancement roadmap

## 🔍 Quick Navigation

### For Users/Judges
Start with HANDICAP_SYSTEM_EXPLAINED.md to understand:
- How the system works
- What each prediction method does (Manual/LLM/ML)
- Statistical terms explained
- Technical deep dive

### For Developers
Read in this order:
1. ReadMe.md - Understand the codebase structure
2. REFACTORING_COMPLETE.md - New modular architecture
3. NewFeatures.md - Future enhancements

## 📁 Project Structure

```
woodchopping-handicap-system/
├── docs/                    # 📄 Documentation (you are here)
├── scripts/                 # 🔧 Utility scripts
├── archive/                 # 📦 Old/backup files
├── woodchopping/           # 📦 Main package
│   ├── predictions/        # Prediction algorithms
│   ├── handicaps/          # Handicap calculation
│   ├── simulation/         # Monte Carlo simulation
│   └── ui/                 # User interface
├── MainProgramV3.1.py      # 🎯 Main entry point
├── FunctionsLibrary.py     # 📚 Function library
├── config.py               # ⚙️ Configuration
├── explanation_system_functions.py  # 💡 Help system
├── woodchopping.xlsx       # 💾 Data persistence
└── CLAUDE.md               # 🤖 Claude Code instructions
```

## 🚀 Quick Start

- **Run the program**: `python MainProgramV3.1.py`
- **View help system**: Menu option 14 in the main program
- **Configuration**: See `../config.py` for all system parameters
