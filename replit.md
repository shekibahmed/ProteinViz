# Overview

This is a Streamlit-based protein interaction research dashboard that enables scientists to query, visualize, and predict protein interactions. The application focuses on both direct binding partners and indirect effects, with machine learning-powered prediction capabilities. It's designed specifically for researchers studying protein interactions like EGCG compounds, with support for multiple species and disease contexts.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses **Streamlit** as the primary web framework, providing an interactive dashboard interface. The main application (`main.py`) follows a single-page application pattern with sidebar controls for user inputs and a main content area for data visualization and results.

Key UI components include:
- **Sidebar controls** for query parameters (protein ID, species selection, disease contexts)
- **Cached data loading** using `@st.cache_data` decorators to optimize performance
- **Interactive visualizations** using Plotly for charts and graphs
- **Expandable sections** and tabs for organizing complex information

## Backend Architecture
The backend follows a **modular utility-based architecture** with separate modules for different concerns:

- **Data Layer** (`utils/data_loader.py`): Handles CSV data loading with error handling and caching
- **ML Models** (`utils/ml_models.py`): Implements machine learning prediction capabilities using scikit-learn
- **Protein Utilities** (`utils/protein_utils.py`): Manages protein sequence retrieval and feature extraction

## Data Storage
The application uses **CSV-based data storage** for protein interaction datasets:
- `EGCG_Direct_Interactions.csv` - Direct protein binding interactions
- `EGCG_Indirect_Effects.csv` - Indirect protein effects

Data is loaded into Pandas DataFrames with fallback mechanisms for missing files, creating empty DataFrames with expected column structures.

## Machine Learning Architecture
The system implements a **dual-model approach** for protein interaction prediction:
- **Random Forest Classifier** - Ensemble method for robust predictions
- **Support Vector Machine (SVM)** - RBF kernel with probability estimates

Models use **caching mechanisms** (`@st.cache_resource`) to avoid retraining and include persistence through joblib serialization.

## Error Handling and Resilience
The architecture emphasizes **graceful degradation** with comprehensive error handling:
- Fallback empty DataFrames when CSV files are missing
- Mock sequence generation when external APIs fail
- Try-catch blocks around external service calls
- User-friendly error messages through Streamlit's error display

# External Dependencies

## Core Python Libraries
- **Streamlit** - Web application framework and UI components
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Plotly** - Interactive data visualization and charting

## Machine Learning Stack
- **Scikit-learn** - Machine learning algorithms (Random Forest, SVM)
- **Joblib** - Model persistence and caching

## Advanced ML Libraries (Planned)
- **PyTorch** - Deep learning framework for advanced models
- **DGL/PyTorch Geometric** - Graph neural networks for protein interactions
- **Transformers** - Pre-trained models for protein analysis

## External APIs
- **UniProt API** - Protein sequence retrieval service
  - Primary endpoint: `https://www.uniprot.org/uniprot/{protein_id}.fasta`
  - Search endpoint: `https://www.uniprot.org/uniprot/?query={protein_id}&format=fasta`
  - Includes fallback mechanisms and timeout handling

## File System Dependencies
- **CSV Data Files** - Stored in `/data/` directory
- **Model Persistence** - Saved in `/models/` directory
- **Requirements Management** - Standard `requirements.txt` for dependency management

## Development Infrastructure
- **Replit Environment** - Cloud-based development and hosting platform
- **HTTP Requests** - For external API communication with timeout and error handling