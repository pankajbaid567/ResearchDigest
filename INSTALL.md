# Installation & Setup Guide

## Quick Setup (Anaconda Users - RECOMMENDED)

Since Gensim has compilation issues with Python 3.12+, use conda for pre-built binaries:

```bash
# Install Gensim via conda (pre-built, no compilation needed)
conda install -y -c conda-forge gensim

# Install other dependencies
pip install streamlit nltk scikit-learn matplotlib wordcloud pandas PyPDF2 pdfplumber

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"

# Run the application
streamlit run ui/streamlit_app.py
```

## Alternative: Virtual Environment with Python 3.10-3.11

If you don't have conda or prefer venv:

```bash
# Install Python 3.11 (if not already installed)
brew install python@3.11

# Create venv with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"

# Run the application
streamlit run ui/streamlit_app.py
```

## Troubleshooting

### Issue: Gensim compilation fails
**Solution**: Use conda installation (see above) or Python 3.10-3.11

### Issue: NLTK data not found
**Solution**: Run the NLTK download commands above

### Issue: Streamlit won't start
**Solution**: Check if port 8501 is available:
```bash
streamlit run ui/streamlit_app.py --server.port 8502
```

## System Requirements

- **Python**: 3.10, 3.11, or 3.12 (with conda for Gensim)
- **OS**: macOS, Linux, or Windows
- **RAM**: 4GB minimum (8GB recommended)
- **Disk**: 500MB free space

## Verifying Installation

Test that everything works:

```bash
python -c "import nltk, sklearn, streamlit, matplotlib, gensim; print('✅ All packages installed!')"
```

Expected output: `✅ All packages installed!`
