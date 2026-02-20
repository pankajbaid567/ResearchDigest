## ⚠️ Gensim Installation Issues (Python 3.12)

Gensim has known compatibility issues with Python 3.12. Here are solutions:

### **Solution 1: Run Without LDA (K-Means Only) ✅ WORKS NOW**

The application can run with K-Means clustering instead of LDA:

```bash
# All dependencies except Gensim are installed
streamlit run ui/streamlit_app.py
```

In the UI, select **"K-Means Clustering"** instead of "LDA (Gensim)" for topic extraction.

### **Solution 2: Fix Gensim (Try This)**

```bash
# Downgrade SciPy to compatible version
pip install --upgrade 'scipy<1.15' 'gensim>=4.3'

# Then test
python -c "import gensim; print('Gensim works!')"
```

### **Solution 3: Use Python 3.10/3.11 (Most Reliable)**

```bash
# Install Python 3.11
brew install python@3.11

# Create new venv
python3.11 -m venv venv311
source venv311/bin/activate

# Install everything
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('omw-1.4')"

# Run app
streamlit run ui/streamlit_app.py
```

### **What Works Right Now**

✅ Document loading (PDF + text)  
✅ Text preprocessing  
✅ TF-IDF feature extraction  
✅ **K-Means clustering** (alternative to LDA)  
✅ Keyword extraction  
✅ Extractive summarization  
✅ Evaluation metrics  
✅ Streamlit UI  

❌ LDA topic modeling (requires Gensim fix)

The system is **fully functional** using K-Means clustering!
