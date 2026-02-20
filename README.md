# Traditional NLP Research Topic Analysis System

**Milestone-1**: Classical NLP & ML Techniques for Intelligent Research Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Overview

This system demonstrates **classical Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques for analyzing research documents. It extracts topics, identifies keywords, and generates extractive summaries using only traditional methods—**no LLMs, no transformers, no agentic workflows**.

### Key Features

✅ **Topic Modeling**: LDA (Latent Dirichlet Allocation) using Gensim  
✅ **Clustering**: K-Means as alternative topic discovery method  
✅ **Keyword Extraction**: TF-IDF based term importance  
✅ **Extractive Summarization**: Sentence-level ranking algorithms  
✅ **Evaluation Metrics**: Topic coherence, diversity, and interpretability  
✅ **Interactive UI**: Streamlit web interface with visualizations  
✅ **Comprehensive Documentation**: Academic-grade explanations and limitations  

---

## 🎯 Project Objectives

This is **Milestone-1** of a two-part academic project:

- **Milestone-1** (This): Traditional NLP to establish baseline and understand limitations
- **Milestone-2** (Future): Agentic AI systems with LLMs, reasoning, and tool use

### What You'll Learn

- How classical NLP algorithms work (TF-IDF, LDA, clustering)
- Strengths and weaknesses of bag-of-words representations
- Proper evaluation of topic models
- Why modern methods (transformers, LLMs) are necessary

---

## 🏗️ System Architecture

```
Input (PDF/Text) → Preprocessing → TF-IDF → LDA/K-Means → Keywords → Summary → Evaluation
```

**Core Components**:
1. **Document Loader**: Extract text from PDFs and text files
2. **Preprocessor**: Tokenization, stop-word removal, lemmatization (NLTK/spaCy)
3. **Feature Extractor**: TF-IDF vectorization (scikit-learn)
4. **Topic Modeler**: LDA topic modeling (Gensim)
5. **Clusterer**: K-Means clustering (scikit-learn)
6. **Keyword Extractor**: TF-IDF based keyword ranking
7. **Summarizer**: Extractive summarization (multiple methods)
8. **Evaluator**: Topic coherence and diversity metrics
9. **Streamlit UI**: Interactive web interface

See [System Architecture](docs/03_system_architecture.md) for detailed diagrams.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 500MB free disk space

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd Project1-GenAI
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (required):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

5. **(Optional) Download spaCy model** for better lemmatization:
```bash
python -m spacy download en_core_web_sm
```

### Running the Application

```bash
streamlit run ui/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 📖 Usage Guide

### Input Methods

**Option 1: Upload Documents**
- Click "Browse files" and select PDF or text files
- Supports multiple files simultaneously
- Formats: `.txt`, `.pdf`, `.md`

**Option 2: Direct Text Input**
- Paste research text or keywords directly
- Useful for quick testing

### Configuration

Adjust parameters in the sidebar:
- **Number of Topics**: 2-10 (default: 5)
- **Keywords per Topic**: 5-20 (default: 10)
- **Summary Sentences**: 3-10 (default: 5)
- **Summarization Method**: TF-IDF, Frequency, Position-weighted
- **Analysis Method**: LDA or K-Means

### Running Analysis

1. Upload documents or enter text
2. Configure parameters (or use defaults)
3. Click **"Run Analysis"**
4. Wait for processing (10-30 seconds typical)
5. View results in tabs:
   - **Topics**: Discovered themes with keywords
   - **Keywords**: Overall top terms and word cloud
   - **Summary**: Extractive summary of documents
   - **Evaluation**: Coherence scores and metrics
   - **Visualizations**: Charts and graphs
   - **Limitations**: Understanding classical NLP constraints

---

## 📁 Project Structure

```
Project1-GenAI/
├── src/                          # Core NLP modules
│   ├── document_loader.py        # PDF/text loading
│   ├── preprocessor.py           # Tokenization, lemmatization
│   ├── feature_extractor.py     # TF-IDF, BoW
│   ├── topic_modeler.py         # LDA implementation
│   ├── clustering.py            # K-Means clustering
│   ├── keyword_extractor.py     # Keyword extraction
│   ├── summarizer.py            # Extractive summarization
│   ├── evaluator.py             # Metrics and evaluation
│   └── utils.py                 # Helper functions
├── ui/
│   └── streamlit_app.py         # Web interface
├── docs/                         # Documentation
│   ├── 01_problem_understanding.md
│   ├── 02_input_output_spec.md
│   ├── 03_system_architecture.md
│   ├── 04_pipeline_explanation.md
│   ├── 05_evaluation_results.md
│   └── 06_limitations_report.md
├── data/
│   └── sample_documents/        # Sample research papers (if provided)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # This file
```

---

## 📊 Technical Details

### NLP Pipeline

1. **Text Preprocessing**
   - Sentence tokenization: NLTK `sent_tokenize`
   - Word tokenization: NLTK `word_tokenize`
   - Stop-word removal: NLTK stopwords + custom academic terms
   - Lemmatization: spaCy (preferred) or NLTK WordNetLemmatizer

2. **Feature Extraction**
   - TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer`
   - Parameters: max_features=500, ngram_range=(1,2), sublinear_tf=True

3. **Topic Modeling**
   - **LDA**: `gensim.models.LdaModel`
     - Hyperparameters: alpha='auto', eta='auto'
     - Passes: 15, Iterations: 100
   - **K-Means**: `sklearn.cluster.KMeans`
     - Optimal K: Elbow method + silhouette score

4. **Summarization**
   - TF-IDF sentence scoring
   - Frequency-based ranking
   - Position-weighted scoring

5. **Evaluation**
   - Coherence: c_v metric (Gensim `CoherenceModel`)
   - Diversity: Unique terms / Total terms
   - Clustering: Silhouette score, Davies-Bouldin index

### Dependencies

Core libraries:
- **NLTK** 3.8+: Tokenization, stop words, lemmatization
- **spaCy** 3.7+: Advanced lemmatization (optional)
- **scikit-learn** 1.3+: TF-IDF, K-Means, metrics
- **Gensim** 4.3+: LDA topic modeling
- **Streamlit** 1.28+: Web interface
- **matplotlib** & **wordcloud**: Visualizations

See `requirements.txt` for full list.

---

## 📈 Evaluation Metrics

### Topic Coherence (c_v)

Measures semantic coherence of topic keywords:
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 0.6+: Excellent
  - 0.5-0.6: Good
  - 0.4-0.5: Moderate
  - < 0.4: Poor (needs tuning)

**Important**: High coherence is necessary but not sufficient for meaningful topics!

### Topic Diversity

Measures uniqueness of topics:
- **Formula**: Unique keywords / Total keywords
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Higher diversity = less redundant topics

### Clustering Metrics (if using K-Means)

- **Silhouette Score**: -1 to 1 (higher is better)
- **Davies-Bouldin Score**: 0+ (lower is better)

---

## ⚠️ Known Limitations

This system intentionally demonstrates the **constraints of classical NLP**:

1. **No Semantic Understanding**: Cannot recognize synonyms ("car" ≠ "automobile")
2. **Context Blindness**: No word sense disambiguation ("bank" = financial or river?)
3. **Statistical Coherence ≠ Meaning**: High coherence doesn't guarantee interpretable topics
4. **Preprocessing Sensitivity**: Results depend heavily on tokenization choices
5. **Extractive-Only Summarization**: Cannot paraphrase or synthesize
6. **No Multi-Document Reasoning**: Cannot detect contradictions across papers

See [Limitations Report](docs/06_limitations_report.md) for detailed analysis.

**Why This Matters**: These limitations motivate the transition to modern agentic AI systems in Milestone-2.

---

## 🧪 Example Usage

### Sample Analysis

**Input**: 3 research papers on machine learning

**Output**:
- **5 Topics** discovered (e.g., "Neural Networks", "Optimization", "Applications")
- **50 Keywords** extracted
- **5-sentence summary** generated
- **Coherence score**: 0.52 (Good)
- **Topic diversity**: 0.68

**Visualizations**: Word clouds, keyword bar charts, topic distributions

---

## 🔬 Academic Context

This project is designed for **academic learning** and demonstrates:

✅ Fundamental NLP concepts and algorithms  
✅ Proper evaluation methodology  
✅ Transparent documentation of limitations  
✅ Interview-ready explanations  
✅ Modular, readable code  

**Suitable for**:
- NLP course projects
- Research methodology demonstrations
- Baseline comparisons for advanced methods
- Understanding traditional ML limitations

---

## 🤝 Contributing

This is an academic project. Contributions welcome for:
- Bug fixes
- Documentation improvements
- Additional classical NLP methods
- Evaluation metrics

**Guidelines**:
- Keep code modular and well-commented
- Maintain academic rigor
- Document limitations honestly
- No LLMs or transformers (Milestone-1 constraint!)

---

## 📚 Documentation

Comprehensive documentation available in `docs/`:

1. [Problem Understanding](docs/01_problem_understanding.md) - Use cases and objectives
2. [Input-Output Specification](docs/02_input_output_spec.md) - Data formats and examples
3. [System Architecture](docs/03_system_architecture.md) - Components and data flow
4. [Pipeline Explanation](docs/04_pipeline_explanation.md) - Algorithm details
5. [Evaluation Results](docs/05_evaluation_results.md) - Sample outputs and metrics
6. [Limitations Report](docs/06_limitations_report.md) - Constraints and future work

---

## 🐛 Troubleshooting

### Common Issues

**1. NLTK data not found**
```bash
python -c "import nltk; nltk.download('all')"
```

**2. PDF extraction fails**
- Ensure PyPDF2 and pdfplumber are installed
- Try converting PDF to text externally
- Check if PDF is scanned (OCR needed)

**3. Coherence score is very low**
- Try adjusting number of topics
- Check preprocessing (too aggressive filtering?)
- Ensure enough documents (minimum 3-5)

**4. Streamlit won't start**
```bash
streamlit run ui/streamlit_app.py --server.port 8502
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Authors

**Mid-Semester Project** - Machine Learning & NLP Course  
**Milestone-1**: Traditional NLP Techniques

---

## 🎓 Citation

If you use this code for research or education, please cite:

```bibtex
@software{traditional_nlp_analysis,
  title={Traditional NLP Research Topic Analysis System},
  author={Your Name},
  year={2026},
  description={Classical NLP and ML techniques for research document analysis},
  milestone={1}
}
```

---

## 🔮 Future Work (Milestone-2)

The next milestone will implement **Agentic AI** features:
- Transformer-based embeddings (BERT, etc.)
- Large Language Models for understanding
- Multi-step reasoning workflows
- Tool use (web search, databases)
- Fact-checking and verification
- Abstractive summarization
- Cross-document reasoning

**Goal**: Demonstrate how modern AI addresses classical NLP limitations.

---

## 📞 Support

For questions or issues:
- Check documentation in `docs/`
- Review code comments
- Consult course materials

---

**Built with ❤️ for understanding the foundations of NLP**
