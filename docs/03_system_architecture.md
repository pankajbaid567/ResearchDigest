# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ File Upload  │  │ Text Input   │  │  Configuration Panel   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┤
│  │             Run Analysis Button                              │
│  └──────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┤
│  │         Results Display (Tabs: Topics, Keywords,             │
│  │         Summary, Evaluation, Visualizations, Limitations)    │
│  └──────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. DOCUMENT LOADER (document_loader.py)                 │  │
│  │     • Load .txt, .pdf files                              │  │
│  │     • Extract text using PyPDF2/pdfplumber              │  │
│  │     • Clean and validate text                            │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  2. TEXT PREPROCESSOR (preprocessor.py)                  │  │
│  │     • Tokenization (NLTK/spaCy)                          │  │
│  │     • Stop-word removal (NLTK stopwords + custom)        │  │
│  │     • Lemmatization (NLTK/spaCy)                         │  │
│  │     • Output: cleaned tokens                             │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  3. FEATURE EXTRACTOR (feature_extractor.py)             │  │
│  │     • TF-IDF Vectorization (sklearn)                     │  │
│  │     • Bag-of-Words (optional)                            │  │
│  │     • Output: feature matrix + feature names             │  │
│  └──────────────┬────────────────────┬──────────────────────┘  │
│                 │                    │                          │
│  ┌──────────────▼────────┐  ┌────────▼──────────────────────┐  │
│  │ 4a. TOPIC MODELER     │  │ 4b. CLUSTERER                 │  │
│  │    (topic_modeler.py) │  │     (clustering.py)           │  │
│  │  • LDA (Gensim)       │  │  • K-Means (sklearn)          │  │
│  │  • Topic-word dist.   │  │  • Elbow method               │  │
│  │  • Coherence calc.    │  │  • Silhouette score           │  │
│  └──────────────┬────────┘  └────────┬──────────────────────┘  │
│                 └────────────┬───────┘                          │
│  ┌──────────────────────────▼─────────────────────────────────┐  │
│  │  5. KEYWORD EXTRACTOR (keyword_extractor.py)              │  │
│  │     • Extract top-N keywords per topic                    │  │
│  │     • Generate theme labels (heuristic)                   │  │
│  │     • Calculate keyword frequencies                       │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  6. SUMMARIZER (summarizer.py)                           │  │
│  │     • TF-IDF sentence scoring                            │  │
│  │     • Frequency-based ranking                            │  │
│  │     • Position-weighted scoring                          │  │
│  │     • Extract top sentences                              │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │  7. EVALUATOR (evaluator.py)                             │  │
│  │     • Topic coherence (c_v, u_mass)                      │  │
│  │     • Topic diversity                                     │  │
│  │     • Clustering metrics (silhouette, Davies-Bouldin)    │  │
│  │     • Interpretation generation                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT & VISUALIZATION                         │
│  • Topics with keywords and weights                              │
│  • Overall keyword rankings                                      │
│  • Extractive summary                                            │
│  • Coherence scores and metrics                                  │
│  • Word clouds and charts                                        │
│  • Limitations documentation                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Architecture

### Core Modules

#### 1. Document Loader (`src/document_loader.py`)

**Responsibilities**:
- Load documents from file paths or Streamlit uploads
- Support multiple formats (.txt, .pdf, .md)
- Extract text from PDFs using PyPDF2 and pdfplumber
- Basic text cleaning

**Key Classes/Functions**:
- `DocumentLoader`: Main class
  - `load_single_document(path)`: Load one document
  - `load_multiple_documents(paths)`: Load batch
  - `load_from_uploaded_files(files)`: Streamlit integration
  - `_load_pdf_file(path)`: PDF-specific extraction

**Dependencies**: PyPDF2, pdfplumber, pathlib

---

#### 2. Text Preprocessor (`src/preprocessor.py`)

**Responsibilities**:
- Tokenize text (words and sentences)
- Remove stop words
- Lemmatize tokens
- Clean special characters

**Key Classes/Functions**:
- `TextPreprocessor`: Main class
  - `preprocess_text(text)`: Complete pipeline
  - `tokenize_sentences(text)`: Sentence splitting
  - `tokenize_words(text)`: Word tokenization
  - `remove_stopwords(tokens)`: Filter stop words
  - `lemmatize_nltk(tokens)` / `lemmatize_spacy(tokens)`: Lemmatization

**Dependencies**: NLTK (punkt, stopwords, wordnet), spaCy (optional)

---

#### 3. Feature Extractor (`src/feature_extractor.py`)

**Responsibilities**:
- Convert text to numerical features
- TF-IDF vectorization (primary)
- Bag-of-Words (optional)
- Extract top terms

**Key Classes/Functions**:
- `FeatureExtractor`: Main class
  - `extract_tfidf(documents)`: TF-IDF matrix
  - `extract_bow(documents)`: BoW matrix
  - `get_top_terms_per_document(matrix, names)`: Document keywords
  - `get_overall_top_terms(matrix, names)`: Global keywords

**Dependencies**: scikit-learn (TfidfVectorizer, CountVectorizer), numpy

---

#### 4. Topic Modeler (`src/topic_modeler.py`)

**Responsibilities**:
- LDA topic modeling
- Topic-word distributions
- Document-topic assignments
- Coherence calculation

**Key Classes/Functions**:
- `TopicModeler`: Main class
  - `train_lda_model(documents, num_topics)`: Train LDA
  - `get_topics(num_words)`: Extract topic keywords
  - `get_document_topics()`: Doc-topic distributions
  - `calculate_coherence()`: Compute c_v coherence

**Dependencies**: Gensim (LdaModel, CoherenceModel, Dictionary)

---

#### 5. Document Clusterer (`src/clustering.py`)

**Responsibilities**:
- K-Means clustering (alternative to LDA)
- Optimal cluster number selection
- Cluster term extraction

**Key Classes/Functions**:
- `DocumentClusterer`: Main class
  - `perform_kmeans(matrix, n_clusters)`: Cluster documents
  - `find_optimal_clusters(matrix)`: Elbow method
  - `get_top_terms_per_cluster(matrix, names)`: Cluster keywords

**Dependencies**: scikit-learn (KMeans, silhouette_score)

---

#### 6. Keyword Extractor (`src/keyword_extractor.py`)

**Responsibilities**:
- Extract keywords from topics
- Generate theme labels
- Calculate keyword frequencies

**Key Classes/Functions**:
- `KeywordExtractor`: Main class
  - `extract_keywords_from_topics(topics)`: Topic keywords
  - `generate_theme_labels(keywords)`: Heuristic labeling
  - `extract_keywords_tfidf(matrix, names)`: TF-IDF keywords

**Dependencies**: numpy, collections.Counter

---

#### 7. Extractive Summarizer (`src/summarizer.py`)

**Responsibilities**:
- Sentence-level extractive summarization
- Multiple scoring methods
- Summary statistics

**Key Classes/Functions**:
- `ExtractiveSummarizer`: Main class
  - `summarize_tfidf(text, n)`: TF-IDF method
  - `summarize_frequency(text, n)`: Frequency method
  - `summarize_position_weighted(text, n)`: Position-aware method
  - `get_summary_statistics(original, summary)`: Metrics

**Dependencies**: scikit-learn, numpy, re

---

#### 8. Evaluator (`src/evaluator.py`)

**Responsibilities**:
- Calculate topic coherence
- Evaluate clustering quality
- Generate interpretations

**Key Classes/Functions**:
- `Evaluator`: Main class
  - `evaluate_topics(model, texts, dict)`: Comprehensive evaluation
  - `evaluate_clustering(labels, matrix)`: Clustering metrics
  - `generate_interpretation(topics, coherence)`: Human-readable report

**Dependencies**: Gensim (CoherenceModel), scikit-learn (metrics)

---

### User Interface Module

#### Streamlit App (`ui/streamlit_app.py`)

**Responsibilities**:
- Web interface
- User input handling
- Pipeline orchestration
- Results visualization

**Key Functions**:
- `main()`: Application entry point
- `run_analysis(...)`: Execute complete pipeline
- `display_results(results)`: Render results in tabs
- `generate_wordcloud(words)`: Word cloud visualization
- `plot_topic_keywords(words)`: Bar chart visualization

**Dependencies**: Streamlit, matplotlib, wordcloud, pandas

---

## Data Flow

### Step-by-Step Processing

1. **User Input** → Documents uploaded or text entered
2. **Document Loader** → Extract text: `{doc_id: text}`
3. **Preprocessor** → Tokenize & clean: `{doc_id: {tokens, sentences, ...}}`
4. **Feature Extractor** → Vectorize: `(tfidf_matrix, feature_names)`
5. **Topic Modeler/Clusterer** → Discover topics: `{topic_id: keywords}`
6. **Keyword Extractor** → Extract & label: `{topic_id: keywords, themes}`
7. **Summarizer** → Select sentences: `summary_text`
8. **Evaluator** → Calculate metrics: `{coherence, diversity, ...}`
9. **UI Display** → Render results in tabs

### Data Structures

**Preprocessed Document**:
```python
{
    'original_text': str,
    'cleaned_text': str,
    'sentences': List[str],
    'tokens': List[str],
    'tokens_no_stopwords': List[str],
    'lemmatized_tokens': List[str],
    'token_count': int,
    'sentence_count': int
}
```

**Topic**:
```python
{
    topic_id: [
        ('keyword1', weight1),
        ('keyword2', weight2),
        ...
    ]
}
```

**Evaluation Results**:
```python
{
    'coherence_score': float,
    'coherence_per_topic': List[float],
    'topic_diversity': float,
    'interpretation': str
}
```

## Deployment Architecture

```
Local Machine
├── Python 3.10+ Runtime
├── Virtual Environment (venv)
├── Dependencies (requirements.txt)
│   ├── NLTK (+ downloaded data)
│   ├── Gensim
│   ├── scikit-learn
│   ├── Streamlit
│   └── PDF libraries
├── Application Code
│   ├── src/ (core modules)
│   ├── ui/ (Streamlit app)
│   └── docs/ (documentation)
└── Streamlit Server (localhost:8501)
```

**No cloud deployment required for Milestone-1.**

---

**Previous**: [Input-Output Specification](02_input_output_spec.md)  
**Next**: [Pipeline Explanation](04_pipeline_explanation.md)
