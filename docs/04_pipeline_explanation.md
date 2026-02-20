# NLP Pipeline Explanation

## Overview

This document provides a detailed, step-by-step explanation of the Traditional NLP pipeline used in this system. Each algorithm is explained with mathematical foundations, implementation choices, and rationale.

---

## Pipeline Overview

```
Raw Documents → Preprocessing → Vectorization → Topic Modeling → Summarization → Evaluation
```

---

## Step 1: Document Loading

### Objective
Extract raw text from various document formats.

### Implementation

**Text Files (.txt, .md)**:
- Direct file reading with UTF-8 encoding
- Handles encoding errors gracefully

**PDF Files (.pdf)**:
- **Primary**: pdfplumber (better text extraction)
- **Fallback**: PyPDF2 (wider compatibility)
- Extracts text page-by-page and concatenates

### Challenges
- Scanned PDFs require OCR (not implemented)
- Multi-column layouts may scramble text order
- Tables and figures are poorly extracted

---

## Step 2: Text Preprocessing

### Objective
Transform raw text into clean, normalized tokens suitable for analysis.

### Sub-Step 2.1: Text Cleaning

**Operations**:
1. Convert to lowercase: `text.lower()`
2. Remove special characters: `re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)`
3. Normalize whitespace: `re.sub(r'\s+', ' ', text)`

**Rationale**: Reduce vocabulary size, normalize variations

### Sub-Step 2.2: Tokenization

**Sentence Tokenization**:
```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
```

**Algorithm**: NLTK's Punkt tokenizer (unsupervised, trained on corpora)
- Detects sentence boundaries using punctuation and capitalization patterns
- Handles abbreviations (e.g., "Dr.", "et al.")

**Word Tokenization**:
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
```

**Algorithm**: Penn Treebank tokenizer
- Splits on whitespace and punctuation
- Handles contractions (e.g., "don't" → ["do", "n't"])

**Post-filtering**:
```python
tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
```
- Keep only alphabetic tokens
- Remove single characters

### Sub-Step 2.3: Stop-word Removal

**Stop Words**: High-frequency words with little semantic content

**Sources**:
1. NLTK English stopwords: ["the", "a", "an", "is", ...]
2. Custom academic terms: ["et", "al", "fig", "table", "pp", ...]

**Implementation**:
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens_filtered = [t for t in tokens if t not in stop_words]
```

**Rationale**: 
- Reduces dimensionality
- Focuses on content words
- Improves topic quality

**Trade-offs**:
- May lose important phrases ("not" is a stopword!)
- Domain-specific stopwords need manual curation

### Sub-Step 2.4: Lemmatization

**Objective**: Reduce words to their base forms

**Examples**:
- "running", "runs", "ran" → "run"
- "better", "best" → "good"
- "studies", "studying" → "study"

**Methods**:

**Option A: NLTK WordNetLemmatizer**:
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token) for token in tokens]
```

**Option B: spaCy (preferred)**:
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
lemmas = [token.lemma_ for token in doc if token.is_alpha]
```

**spaCy Advantages**:
- Uses part-of-speech tagging for better accuracy
- Faster for large texts
- Better handling of irregular verbs

**Output**: List of lemmatized tokens

---

## Step 3: Feature Extraction (TF-IDF)

### Objective
Convert text to numerical vectors suitable for ML algorithms.

### TF-IDF: Term Frequency-Inverse Document Frequency

**Mathematical Foundation**:

**Term Frequency (TF)**:
```
TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
```

**Inverse Document Frequency (IDF)**:
```
IDF(t, D) = log(Total documents / Documents containing term t)
```

**TF-IDF**:
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Intuition**:
- **High TF**: Term appears frequently in document → Important to that document
- **High IDF**: Term appears in few documents → Discriminative
- **TF-IDF**: Balances frequency and uniqueness

### Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500,        # Top 500 most important terms
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Term must appear in ≥2 docs
    max_df=0.85,             # Ignore terms in >85% of docs
    sublinear_tf=True,       # Use log(TF) instead of TF
    use_idf=True,
    smooth_idf=True          # Add 1 to IDF to avoid divide-by-zero
)

tfidf_matrix = vectorizer.fit_transform(documents)
```

**Parameters Explained**:

- **max_features=500**: Keep only top 500 terms (reduces dimensionality)
- **ngram_range=(1,2)**: Capture phrases like "machine learning" (bigram)
- **min_df=2**: Ignore typos and rare terms (appear in only 1 doc)
- **max_df=0.85**: Ignore ubiquitous terms (appear in almost every doc)
- **sublinear_tf=True**: Scale TF using `1 + log(TF)` to dampen very frequent terms

**Output**: Sparse matrix of shape `(n_documents, n_features)`

### Alternative: Bag-of-Words (BoW)

Similar to TF-IDF but without IDF weighting:
```python
from sklearn.feature_extraction.text import CountVectorizer
bow_matrix = CountVectorizer().fit_transform(documents)
```

**When to use**: Naive Bayes classification, simple baselines

---

## Step 4: Topic Modeling (LDA)

### Objective
Discover latent topics in document collection.

### LDA: Latent Dirichlet Allocation

**Generative Model**:

LDA assumes each document is a mixture of topics, and each topic is a mixture of words.

**Assumptions**:
1. Documents are probability distributions over topics
2. Topics are probability distributions over words

**Process** (for each document):
1. Sample topic distribution θ ~ Dirichlet(α)
2. For each word position:
   - Sample topic z ~ Categorical(θ)
   - Sample word w ~ Categorical(β_z)

**Hyperparameters**:
- **α (alpha)**: Document-topic density
  - Low α → Documents focus on few topics
  - High α → Documents mix many topics
- **β (eta)**: Topic-word density
  - Low β → Topics focus on few words
  - High β → Topics mix many words

### Implementation

```python
from gensim import corpora
from gensim.models import LdaModel

# Prepare corpus
dictionary = corpora.Dictionary(tokenized_docs)
dictionary.filter_extremes(no_below=2, no_above=0.8, keep_n=1000)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Train LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,
    passes=15,             # Number of passes through corpus
    iterations=100,        # Max iterations for convergence
    alpha='auto',          # Learn optimal α
    eta='auto',            # Learn optimal η
    random_state=42        # Reproducibility
)
```

**Training Process**:
1. Initialize topics randomly
2. **E-step**: Estimate topic assignments for words
3. **M-step**: Update topic-word and document-topic distributions
4. Repeat until convergence

**Output**:
- Topic-word distributions: `P(word | topic)`
- Document-topic distributions: `P(topic | document)`

### Extracting Topics

```python
topics = lda_model.show_topics(num_topics=5, num_words=10)
# Returns: [(topic_id, [(word, weight), ...]), ...]
```

**Example Output**:
```
Topic 0: [("learning", 0.045), ("machine", 0.042), ("algorithm", 0.038), ...]
Topic 1: [("neural", 0.051), ("network", 0.047), ("layer", 0.041), ...]
```

---

## Step 5: Alternative - K-Means Clustering

### Objective
Group similar documents using distance-based clustering.

### Algorithm

**K-Means**:
1. Initialize K cluster centroids randomly
2. **Assignment step**: Assign each document to nearest centroid
3. **Update step**: Recalculate centroids as mean of assigned documents
4. Repeat until convergence

**Distance Metric**: Euclidean distance in TF-IDF space

### Implementation

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    max_iter=300,
    n_init=10             # Run 10 times, keep best
)

labels = kmeans.fit_predict(tfidf_matrix)
```

### Finding Optimal K

**Elbow Method**:
- Plot inertia (within-cluster sum of squares) vs. K
- Look for "elbow" where improvement diminishes

**Silhouette Score**:
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(tfidf_matrix, labels)
# Range: -1 to 1 (higher is better)
```

### Extracting Cluster Terms

```python
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    top_indices = centroid.argsort()[-10:][::-1]
    top_terms = [feature_names[idx] for idx in top_indices]
    print(f"Cluster {i}: {top_terms}")
```

---

## Step 6: Keyword Extraction

### Method 1: From Topics (LDA)

Simply extract top words from each topic:
```python
keywords = {topic_id: [word for word, _ in topic_words[:10]] 
            for topic_id, topic_words in topics.items()}
```

### Method 2: Overall TF-IDF

Calculate mean TF-IDF across all documents:
```python
mean_tfidf = np.mean(tfidf_matrix, axis=0)
top_indices = np.argsort(mean_tfidf)[::-1][:20]
top_keywords = [feature_names[i] for i in top_indices]
```

---

## Step 7: Extractive Summarization

### Objective
Select most important sentences from original text.

### Method 1: TF-IDF Sentence Scoring

**Algorithm**:
1. Split text into sentences
2. Compute TF-IDF for each sentence (treating as mini-document)
3. Calculate sentence score = sum of TF-IDF values
4. Select top-N sentences

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(sentences)

# Sum TF-IDF values for each sentence
sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)

# Get top N sentences
top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
top_indices_sorted = sorted(top_indices)  # Maintain original order

summary = ' '.join([sentences[i] for i in top_indices_sorted])
```

**Rationale**: Sentences with many important (high TF-IDF) words are most informative

### Method 2: Frequency-Based Scoring

**Algorithm**:
1. Calculate word frequencies in entire document
2. Normalize frequencies: `freq / max_freq`
3. Score each sentence = average frequency of its words
4. Select top-N sentences

```python
word_freq = Counter(words)
max_freq = max(word_freq.values())
for word in word_freq:
    word_freq[word] = word_freq[word] / max_freq

sentence_scores = {}
for i, sent in enumerate(sentences):
    words = sent.split()
    score = sum(word_freq.get(w, 0) for w in words) / len(words)
    sentence_scores[i] = score
```

**Rationale**: Sentences containing frequently occurring words capture main themes

### Method 3: Position-Weighted Scoring

Combines TF-IDF with position bias:
```python
content_score = tfidf_sentence_score / max(tfidf_sentence_scores)
position_score = 1 - (sentence_index / total_sentences)

final_score = 0.7 * content_score + 0.3 * position_score
```

**Rationale**: First sentences often contain key information (lead paragraph bias)

---

## Step 8: Evaluation

### Topic Coherence (c_v)

**Objective**: Measure how semantically interpretable topics are.

**Algorithm**:
1. For top words in each topic, compute pairwise similarity
2. Similarity based on co-occurrence in sliding windows over reference corpus
3. Aggregate similarities for overall coherence

```python
from gensim.models.coherencemodel import CoherenceModel

coherence_model = CoherenceModel(
    model=lda_model,
    texts=tokenized_docs,
    dictionary=dictionary,
    coherence='c_v'
)

coherence_score = coherence_model.get_coherence()
```

**Range**: 0.0 to 1.0 (higher better, typically 0.3-0.7)

**Interpretation**:
- > 0.6: Excellent
- 0.5-0.6: Good
- 0.4-0.5: Moderate
- < 0.4: Poor (needs tuning)

### Topic Diversity

**Formula**:
```
Diversity = Unique words across all topics / Total words in all topics
```

**Range**: 0.0 to 1.0 (higher is better)
- 1.0: All topics completely unique
- 0.0: All topics identical

### Clustering Metrics

**Silhouette Score**:
- Measures how similar documents are to their own cluster vs. other clusters
- Range: -1 (wrong cluster) to +1 (perfect cluster)

**Davies-Bouldin Index**:
- Average similarity between each cluster and its most similar cluster
- Range: 0+ (lower is better)

---

## Summary

This pipeline demonstrates:
✅ Classical preprocessing techniques  
✅ TF-IDF for feature extraction  
✅ LDA for probabilistic topic modeling  
✅ K-Means for distance-based clustering  
✅ Extractive summarization methods  
✅ Proper evaluation metrics  

**Limitations**: All methods are statistical, not semantic—see [Limitations Report](06_limitations_report.md).

---

**Previous**: [System Architecture](03_system_architecture.md)  
**Next**: [Evaluation Results](05_evaluation_results.md)
