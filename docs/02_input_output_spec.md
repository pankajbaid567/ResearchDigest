# Input-Output Specification

## System Inputs

### Input Option 1: Document Upload

**Format**: Multiple files  
**Supported Types**:
- Plain text files (`.txt`, `.md`)
- PDF documents (`.pdf`)

**Constraints**:
- File size: Recommended < 10 MB per file
- Encoding: UTF-8 preferred
- Content: Research papers, articles, or technical documents

**Example Input**:
```
research_paper_1.pdf
research_paper_2.pdf
literature_review.txt
```

### Input Option 2: Direct Text Input

**Format**: Plain text  
**Content**: Research text, topics, or keywords  
**Length**: No strict limit (longer texts may take more processing time)

**Example Input**:
```
Machine learning algorithms for natural language processing 
have evolved significantly. Deep learning models such as 
transformers have revolutionized text classification, 
named entity recognition, and semantic analysis...
```

## Configuration Parameters

Users can configure the following parameters via the UI:

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `num_topics` | Integer | 2-10 | 5 | Number of topics to extract |
| `num_keywords` | Integer | 5-20 | 10 | Keywords per topic |
| `num_sentences` | Integer | 3-10 | 5 | Sentences in summary |
| `summary_method` | String | TF-IDF, Frequency, Position | TF-IDF | Summarization algorithm |
| `modeling_method` | String | LDA, K-Means | LDA | Topic extraction method |

## Processing Pipeline

```
Input Documents
    ↓
Text Extraction (PDF/Text parsing)
    ↓
Preprocessing (Tokenization, Lemmatization, Stop-word removal)
    ↓
Feature Extraction (TF-IDF Vectorization)
    ↓
Topic Modeling (LDA) OR Clustering (K-Means)
    ↓
Keyword Extraction (Top-N terms per topic)
    ↓
Extractive Summarization (Sentence ranking)
    ↓
Evaluation (Coherence metrics)
    ↓
Output (Topics, Keywords, Summary, Metrics)
```

## System Outputs

### Output 1: Topics

**Format**: Structured dictionary  
**Content**: Topic ID, keywords with weights, theme label

**Example**:
```python
{
    "Topic 0": {
        "keywords": [
            ("machine", 0.045),
            ("learning", 0.042),
            ("neural", 0.038),
            ("network", 0.035),
            ("algorithm", 0.032)
        ],
        "theme": "Machine Learning: machine + learning + neural"
    },
    "Topic 1": {
        "keywords": [
            ("language", 0.051),
            ("nlp", 0.047),
            ("text", 0.041),
            ("processing", 0.039),
            ("semantic", 0.036)
        ],
        "theme": "NLP: language + nlp + text"
    }
}
```

### Output 2: Overall Keywords

**Format**: Ranked list  
**Content**: Top-20 keywords with TF-IDF scores

**Example**:
```python
[
    ("machine learning", 0.125),
    ("natural language", 0.118),
    ("deep learning", 0.112),
    ("neural network", 0.105),
    ("text classification", 0.098),
    ...
]
```

### Output 3: Extractive Summary

**Format**: Plain text  
**Content**: Top-N most important sentences from input

**Example**:
```
Machine learning algorithms have revolutionized natural language 
processing. Transformer models achieve state-of-the-art performance 
on many NLP tasks. However, classical methods remain valuable for 
interpretability and efficiency. Topic modeling using LDA enables 
unsupervised discovery of themes in document collections.
```

**Statistics**:
- Original sentences: 45
- Summary sentences: 5
- Original words: 1,250
- Summary words: 312
- Compression ratio: 24.96%

### Output 4: Evaluation Metrics

**LDA Method**:
```python
{
    "coherence_score": 0.487,
    "coherence_type": "c_v",
    "topic_diversity": 0.72,
    "num_topics": 5,
    "interpretation": "Good (0.487): Topics show good coherence with clear themes.",
    "coherence_per_topic": [0.52, 0.48, 0.51, 0.45, 0.43]
}
```

**K-Means Method**:
```python
{
    "silhouette_score": 0.35,
    "davies_bouldin_score": 1.42,
    "cluster_distribution": {
        0: {"docs": 8, "percentage": 40.0},
        1: {"docs": 6, "percentage": 30.0},
        2: {"docs": 4, "percentage": 20.0},
        3: {"docs": 2, "percentage": 10.0}
    }
}
```

### Output 5: Visualizations

1. **Word Cloud**: Visual representation of top keywords
2. **Topic Keywords Bar Chart**: Horizontal bar chart of keyword weights
3. **Topic Distribution**: (Optional) Pie chart of document distribution across topics

## Data Flow Example

### Input
```
Files: [paper1.pdf, paper2.pdf, paper3.txt]
Config: {topics: 5, keywords: 10, summary_sentences: 5}
```

### Processing
```
1. Extract text: 3 documents, ~15,000 words total
2. Preprocess: 12,354 tokens after cleaning
3. TF-IDF: 500 features extracted
4. LDA: 5 topics discovered
5. Summary: 5 sentences selected (compression: 18%)
```

### Output
```json
{
  "topics": {
    "0": ["machine", "learning", "model", ...],
    "1": ["data", "analysis", "statistical", ...],
    ...
  },
  "keywords": [["machine learning", 0.125], ...],
  "summary": "Machine learning has transformed...",
  "coherence": 0.487,
  "stats": {
    "documents": 3,
    "features": 500,
    "compression": 0.18
  }
}
```

## Error Handling

| Error Type | Cause | System Response |
|------------|-------|-----------------|
| No documents | User clicks "Run" without input | Warning message |
| Unsupported format | User uploads .docx or other format | Skip file, notify user |
| PDF extraction failed | Corrupted or scanned PDF | Log error, try alternative method |
| Too few documents | Only 1 document for clustering | Proceed with LDA only |
| Empty text | Document has no extractable text | Skip document, notify user |

## Output Formats for Export

While the primary interface is web-based, the system can be extended to export results in:

- **JSON**: Machine-readable format for downstream processing
- **CSV**: Topic-keyword matrix for analysis
- **Markdown**: Formatted report for documentation
- **Text**: Plain text summary and keywords

---

**Previous**: [Problem Understanding](01_problem_understanding.md)  
**Next**: [System Architecture](03_system_architecture.md)
