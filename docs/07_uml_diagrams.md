# System UML Diagrams

## 1. Class Diagram

This diagram illustrates the structure of the `Project1-GenAI` system, showing the main classes, their methods, and relationships. The system follows a modular design with a clean separation of concerns.

```mermaid
classDiagram
    class DocumentLoader {
        +load_single_document(file_path: str) str
        +load_multiple_documents(file_paths: List[str]) Dict
        +load_from_uploaded_files(uploaded_files: List) Dict
        -read_pdf(file_path: str) str
        -read_text_file(file_path: str) str
    }

    class TextPreprocessor {
        -use_spacy: bool
        -stop_words: set
        +preprocess_text(text: str) Dict
        -clean_text(text: str) str
        -tokenize(text: str) List
        -remove_stopwords(tokens: List) List
        -lemmatize(tokens: List) List
    }

    class FeatureExtractor {
        -tfidf_vectorizer: TfidfVectorizer
        -bow_vectorizer: CountVectorizer
        +extract_tfidf(documents: List, max_features: int) Tuple
        +extract_bow(documents: List, max_features: int) Tuple
        +get_overall_top_terms(tfidf_matrix: ndarray, features: List, n: int) List
        +get_feature_statistics(tfidf_matrix: ndarray) Dict
    }

    class TopicModeler {
        -lda_model: LdaModel
        -dictionary: Dictionary
        -corpus: List
        +train_lda_model(documents: List, num_topics: int) LdaModel
        +get_topics(num_words: int) Dict
        +get_document_topics() List
        +calculate_coherence(coherence_type: str) float
    }

    class DocumentClusterer {
        -kmeans: KMeans
        +perform_kmeans(tfidf_matrix: ndarray, n_clusters: int) KMeans
        +get_cluster_labels() ndarray
        +get_top_terms_per_cluster(tfidf_matrix: ndarray, features: List, n: int) Dict
        +get_cluster_statistics() Dict
    }

    class ExtractiveSummarizer {
        +summarize_tfidf(text: str, num_sentences: int) str
        +summarize_frequency(text: str, num_sentences: int) str
        +summarize_position_weighted(text: str, num_sentences: int) str
        +get_summary_statistics(original: str, summary: str) Dict
    }

    class KeywordExtractor {
        +extract_keywords_from_topics(topics: Dict, n: int) Dict
        +extract_keywords_tfidf(matrix, features, n) List
        +generate_theme_labels(keywords: Dict) Dict
        -_calculate_tfidf_scores(matrix, features) Dict
    }

    class Evaluator {
        +evaluate_topics(model, texts, dictionary) Dict
        +evaluate_clusters(matrix, labels) Dict
        -calculate_topic_diversity(topics) float
    }

    class StreamlitApp {
        +main()
        +run_analysis(documents, config) Dict
        +display_results(results)
    }

    %% Relationships
    StreamlitApp --> DocumentLoader : uses
    StreamlitApp --> TextPreprocessor : uses
    StreamlitApp --> FeatureExtractor : uses
    StreamlitApp --> TopicModeler : uses
    StreamlitApp --> DocumentClusterer : uses
    StreamlitApp --> KeywordExtractor : uses
    StreamlitApp --> ExtractiveSummarizer : uses
    StreamlitApp --> Evaluator : uses

    TopicModeler ..> TextPreprocessor : expects tokenized input
    DocumentClusterer ..> FeatureExtractor : expects TF-IDF matrix
    KeywordExtractor ..> TopicModeler : processes topics
```

## 2. Sequence Diagram (Analysis Flow)

This diagram shows the sequence of operations when a user runs an analysis in the Streamlit application.

```mermaid
sequenceDiagram
    actor User
    participant App as StreamlitApp
    participant Loader as DocumentLoader
    participant Prep as TextPreprocessor
    participant Feat as FeatureExtractor
    participant Topic as TopicModeler
    participant Cluster as DocumentClusterer
    participant Kw as KeywordExtractor
    participant Summ as ExtractiveSummarizer
    participant Eval as Evaluator

    User->>App: Upload Documents / Enter Text
    App->>Loader: load_documents()
    Loader-->>App: Raw Documents

    User->>App: Click "Run Analysis"
    
    rect rgb(240, 248, 255)
        note right of App: Step 1: Preprocessing
        loop For each document
            App->>Prep: preprocess_text(raw_text)
            Prep-->>App: Tokens & Lemmas
        end
    end

    rect rgb(255, 248, 240)
        note right of App: Step 2: Feature Extraction
        App->>Feat: extract_tfidf(tokens)
        Feat-->>App: TF-IDF Matrix & Feature Names
    end

    alt Topic Extraction Method = LDA
        rect rgb(240, 255, 240)
            note right of App: Step 3A: Topic Modeling (LDA)
            App->>Topic: train_lda_model(tokens)
            Topic-->>App: LDA Model
            App->>Topic: get_topics()
            Topic-->>App: Topics List
            App->>Topic: calculate_coherence()
            Topic-->>App: Coherence Score
        end
    else Topic Extraction Method = K-Means
        rect rgb(255, 240, 240)
            note right of App: Step 3B: Clustering (K-Means)
            App->>Cluster: perform_kmeans(tfidf_matrix)
            Cluster-->>App: KMeans Model & Labels
            App->>Cluster: get_top_terms_per_cluster()
            Cluster-->>App: Cluster Terms
        end
    end

    rect rgb(250, 250, 230)
        note right of App: Step 4: Keyword Extraction
        App->>Kw: extract_keywords()
        Kw-->>App: Top Keywords & Theme Labels
    end

    rect rgb(240, 240, 255)
        note right of App: Step 5: Summarization
        App->>Summ: summarize(combined_text)
        Summ-->>App: Extractive Summary
    end

    rect rgb(255, 240, 255)
        note right of App: Evaluation
        App->>Eval: evaluate_topics/clusters()
        Eval-->>App: Evaluation Metrics
    end

    App-->>User: Display Results (Tabs)
```
