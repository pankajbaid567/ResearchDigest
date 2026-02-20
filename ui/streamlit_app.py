"""
Streamlit Web Application
Interactive UI for Traditional NLP Research Topic Analysis System.
"""

import streamlit as st
import sys
from pathlib import Path
import logging
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from document_loader import DocumentLoader
from preprocessor import TextPreprocessor
from feature_extractor import FeatureExtractor
from topic_modeler import TopicModeler
from clustering import DocumentClusterer
from keyword_extractor import KeywordExtractor
from summarizer import ExtractiveSummarizer
from evaluator import Evaluator

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Traditional NLP Research Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}


def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.title("📚 Traditional NLP Research Topic Analysis")
    st.markdown("**Milestone-1**: Classical NLP & ML Techniques for Research Document Analysis")
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Topic Modeling")
        num_topics = st.slider("Number of Topics", 2, 10, 5)
        num_keywords = st.slider("Keywords per Topic", 5, 20, 10)
        
        st.subheader("Summarization")
        num_sentences = st.slider("Summary Sentences", 3, 10, 5)
        summary_method = st.selectbox(
            "Summarization Method",
            ["TF-IDF", "Frequency-based", "Position-weighted"]
        )
        
        st.subheader("Analysis Method")
        modeling_method = st.selectbox(
            "Topic Extraction",
            ["LDA (Gensim)", "K-Means Clustering"]
        )
        
        st.markdown("---")
        st.info("💡 **Note**: This system uses only classical NLP techniques—no LLMs or transformers!")
    
    # Input section
    st.header("📥 Input Documents")
    
    input_method = st.radio(
        "Choose input method:",
        ["Upload Documents (PDF/Text)", "Enter Text Directly"]
    )
    
    documents = {}
    
    if input_method == "Upload Documents (PDF/Text)":
        uploaded_files = st.file_uploader(
            "Upload research documents",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True,
            help="Upload one or more text or PDF files"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            loader = DocumentLoader()
            
            with st.spinner("Loading documents..."):
                documents = loader.load_from_uploaded_files(uploaded_files)
            
            if documents:
                st.info(f"📄 Loaded {len(documents)} documents")
    
    else:  # Enter text directly
        text_input = st.text_area(
            "Paste your research text or keywords:",
            height=200,
            placeholder="Enter research documents, topics, or keywords..."
        )
        
        if text_input:
            documents['input_text'] = text_input
            st.success("✅ Text received")
    
    # Analysis button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Run analysis
    if analyze_button and documents:
        with st.spinner("🔄 Processing... This may take a moment."):
            try:
                results = run_analysis(
                    documents, 
                    num_topics, 
                    num_keywords, 
                    num_sentences,
                    summary_method,
                    modeling_method
                )
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {str(e)}", exc_info=True)
    
    elif analyze_button and not documents:
        st.warning("⚠️ Please upload documents or enter text first!")
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, modeling_method)


def run_analysis(documents, num_topics, num_keywords, num_sentences, summary_method, modeling_method):
    """
    Run complete NLP analysis pipeline.
    
    Args:
        documents: Dictionary of documents
        num_topics: Number of topics to extract
        num_keywords: Keywords per topic
        num_sentences: Sentences in summary
        summary_method: Summarization method
        modeling_method: Topic modeling method
    
    Returns:
        Dictionary of analysis results
    """
    results = {}
    
    # Step 1: Preprocessing
    st.write("**Step 1/5**: Preprocessing documents...")
    preprocessor = TextPreprocessor(use_spacy=False)
    
    preprocessed_docs = {}
    for doc_id, text in documents.items():
        preprocessed_docs[doc_id] = preprocessor.preprocess_text(text)
    
    # Combine documents for analysis
    combined_text = ' '.join([text for text in documents.values()])
    all_tokens = [doc['lemmatized_tokens'] for doc in preprocessed_docs.values()]
    token_strings = [' '.join(tokens) for tokens in all_tokens]
    
    # Validate that we have enough content
    total_tokens = sum(len(tokens) for tokens in all_tokens)
    if total_tokens < 10:
        raise ValueError(
            f"Input text is too short! After preprocessing, only {total_tokens} tokens remain. "
            f"Please provide at least 1-2 paragraphs of content (100+ words). "
            f"Tip: Avoid very short bullet points—they often get filtered out by stop-word removal."
        )
    
    # Step 2: Feature Extraction
    st.write("**Step 2/5**: Extracting features (TF-IDF)...")
    feature_extractor = FeatureExtractor()
    tfidf_matrix, feature_names = feature_extractor.extract_tfidf(
        token_strings,
        max_features=500,
        ngram_range=(1, 2)
    )
    
    results['tfidf_matrix'] = tfidf_matrix
    results['feature_names'] = feature_names
    results['feature_stats'] = feature_extractor.get_feature_statistics(tfidf_matrix)
    
    # Step 3: Topic Modeling or Clustering
    st.write(f"**Step 3/5**: {modeling_method}...")
    
    if modeling_method == "LDA (Gensim)":
        # LDA Topic Modeling
        topic_modeler = TopicModeler()
        lda_model = topic_modeler.train_lda_model(
            all_tokens,
            num_topics=num_topics,
            passes=15
        )
        
        topics = topic_modeler.get_topics(num_words=num_keywords)
        doc_topics = topic_modeler.get_document_topics()
        dominant_topics = topic_modeler.assign_dominant_topic(doc_topics)
        
        results['method'] = 'LDA'
        results['topics'] = topics
        results['doc_topics'] = doc_topics
        results['dominant_topics'] = dominant_topics
        results['lda_model'] = lda_model
        results['topic_modeler'] = topic_modeler
        
        # Calculate coherence
        coherence_score = topic_modeler.calculate_coherence()
        results['coherence_score'] = coherence_score
        
    else:
        # K-Means Clustering
        clusterer = DocumentClusterer()
        kmeans_model = clusterer.perform_kmeans(tfidf_matrix, n_clusters=num_topics)
        
        cluster_labels = clusterer.get_cluster_labels()
        cluster_terms = clusterer.get_top_terms_per_cluster(
            tfidf_matrix,
            feature_names,
            n_terms=num_keywords
        )
        cluster_stats = clusterer.get_cluster_statistics()
        
        results['method'] = 'K-Means'
        results['topics'] = cluster_terms
        results['cluster_labels'] = cluster_labels
        results['cluster_stats'] = cluster_stats
        results['kmeans_model'] = kmeans_model
    
    # Step 4: Keyword Extraction
    st.write("**Step 4/5**: Extracting keywords and themes...")
    keyword_extractor = KeywordExtractor()
    
    # Extract keywords from topics
    topic_keywords = keyword_extractor.extract_keywords_from_topics(
        results['topics'],
        n_keywords=num_keywords
    )
    
    # Generate theme labels
    theme_labels = keyword_extractor.generate_theme_labels(topic_keywords)
    
    # Overall top keywords
    overall_keywords = keyword_extractor.extract_keywords_tfidf(
        tfidf_matrix,
        feature_names,
        n=20
    )
    
    results['topic_keywords'] = topic_keywords
    results['theme_labels'] = theme_labels
    results['overall_keywords'] = overall_keywords
    
    # Step 5: Extractive Summarization
    st.write("**Step 5/5**: Generating extractive summary...")
    summarizer = ExtractiveSummarizer()
    
    if summary_method == "TF-IDF":
        summary = summarizer.summarize_tfidf(combined_text, num_sentences=num_sentences)
    elif summary_method == "Frequency-based":
        summary = summarizer.summarize_frequency(combined_text, num_sentences=num_sentences)
    else:  # Position-weighted
        summary = summarizer.summarize_position_weighted(combined_text, num_sentences=num_sentences)
    
    summary_stats = summarizer.get_summary_statistics(combined_text, summary)
    
    results['summary'] = summary
    results['summary_stats'] = summary_stats
    results['summary_method'] = summary_method
    
    # Evaluation
    if modeling_method == "LDA (Gensim)":
        evaluator = Evaluator()
        evaluation = evaluator.evaluate_topics(
            lda_model,
            all_tokens,
            topic_modeler.dictionary
        )
        results['evaluation'] = evaluation
    
    return results


def display_results(results, modeling_method):
    """Display analysis results in organized tabs."""
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Topics", 
        "🔑 Keywords", 
        "📝 Summary", 
        "📈 Evaluation",
        "🎨 Visualizations",
        "⚠️ Limitations"
    ])
    
    # Tab 1: Topics
    with tab1:
        st.subheader(f"Discovered Topics ({modeling_method})")
        
        topics = results['topics']
        theme_labels = results['theme_labels']
        
        for topic_id in sorted(topics.keys()):
            with st.expander(f"**Topic {topic_id}**: {theme_labels.get(topic_id, 'N/A')}", expanded=True):
                topic_words = topics[topic_id]
                
                # Display as DataFrame
                if isinstance(topic_words[0], tuple):
                    df = pd.DataFrame(topic_words, columns=['Word', 'Weight'])
                    df['Weight'] = df['Weight'].round(4)
                else:
                    df = pd.DataFrame({'Word': topic_words})
                
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Tab 2: Keywords
    with tab2:
        st.subheader("Overall Top Keywords")
        
        keywords = results['overall_keywords']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Keywords table
            kw_df = pd.DataFrame(keywords, columns=['Keyword', 'TF-IDF Score'])
            kw_df['TF-IDF Score'] = kw_df['TF-IDF Score'].round(4)
            st.dataframe(kw_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Word cloud
            st.markdown("**Word Cloud**")
            try:
                wordcloud_dict = {word: score for word, score in keywords}
                generate_wordcloud(wordcloud_dict)
            except Exception as e:
                st.error(f"Could not generate word cloud: {str(e)}")
    
    # Tab 3: Summary
    with tab3:
        st.subheader(f"Extractive Summary ({results['summary_method']})")
        
        st.info(results['summary'])
        
        st.markdown("---")
        st.subheader("Summary Statistics")
        
        stats = results['summary_stats']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Sentences", stats['original_sentences'])
        with col2:
            st.metric("Summary Sentences", stats['summary_sentences'])
        with col3:
            st.metric("Original Words", stats['original_words'])
        with col4:
            st.metric("Summary Words", stats['summary_words'])
        
        st.write(f"**Compression Ratio**: {stats['compression_ratio']:.2%}")
    
    # Tab 4: Evaluation
    with tab4:
        st.subheader("Evaluation Metrics")
        
        if results['method'] == 'LDA':
            evaluation = results['evaluation']
            
            # Coherence score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coherence Score (c_v)", f"{evaluation['coherence_score']:.4f}")
            with col2:
                st.metric("Topic Diversity", f"{evaluation['topic_diversity']:.4f}")
            with col3:
                st.metric("Number of Topics", evaluation['num_topics'])
            
            # Interpretation
            st.markdown("---")
            st.subheader("Interpretation")
            st.write(evaluation['interpretation'])
            
            # Per-topic coherence
            st.markdown("---")
            st.subheader("Coherence per Topic")
            topic_coh_df = pd.DataFrame({
                'Topic ID': range(len(evaluation['coherence_per_topic'])),
                'Coherence': evaluation['coherence_per_topic']
            })
            st.dataframe(topic_coh_df, use_container_width=True, hide_index=True)
            
        else:
            st.info("K-Means clustering evaluation metrics")
            stats = results.get('cluster_stats', {})
            
            stats_df = pd.DataFrame([
                {'Cluster': k, 'Documents': v['num_documents'], 'Percentage': f"{v['percentage']:.1f}%"}
                for k, v in stats.items()
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Feature statistics
        st.markdown("---")
        st.subheader("Feature Extraction Statistics")
        feat_stats = results['feature_stats']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", feat_stats['num_documents'])
        with col2:
            st.metric("Features", feat_stats['num_features'])
        with col3:
            st.metric("Sparsity", f"{feat_stats['sparsity']:.2%}")
    
    # Tab 5: Visualizations
    with tab5:
        st.subheader("Visualizations")
        
        # Topic keywords bar chart
        st.markdown("**Top Keywords per Topic**")
        
        topic_id_to_plot = st.selectbox(
            "Select Topic:",
            options=list(results['topics'].keys()),
            format_func=lambda x: f"Topic {x}: {results['theme_labels'].get(x, 'N/A')}"
        )
        
        plot_topic_keywords(results['topics'][topic_id_to_plot], topic_id_to_plot)
    
    # Tab 6: Limitations
    with tab6:
        st.subheader("⚠️ Limitations of Classical NLP")
        
        st.markdown("""
        This system demonstrates the **capabilities and constraints** of traditional NLP techniques:
        
        ### 🔴 Key Limitations
        
        #### 1. **No Semantic Understanding**
        - Bag-of-words and TF-IDF treat words as independent tokens
        - Cannot understand that "car" and "automobile" are synonyms
        - No comprehension of word order or sentence structure
        
        #### 2. **Context Blindness**
        - Cannot resolve polysemy (e.g., "bank" = financial institution vs. river bank)
        - Struggles with homonyms and contextual meanings
        - No understanding of discourse relationships
        
        #### 3. **Topic Coherence ≠ Meaningfulness**
        - High coherence scores measure statistical co-occurrence, not semantic coherence
        - Topics may group unrelated words that frequently appear together
        - Automatic theme labeling is unreliable and often nonsensical
        
        #### 4. **Preprocessing Sensitivity**
        - Results heavily depend on tokenization, stop-word lists, and lemmatization choices
        - Different preprocessing pipelines yield vastly different topics
        - No robustness to variations in text quality
        
        #### 5. **Summarization Limitations**
        - Extractive methods simply copy sentences—no paraphrasing or synthesis
        - Summaries may lack coherence and logical flow
        - Cannot combine information across sentences
        - Position bias (favors earlier sentences)
        
        #### 6. **No Multi-document Reasoning**
        - Cannot identify conflicting information across documents
        - No synthesis of information from multiple sources
        - Cannot track entities or relationships across documents
        
        ---
        
        ### ✅ Why This Motivates Agentic AI (Milestone-2)
        
        **Agentic AI systems** address these limitations by:
        - Using **transformer-based models** for semantic understanding
        - Employing **contextual embeddings** for word meaning disambiguation
        - Implementing **reasoning chains** for multi-step analysis
        - Leveraging **tool use** to access external knowledge
        - Performing **fact-checking** and contradiction detection
        - Generating **abstractive summaries** with true comprehension
        
        Traditional NLP provides interpretable, deterministic results but lacks the **semantic depth** 
        and **reasoning capabilities** needed for advanced research analysis.
        """)
        
        st.info("💡 **Academic Insight**: The coherence scores you see measure statistical patterns, "
                "not whether topics are semantically meaningful to humans!")


def generate_wordcloud(word_freq_dict):
    """Generate and display word cloud."""
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate_from_frequencies(word_freq_dict)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Word cloud generation failed: {str(e)}")


def plot_topic_keywords(topic_words, topic_id):
    """Plot topic keywords as bar chart."""
    try:
        if isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words[:10]]
            weights = [wt for _, wt in topic_words[:10]]
        else:
            words = topic_words[:10]
            weights = [1] * len(words)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(words, weights, color='steelblue')
        ax.set_xlabel('Weight')
        ax.set_title(f'Top Keywords - Topic {topic_id}')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Chart generation failed: {str(e)}")


if __name__ == "__main__":
    main()
