# Quick Reference Guide - Testing the System

## ✅ Fixed Issues

1. **TF-IDF parameters** - Now adaptive for 1-3 documents
2. **Input validation** - Checks for sufficient content after preprocessing
3. **Error messages** - Helpful guidance when input is too short

## 🚀 How to Test Successfully

### Option 1: Use the Sample File

1. In the Streamlit interface, select **"Upload Documents (PDF/Text)"**
2. Click "Browse files"
3. Navigate to: `Project1-GenAI/data/sample_documents/sample_ml_nlp.txt`
4. Upload the file
5. Click "🚀 Run Analysis"

### Option 2: Copy-Paste Longer Text

For "Enter Text Directly", you need **at least 1-2 full paragraphs** (100+ words).

**✅ Good Example** (enough content):
```
Machine learning algorithms have revolutionized natural language processing. Modern deep learning models can process and understand text with remarkable accuracy. Transformer architectures use self-attention mechanisms to capture contextual relationships between words. These models are pre-trained on massive text corpora and can be fine-tuned for specific tasks like classification, translation, and summarization. Recent advances in large language models demonstrate impressive capabilities in understanding and generating human language. However, challenges remain in handling ambiguity, sarcasm, and cultural nuances.
```

**❌ Bad Example** (too short - will fail):
```
Products:
- Modular pipelines  
- TF-IDF code
- Clean preprocessing
```

## 📝 Minimum Requirements

After stop-word removal and preprocessing, the system needs:
- **Minimum**: 10 tokens (enforced)
- **Recommended**: 50+ tokens (1-2 paragraphs)
- **Ideal**: 200+ tokens (multiple paragraphs or documents)

## 🔍 What Happens During Preprocessing

Common words removed as "stop words":
- Articles: a, an, the
- Prepositions: of, in, at, to, from
- Conjunctions: and, or, but
- Common verbs: is, are, was, were, been, being

**Result**: Very short bullet points often become empty!

## 💡 Tips for Success

1. **Upload actual research papers** (best results)
2. **Paste full paragraphs**, not just keywords
3. **Combine multiple short snippets** into one longer text
4. **Check the sample file** for an example of good input
5. **Use K-Means** if you want to avoid LDA entirely

## 📂 Test with Sample File

Sample file location:
```
/Users/pankajbaid/pankajbaid567/Project1-GenAI/data/sample_documents/sample_ml_nlp.txt
```

This file contains 8 paragraphs about NLP (300+ words) - perfect for testing!
