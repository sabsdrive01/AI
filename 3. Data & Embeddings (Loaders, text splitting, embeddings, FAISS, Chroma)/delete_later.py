import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add cells to the notebook
nb['cells'] = [
    # Title
    nbf.v4.new_markdown_cell("""# 🎓 LangChain Embeddings Tutorial

## Understanding Text Embeddings with LangChain & HuggingFace

### 📚 What You'll Learn
- How to use LangChain's embedding interface
- Converting text into numerical vectors
- Understanding embedding dimensions
- Working with HuggingFace models through LangChain

---

**Let's transform text into AI-readable numbers! 🚀**"""),

    # Cell 1: Installation & Setup
    nbf.v4.new_markdown_cell("""---

## 📦 Cell 1: Installation & Setup

### What We're Installing:
1. **`langchain-huggingface`** - LangChain's integration with HuggingFace models
2. **`sentence-transformers`** - The underlying embedding model library

### Why LangChain?
LangChain provides a **unified interface** for working with different embedding models. Whether you use HuggingFace, OpenAI, or Cohere, the code looks similar!

**Benefit:** Easy to switch between embedding providers without rewriting code."""),

    nbf.v4.new_code_cell("""# Cell 1: Install required packages
print("📦 Installing LangChain and HuggingFace embeddings...\\n")

# Install the packages
!pip install -q langchain-huggingface sentence-transformers

print("✅ Installation complete!\\n")

# Verify installation
import langchain_huggingface
print(f"✅ langchain-huggingface version: {langchain_huggingface.__version__}")

print("\\n💡 Ready to create embeddings!")"""),

    # Cell 2: Import and Initialize
    nbf.v4.new_markdown_cell("""---

## 🔧 Cell 2: Import and Initialize the Embedding Model

### What's Happening:
1. **Import** the HuggingFaceEmbeddings class from LangChain
2. **Initialize** the model with a specific HuggingFace model name
3. **Download** the model weights (happens automatically on first run)

### The Model: `all-MiniLM-L6-v2`
- **Size:** ~80MB
- **Speed:** Very fast (~1000 sentences/second)
- **Dimensions:** 384 (each text becomes 384 numbers)
- **Quality:** Excellent for most applications
- **Use Case:** General-purpose semantic similarity

### Technical Details:
- This model is trained on 1 billion+ sentence pairs
- It understands semantic meaning, not just keywords
- Optimized for sentence-level embeddings"""),

    nbf.v4.new_code_cell("""# Cell 2: Import and initialize the embedding model
from langchain_huggingface import HuggingFaceEmbeddings

print("🔄 Loading HuggingFace embedding model...\\n")

# Initialize the embeddings model
# This will download the model on first run (~80MB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("✅ Model loaded successfully!\\n")

print("📊 Model Information:")
print(f"   • Model: all-MiniLM-L6-v2")
print(f"   • Provider: HuggingFace (via LangChain)")
print(f"   • Embedding Dimension: 384")
print(f"   • Max Sequence Length: 256 tokens")

print("\\n💡 The model is now ready to convert text into vectors!")"""),

    # Cell 3: Create Sample Text
    nbf.v4.new_markdown_cell("""---

## 📝 Cell 3: Prepare Sample Text

### What is an Embedding?
An **embedding** is a numerical representation of text that captures its meaning.

### The Magic:
```
Text: "This is a test document."
        ↓
Embedding Model
        ↓
Vector: [0.234, -0.456, 0.789, ..., 0.123]
         (384 numbers total)
```

### Why 384 Numbers?
- Each number represents a different "feature" or "aspect" of meaning
- Together, they encode: grammar, semantics, context, and relationships
- Similar texts → similar vectors (close in 384-dimensional space)

### Real-World Analogy:
Think of GPS coordinates:
- **Text:** "Eiffel Tower"
- **2D Coordinates:** (48.8584°N, 2.2945°E)
- **Embeddings:** Like coordinates, but in 384-dimensional "meaning space"!"""),

    nbf.v4.new_code_cell("""# Cell 3: Prepare sample text
text = "This is a test document."

print("📝 Sample Text Prepared\\n")
print("=" * 60)
print(f"Text: \\"{text}\\"")
print("=" * 60)

print("\\n🎯 Next Step: Convert this text into an embedding vector")
print("\\n💡 Properties of the text:")
print(f"   • Length: {len(text)} characters")
print(f"   • Words: {len(text.split())} words")
print(f"   • Will become: 384 numerical values")

print("\\n📊 What the embedding will capture:")
print("   ✓ Semantic meaning (what it's about)")
print("   ✓ Context (formal/informal, topic)")
print("   ✓ Relationships (similar to other 'test document' phrases)")"""),

    # Cell 4: Generate Embedding
    nbf.v4.new_markdown_cell("""---

## 🧮 Cell 4: Generate the Embedding Vector

### The Process:
1. **Tokenization:** Split text into tokens (words/subwords)
2. **Model Processing:** Neural network processes the tokens
3. **Vector Creation:** Output is a 384-dimensional vector
4. **Result:** A list of floating-point numbers

### `embed_query()` Method:
- **Purpose:** Convert a single piece of text into an embedding
- **Input:** String (your text)
- **Output:** List of 384 floating-point numbers
- **Speed:** ~1-5 milliseconds per text

### Understanding the Output:
Each number in the vector:
- Ranges typically from **-1 to +1** (normalized)
- Represents activation of a "semantic feature"
- Individually hard to interpret, but collectively powerful

### Technical Note:
The same text will **always** produce the same embedding (deterministic)."""),

    nbf.v4.new_code_cell("""# Cell 4: Generate the embedding
print("🔄 Generating embedding for the text...\\n")

# Call embed_query() to convert text to vector
query_result = embeddings.embed_query(text)

print("✅ Embedding generated successfully!\\n")

print("📊 Embedding Details:")
print("=" * 60)
print(f"Type: {type(query_result)}")
print(f"Length (dimensions): {len(query_result)}")
print(f"Data type: {type(query_result[0])}")

print("\\n🔢 Statistical Summary:")
import numpy as np
arr = np.array(query_result)
print(f"   • Min value: {arr.min():.6f}")
print(f"   • Max value: {arr.max():.6f}")
print(f"   • Mean value: {arr.mean():.6f}")
print(f"   • Standard deviation: {arr.std():.6f}")

print("\\n📈 Sample Values (first 10):")
for i, val in enumerate(query_result[:10]):
    print(f"   Dimension {i}: {val:.6f}")

print("\\n💡 This vector now represents the meaning of our text!")"""),

    # Cell 5: Visualize and Understand
    nbf.v4.new_markdown_cell("""---

## 🔍 Cell 5: Visualize and Understand the Embedding

### What We're Doing:
1. **Display** a preview of the vector (first 100 characters)
2. **Explain** what these numbers mean
3. **Compare** different texts to show how embeddings work

### Key Insight:
The numbers themselves aren't meaningful individually, but when you **compare** embeddings using mathematical operations (like cosine similarity), you can measure how similar two texts are!

### Practical Applications:
- **Search:** Find documents similar to a query
- **Clustering:** Group similar texts together
- **Classification:** Categorize text by meaning
- **Recommendation:** Suggest related content"""),

    nbf.v4.new_code_cell("""# Cell 5: Display and analyze the embedding

print("🎨 Embedding Visualization\\n")
print("=" * 60)

# Show preview (first 100 characters)
print("📋 Vector Preview (first 100 characters):")
print(str(query_result)[:100] + "...")

print("\\n" + "=" * 60)

# Full statistics
print("\\n📊 Complete Vector Analysis:\\n")

print(f"Original Text: \\"{text}\\"")
print(f"Vector Length: {len(query_result)} dimensions")
print(f"Total Storage: {len(query_result) * 4} bytes (4 bytes per float32)")

# Show distribution of values
import numpy as np
arr = np.array(query_result)

print("\\n📈 Value Distribution:")
print(f"   • Values > 0: {np.sum(arr > 0)} ({np.sum(arr > 0)/len(arr)*100:.1f}%)")
print(f"   • Values < 0: {np.sum(arr < 0)} ({np.sum(arr < 0)/len(arr)*100:.1f}%)")
print(f"   • Values ≈ 0: {np.sum(np.abs(arr) < 0.01)} ({np.sum(np.abs(arr) < 0.01)/len(arr)*100:.1f}%)")

# Visualize first 20 dimensions
print("\\n📊 First 20 Dimensions Visualization:")
print("\\nDim  Value      Bar")
print("-" * 40)
for i in range(20):
    val = query_result[i]
    bar_length = int(abs(val) * 20)
    bar = "█" * bar_length
    sign = "+" if val >= 0 else "-"
    print(f"{i:3d}  {val:+.4f}  {sign}{bar}")

print("\\n💡 Each dimension captures a different aspect of meaning!")
print("💡 Together, they form a unique 'fingerprint' of the text's semantics")"""),

    # Bonus Cell: Compare Multiple Texts
    nbf.v4.new_markdown_cell("""---

## 🎯 Bonus: Compare Multiple Text Embeddings

Let's see how embeddings capture similarity between different texts!"""),

    nbf.v4.new_code_cell("""# Bonus: Compare embeddings of different texts
print("🔬 Comparing Multiple Text Embeddings\\n")
print("=" * 60)

# Create diverse sample texts
texts = [
    "This is a test document.",           # Original
    "This is a test file.",               # Similar (synonym)
    "The weather is sunny today.",        # Different topic
    "Machine learning is fascinating.",   # Different topic
]

print("📝 Sample Texts:\\n")
for i, t in enumerate(texts, 1):
    print(f"{i}. \\"{t}\\"")

# Generate embeddings for all texts
print("\\n🔄 Generating embeddings...")
all_embeddings = [embeddings.embed_query(t) for t in texts]
print("✅ All embeddings generated!\\n")

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compare text 1 with all others
print("📊 Similarity Scores (comparing Text 1 with others):\\n")
base_emb = all_embeddings[0]

for i, (text, emb) in enumerate(zip(texts, all_embeddings), 1):
    similarity = cosine_similarity(base_emb, emb)
    
    # Create visual bar
    bar_length = int(similarity * 30)
    bar = "█" * bar_length
    
    # Status indicator
    if similarity > 0.8:
        status = "🟢 Very Similar"
    elif similarity > 0.5:
        status = "🟡 Somewhat Similar"
    else:
        status = "🔴 Different"
    
    print(f"Text {i}: {similarity:.3f} {bar} {status}")
    if i == 1:
        print("         (baseline - comparing to itself)")
    print()

print("\\n💡 Key Insights:")
print("   • Text 1 vs Text 2: HIGH similarity (synonyms: 'document' ≈ 'file')")
print("   • Text 1 vs Text 3: LOW similarity (different topics)")
print("   • Text 1 vs Text 4: LOW similarity (unrelated content)")
print("\\n✨ Embeddings successfully capture semantic meaning!")"""),

    # Summary
    nbf.v4.new_markdown_cell("""---

## 🎓 Summary: What You Learned

### Step-by-Step Recap:

1. **Cell 1:** Installed LangChain and HuggingFace integration
2. **Cell 2:** Loaded the `all-MiniLM-L6-v2` embedding model
3. **Cell 3:** Prepared sample text for embedding
4. **Cell 4:** Generated a 384-dimensional embedding vector
5. **Cell 5:** Analyzed and visualized the embedding

### Key Concepts:

✅ **Embeddings** = Numerical representations of text meaning  
✅ **384 dimensions** = 384 numbers that encode semantics  
✅ **Cosine similarity** = Measure how similar two embeddings are  
✅ **LangChain** = Unified interface for different embedding models  

### Practical Applications:

🔍 **Semantic Search** - Find relevant documents by meaning  
📊 **Clustering** - Group similar texts automatically  
🤖 **RAG Systems** - Give LLMs access to custom knowledge  
💬 **Chatbots** - Understand user intent, not just keywords  

---

## 🚀 Next Steps

1. **Experiment:** Try embedding different types of text (short, long, technical, casual)
2. **Compare Models:** Test other HuggingFace models (e.g., `all-mpnet-base-v2`)
3. **Build Something:** Create a simple document search system
4. **Learn More:** Explore vector databases (FAISS, ChromaDB, Pinecone)

---

### 📚 Further Reading

- [LangChain Embeddings Docs](https://python.langchain.com/docs/integrations/text_embedding/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Understanding Vector Embeddings](https://www.pinecone.io/learn/vector-embeddings/)

**Happy embedding! 🎉**"""),
]

# Save the notebook
with open('langchain_embeddings_tutorial.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Notebook created: langchain_embeddings_tutorial.ipynb")