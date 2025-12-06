# **1. Introduction to LangChain Core**

**Purpose:** Give foundational understanding of what LangChain Core is and why it matters.

### **Sub-topics**

1. **What is LangChain Core?**
    - Framework for building applications powered by LLMs
    - Core modules: prompts, models, chains, and runnables
2. **Why LangChain is useful for AI applications**
    - Standardization
    - Reusability
    - Scalability
    - Modularity
3. **Key Concepts Overview**
    - Prompts
    - Chains
    - Tools & Runnables
    - Outputs & parsers

---

# **2. ChatPromptTemplate**

**Purpose:** Explain how prompts are structured and generated dynamically.

### **Sub-topics**

1. **Definition & Purpose**
    - Template for creating prompts with placeholders
    - Helps maintain structure and formatting
2. **Types of Prompt Templates**
    - System message
    - Human message
    - AI message
    - Mixed multi-turn templates
3. **Dynamic Prompting**
    - Using variables: `{context}`, `{question}`
    - Setting default values
4. **Best Practices**
    - Clear instructions
    - Guardrails
    - Formatting for better LLM responses

---

# **3. LLMChain**

**Purpose:** Show how prompts + LLMs are combined to create structured workflows.

### **Sub-topics**

1. **What Is LLMChain?**
    - A sequence: PROMPT → MODEL → OUTPUT
    - Core abstraction for single-step pipelines
2. **Components of LLMChain**
    - PromptTemplate
    - LLM (e.g., GPT, Gemini, Claude)
    - Output parser
3. **Use Cases**
    - Question answering
    - Data extraction
    - Text generation
    - Document understanding
4. **Advantages**
    - Reusable
    - Configurable
    - Integrates with other chains

---

# **4. Runnable Interface**

**Purpose:** Explain LangChain’s newer execution engine for flexible pipelines.

### **Sub-topics**

1. **What is Runnable?**
    - Universal abstraction for all workflow components
    - Everything becomes a “runnable” (prompt, chain, model, function)
2. **Types of Runnables**
    - `RunnableMap`
    - `RunnableSequence`
    - `RunnableLambda`
    - `RunnableBranch`
3. **Async vs Sync Execution**
    - Faster execution
    - Parallel operations
4. **Composability**
    - Combine prompt → model → post-processing easily

---

# **5. Prompt Formatting & Best Practices**

**Purpose:** Ensure employees follow good standards for stable, accurate model outputs.

### **Sub-topics**

1. **Formatting Techniques**
    - Using `f-string-like` placeholders
    - Multi-line prompts
    - Role-based prompting (system, user, assistant)
2. **Prompt Structuring**
    - Task → Context → Rules → Output Format
    - Use JSON schema for structured outputs
3. **Reducing Hallucinations**
    - Grounding via context
    - Clear instructions
    - Using few-shot examples
4. **Evaluation & Testing**
    - Prompt testing through prompt benches
    - Validating deterministic outputs
    - Logging prompt changes