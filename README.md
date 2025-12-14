# 🚀 Agentic Text-to-SQL System v4.0

> **Query ANY database in natural language. No SQL knowledge required.**

Transform your natural language questions into precise SQL queries using AI agents powered by Ollama. Perfect for researchers, data analysts, and developers who want to interact with databases without writing SQL.

---

## ✨ Features

### 🎯 Smart Database Detection
- **Auto-detect** the right database from your question keywords
- **Fuzzy matching** handles typos and name variations (`singer_concert` → `concert_singer`)
- **Manual override** when you know the exact database

### 🧠 AI-Powered Query Generation
- **Manager Agent**: Understands your question and creates an execution plan
- **Coder Agent**: Generates syntactically correct SQL from the plan
- **Fallback Mode**: Works offline with simulated agents

### 📊 Works with 166+ Databases
- Support for the entire **Spider dataset** (Spider-Web research benchmark)
- Automatically discovers and catalogs all available databases
- Rich metadata: tables, columns, types, foreign keys, primary keys

### 🛡️ Robust Error Handling
- Graceful fallbacks when LLM unavailable
- Clear error messages with suggestions
- Fully qualified column names to prevent SQL ambiguity

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.8+
python --version

# Install dependencies
pip install langchain langchain-community pandas

# Install Ollama (for local LLM)
# Download from: https://ollama.ai/download
```

### Install Ollama Models

```bash
# Start Ollama server
ollama serve

# In a new terminal, pull models
ollama pull qwen2.5:7b      # Manager agent (reasoning)
ollama pull sqlcoder:7b     # Coder agent (SQL generation)
```

### Run the System

```python
from agentic_pipeline import interactive_demo, run_agentic_pipeline

# Option 1: Interactive Mode (Recommended)
interactive_demo()

# Option 2: Direct Query with Auto-Detection
result = run_agentic_pipeline(question="How many singers are there?")
print(result["final_answer"])

# Option 3: Explicit Database
result = run_agentic_pipeline(
    db_id="concert_singer",
    question="List all singers"
)
```

---

## 📖 Usage Examples

### Example 1: Auto-Detect Database

```python
from agentic_pipeline import run_agentic_pipeline

# Ask a question - system finds the right database!
result = run_agentic_pipeline(
    question="How many flights are there?",
    use_llm=True,
    auto_detect_db=True
)

print(result["final_answer"])
# Output: "Answer: 1234"
```

### Example 2: Specify Database Explicitly

```python
result = run_agentic_pipeline(
    db_id="aircraft",
    question="List all pilots with their aircraft"
)

print(result["generated_sql"])
# Output: "SELECT pilot.Name, aircraft.Aircraft FROM pilot JOIN aircraft..."
```

### Example 3: Search for Databases

```python
from agentic_pipeline import db_discovery

# Find databases about singers
matches = db_discovery.search_databases("singer")
print(matches)
# Output: ['concert_singer', 'artist_album', ...]

# List all available databases
db_discovery.print_all_databases()
```

### Example 4: Smart DB Matching

```python
from agentic_pipeline import db_discovery

# AI finds best matching database based on question
best_db = db_discovery.get_best_matching_db(
    "Show me artists from the UK"
)
print(best_db)
# Output: 'music4'
```

### Example 5: Handle Typos Gracefully

```python
# Typo: singer_concert doesn't exist
try:
    result = run_agentic_pipeline(
        db_id="singer_concert",  # Wrong!
        question="How many singers?"
    )
except ValueError as e:
    print(e)
    # Output: "❌ Database 'singer_concert' not found!
    #          Did you mean: concert_singer, artist_concert?"
```

---

## 🎮 Interactive Commands

Start the interactive demo:

```python
from agentic_pipeline import interactive_demo
interactive_demo()
```

Available commands:

| Command | Description | Example |
|---------|-------------|---------|
| `list` | Show all available databases | `list` |
| `search <keyword>` | Search databases by keyword | `search singer` |
| `query` | Ask a question interactively | `query` → "How many singers?" |
| `quit` | Exit the interactive mode | `quit` |

**Interactive Menu for Queries:**

```
> query
Question: How many singers?

Options:
  1. Specify database ID manually
  2. Auto-detect database from question

Choice (1 or 2, default=2): 2
[Auto-detecting best database...]
✓ Selected: concert_singer
...
```

---

## 🏗️ Architecture

### Pipeline Flow

```
User Question
    ↓
[Validation] Check if question is valid
    ↓
[DB Detection] Find relevant database(s)
    ↓
[Schema Retrieval] Load table/column metadata
    ↓
[Manager Agent] 🧠 Analyze question → Create plan
    ↓
[Coder Agent] 💻 Generate SQL from plan
    ↓
[Executor] 🚀 Run SQL on database
    ↓
[Formatter] 📊 Format results for output
    ↓
Answer to User
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **DatabaseDiscovery** | Find & search databases, auto-detect best match |
| **Validation Layer** | Clean input, suggest close matches on typos |
| **Manager Agent** | LLM that understands questions & plans execution |
| **Coder Agent** | LLM that generates SQL from execution plan |
| **SQL Executor** | Runs generated SQL on SQLite databases |
| **Interactive Shell** | Menu-driven CLI for exploration |

---

## 🔧 Configuration

Edit `agentic_pipeline.py`:

```python
# 1. Set project root
PROJECT_ROOT = r"C:\path\to\your\project"

# 2. Configure LLM models
manager_model = "qwen2.5:7b"      # Change for different reasoning model
coder_model = "sqlcoder:7b"       # Change for different SQL generation model

# 3. Ollama server address
base_url = "http://localhost:11434"  # Change if running on different host
```

---

## 📊 Supported Databases

The system supports **166 databases** from the Spider dataset:

| Category | Examples |
|----------|----------|
| **Business** | airline, aircraft, employee_hire_evaluation, departmentmanagement |
| **Academic** | student, dorm, school_player, college_1 |
| **Finance** | bank, insurance_fnol, customersandinvoices |
| **Entertainment** | concert_singer, movie_1, tvshow, music_4 |
| **Misc** | pets1, restaurant1, network1, game_1 |

Use `db_discovery.get_all_databases()` to see your available databases.

---

## 🎓 How It Works

### Step 1: Understanding the Question
The **Manager Agent** reads your question and:
- Extracts the intent (count, list, compare, etc.)
- Identifies relevant tables and columns
- Determines what joins and filters are needed
- Creates a step-by-step execution plan

**Example:**
```
Question: "How many singers performed in 2023?"

Manager's Analysis:
- INTENT: Count singers from a specific year
- RELEVANT_TABLES: singer, concert
- RELEVANT_COLUMNS: singer.name, concert.year
- JOINS_NEEDED: Yes (singer ← concert)
- FILTERS: WHERE concert.year = 2023
- AGGREGATION: COUNT
```

### Step 2: Generating SQL
The **Coder Agent** reads the plan and:
- Validates against the database schema
- Uses fully qualified column names (e.g., `singer.name`)
- Generates syntactically correct SQL
- Ensures tables/columns exist in the schema

**Example Output:**
```sql
SELECT COUNT(DISTINCT singer.singer_id) 
FROM singer 
JOIN concert ON singer.singer_id = concert.singer_id 
WHERE concert.year = 2023;
```

### Step 3: Executing & Formatting
- Run the SQL on the actual database
- Fetch results
- Format as human-readable answer

**Output:**
```
Answer: 142
```

---

## 🤖 Offline Mode (No Ollama)

If Ollama is unavailable, the system uses **simulated agents**:

```python
# Ollama not running? No problem!
result = run_agentic_pipeline(
    question="How many singers?",
    use_llm=False  # Use simulated agents
)
```

**Limitations of simulated mode:**
- ✅ Basic COUNT, SUM, SELECT queries work
- ❌ Complex joins and aggregations may fail
- ❌ No advanced reasoning

For production use, **always use real LLMs with Ollama**.

---

## 🐛 Troubleshooting

### Issue: "Database 'X' not found"

**Solution:**
```python
# See available databases
db_discovery.print_all_databases()

# Or search for the right name
matches = db_discovery.search_databases("your_keyword")
```

### Issue: "Ollama not available"

**Solution:**
```bash
# Start Ollama server
ollama serve

# In another terminal, check models
ollama list
```

### Issue: "Cannot connect to Ollama"

**Solution:**
```python
# Check if Ollama is running on correct port
# Default: http://localhost:11434

# If running on different host:
base_url = "http://your-host:11434"
```

### Issue: "Module not found: langchain"

**Solution:**
```bash
pip install langchain langchain-community
```

---

## 📈 Performance Tips

### 1. Reduce Schema Size
```python
# Only pass relevant schema to agents (reduces LLM context)
relevant_schema = get_schema_text("concert_singer")
```

### 2. Use Larger Models for Complex Queries
```python
# For complex multi-table joins, use larger model
manager_model = "qwen2.5:14b"  # Bigger = better reasoning
```

### 3. Enable Caching
```python
# The system caches schema_lookup and db_cache automatically
# No action needed - it's already optimized!
```

---

## 🔐 Security Notes

⚠️ **Important:** This system is designed for **research and controlled environments**.

### Limitations:
- ❌ No SQL injection prevention (uses LLM-generated queries)
- ❌ No access control or row-level security
- ❌ No query auditing or logging
- ❌ All users can query all databases

### Safe Usage:
- ✅ Use only with research/test databases (not production)
- ✅ Restrict database credentials to read-only
- ✅ Run behind a secure API layer for production
- ✅ Implement query review before execution

---

## 🧪 Testing

### Test Auto-Detection

```python
from agentic_pipeline import db_discovery

test_cases = [
    ("How many singers?", "concert_singer"),
    ("List all flights", "flight1"),
    ("Show customers and orders", "customersandinvoices"),
]

for question, expected_db in test_cases:
    best_db = db_discovery.get_best_matching_db(question)
    print(f"Q: {question}")
    print(f"Expected: {expected_db}, Got: {best_db}, Status: {'✓' if best_db == expected_db else '✗'}")
```

### Test with Different LLM Settings

```python
# Test with real LLM
result_real = run_agentic_pipeline(question="How many singers?", use_llm=True)

# Test with simulated agent
result_sim = run_agentic_pipeline(question="How many singers?", use_llm=False)

# Compare results
print(f"Real LLM: {result_real['final_answer']}")
print(f"Simulated: {result_sim['final_answer']}")
```

---

## 📚 Project Structure

```
NLP_Project/
├── agentic_pipeline.py           # Main system (this file)
├── spider_dataset/
│   └── spider/
│       ├── train_spider.json     # Training examples
│       ├── tables.json           # Schema metadata
│       ├── dev.json              # Dev set
│       └── database/             # 166 SQLite databases
│           ├── aircraft/
│           ├── concert_singer/
│           └── ... (164 more)
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

---

## 📋 Requirements

```
Python 3.8+
langchain>=0.1.0
langchain-community>=0.0.1
pandas>=1.0.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🎯 Use Cases

### 1. Data Exploration
```python
# Explore a database you're unfamiliar with
result = run_agentic_pipeline(
    db_id="aircraft",
    question="Show me the schema overview"
)
```

### 2. Quick Analytics
```python
# Get instant insights without writing SQL
result = run_agentic_pipeline(
    question="How many flights are delayed?"
)
```

### 3. Research & Benchmarking
```python
# Test how well the system handles different query types
questions = [
    "Count singers by country",
    "Find flights between two airports",
    "List customers with total spending > $1000"
]

for q in questions:
    result = run_agentic_pipeline(question=q)
    print(f"Q: {q}\nSQL: {result['generated_sql']}\n")
```

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Support for more database types (MySQL, PostgreSQL, etc.)
- [ ] Query result visualization
- [ ] Multi-language question support
- [ ] Query optimization suggestions
- [ ] Caching of generated queries
- [ ] Web UI for interactive exploration

---

## 📄 Citation

If you use this system in your research, please cite:

```bibtex
@software{agentic_text_to_sql_2025,
  author = {Your Name},
  title = {Agentic Text-to-SQL System v4.0},
  url = {https://github.com/yourusername/agentic-text-to-sql},
  year = {2025}
}
```

---

## 📞 Support

### Getting Help

1. **Check the [Troubleshooting](#-troubleshooting) section**
2. **Search existing [Issues](https://github.com/yourusername/agentic-text-to-sql/issues)**
3. **Try with `use_llm=False` to isolate LLM issues**
4. **Check Ollama is running: `ollama serve`**

### Report Issues

Please include:
- Your question
- The database you used
- The error message (full stack trace)
- Your Ollama model versions
- Python version

---

## 📜 License

MIT License - feel free to use this in your projects!

---

## 🌟 Highlights

- **166+ Databases** - Comprehensive Spider dataset support
- **Intelligent Matching** - Fuzzy matching for typos
- **Offline Ready** - Works without Ollama (with limitations)
- **Transparent** - See the entire reasoning chain
- **Type Safe** - Full type hints throughout
- **Production Ready** - Comprehensive error handling

---

## 🚀 What's Next?

### v4.1 (Planned)
- [ ] Multi-step reasoning for complex queries
- [ ] Query result caching
- [ ] Performance optimization

### v5.0 (Planned)
- [ ] Web UI
- [ ] Advanced visualizations
- [ ] PostgreSQL/MySQL support

---

**Built with ❤️ for data exploration & research**

*Make data accessible. Make SQL optional. Make insights instant.*

---

<div align="center">

⭐ **Found this useful? Star it on GitHub!**



</div>
