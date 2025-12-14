import os
import json
import sqlite3
import re
import pandas as pd
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict
import subprocess

# ============================================================
# PART 0: OLLAMA MODEL DETECTION
# ============================================================

def get_available_ollama_models() -> List[str]:
    """Get list of models already downloaded in Ollama."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]  # e.g., "qwen2.5:7b"
                        models.append(model_name)
            return models
    except:
        pass
    return []


def find_suitable_manager_model(available_models: List[str]) -> Optional[str]:
    """Find a suitable model for manager agent from what you have."""
    preferred = ['qwen2.5:7b', 'qwen2.5:14b', 'mistral:7b', 'neural-chat:7b', 'llama2:7b']
    
    for model in preferred:
        if model in available_models:
            return model
    
    if available_models:
        return available_models[0]
    
    return None


def find_suitable_coder_model(available_models: List[str]) -> Optional[str]:
    """Find a suitable model for SQL coder from what you have."""
    preferred = ['sqlcoder:7b', 'sqlcoder:15b', 'starcoder:7b', 'mistral:7b', 'qwen2.5:7b']
    
    for model in preferred:
        if model in available_models:
            return model
    
    if available_models:
        return available_models[0]
    
    return None


# ============================================================
# PART 0.5: INPUT VALIDATION UTILITIES - PROPER TYPE HINTS
# ============================================================

def validate_string(value: Any, param_name: str, default: Optional[str] = None, allow_none: bool = False) -> str:
    """Validate and clean string input. Always returns str (never None)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_none:
            return ""
        if default is None:
            raise ValueError(f"{param_name} cannot be None or empty!")
        return default
    if not isinstance(value, str):
        return str(value).strip()
    return value.strip()


def validate_db_id(db_id: Optional[str], schema_lookup: Dict[str, Any]) -> str:
    """Validate database ID exists in schema lookup. Returns str."""
    if db_id is None or (isinstance(db_id, str) and not db_id.strip()):
        raise ValueError("❌ Database ID required! Use db_discovery.get_all_databases()")
    
    db_id_str = validate_string(db_id, "db_id")
    
    if db_id_str and db_id_str not in schema_lookup:
        available = list(schema_lookup.keys())[:5]
        raise ValueError(f"❌ Database '{db_id_str}' not found!\n   Available: {available}")
    
    return db_id_str


def validate_question(question: Optional[str]) -> str:
    """Validate question is not empty. Returns str."""
    if question is None or (isinstance(question, str) and not question.strip()):
        raise ValueError("❌ Question cannot be empty!")
    
    return validate_string(question, "question")


def validate_db_list(db_ids: Any, schema_lookup: Dict[str, Any]) -> List[str]:
    """Validate database list. Returns List[str]."""
    if db_ids is None or (isinstance(db_ids, list) and len(db_ids) == 0):
        raise ValueError("❌ At least one database required!")
    
    if not isinstance(db_ids, list):
        db_ids = [db_ids]
    
    valid_dbs: List[str] = []
    invalid_dbs: List[str] = []
    
    for db in db_ids:
        db_clean = validate_string(db, "db_id", allow_none=True)
        if db_clean:
            if db_clean in schema_lookup:
                valid_dbs.append(db_clean)
            else:
                invalid_dbs.append(db_clean)
    
    if invalid_dbs:
        print(f"\n⚠️  Warning: Invalid databases: {invalid_dbs}")
    
    if not valid_dbs:
        raise ValueError(f"❌ No valid databases found!")
    
    return valid_dbs


def clean_sql_output(sql: str) -> str:
    """Clean LLM output: remove tokenizer artifacts, markdown, etc."""
    # Remove LLM special tokens
    sql = sql.replace('<s>', '').replace('</s>', '').replace('<unk>', '')
    sql = sql.replace('[BOS]', '').replace('[EOS]', '')
    
    # Remove markdown code blocks
    if '```' in sql:
        lines = [l for l in sql.split('\n') if not l.strip().startswith('```')]
        sql = '\n'.join(lines).strip()
    
    # Remove SQL comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    
    # Remove leading/trailing whitespace
    sql = sql.strip()
    
    return sql

# ============================================================
# PART 1: CONFIGURATION & PATHS
# ============================================================

PROJECT_ROOT: str = r"C:\Users\mrafi\Downloads\NLP_Project\NLP_Project"

SPIDER_PATH: str = os.path.join(PROJECT_ROOT, "spider_dataset", "spider")
SPIDER_TRAIN: str = os.path.join(SPIDER_PATH, "train_spider.json")
SPIDER_DEV: str = os.path.join(SPIDER_PATH, "dev.json")
SPIDER_TABLES: str = os.path.join(SPIDER_PATH, "tables.json")
SPIDER_DB_DIR: str = os.path.join(SPIDER_PATH, "database")

print("=" * 60)
print("SYSTEM INITIALIZATION")
print("=" * 60)
print(f"Spider train exists: {os.path.exists(SPIDER_TRAIN)}")
print(f"Spider tables exists: {os.path.exists(SPIDER_TABLES)}")
print(f"Spider DB dir exists: {os.path.exists(SPIDER_DB_DIR)}")

# ============================================================
# PART 2: LOAD DATASETS
# ============================================================

with open(SPIDER_TRAIN, 'r', encoding='utf-8') as f:
    spider_train_data: List[Dict[str, Any]] = json.load(f)

with open(SPIDER_TABLES, 'r', encoding='utf-8') as f:
    spider_tables: List[Dict[str, Any]] = json.load(f)

schema_lookup: Dict[str, Dict[str, Any]] = {}
for table_info in spider_tables:
    db_id = table_info['db_id']
    schema_lookup[db_id] = table_info

print(f"\nLoaded {len(spider_train_data)} training examples")
print(f"Loaded {len(schema_lookup)} database schemas")

# ============================================================
# PART 3: DATABASE DISCOVERY
# ============================================================

class DatabaseDiscovery:
    """Automatically discover and manage available databases."""
    
    def __init__(self, schema_lookup: Dict[str, Dict[str, Any]], db_dir: str) -> None:
        self.schema_lookup: Dict[str, Dict[str, Any]] = schema_lookup
        self.db_dir: str = db_dir
        self.db_cache: Dict[str, Dict[str, Any]] = self._discover_databases()
    
    def _discover_databases(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available databases and their metadata."""
        db_info: Dict[str, Dict[str, Any]] = {}
        
        for db_id, schema_info in self.schema_lookup.items():
            db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            
            db_info[db_id] = {
                "db_id": db_id,
                "exists": os.path.exists(db_path),
                "path": db_path,
                "table_count": len(schema_info['table_names_original']),
                "column_count": len(schema_info['column_names_original']),
                "tables": schema_info['table_names_original'],
                "columns": [col[1] for col in schema_info['column_names_original']],
            }
        
        return db_info
    
    def get_all_databases(self, filter_exists: bool = True) -> List[str]:
        """Get all available database IDs."""
        if filter_exists:
            return [db for db, info in self.db_cache.items() if info['exists']]
        return list(self.db_cache.keys())
    
    def search_databases(self, keyword: str) -> List[str]:
        """Search for databases matching a keyword."""
        keyword_clean = validate_string(keyword, "keyword", allow_none=True)
        if not keyword_clean:
            return []
        
        keyword_lower = keyword_clean.lower()
        matches: List[str] = []
        
        for db_id, info in self.db_cache.items():
            if not info['exists']:
                continue
            
            if keyword_lower in db_id.lower():
                matches.append(db_id)
            elif any(keyword_lower in table.lower() for table in info['tables']):
                matches.append(db_id)
            elif any(keyword_lower in col.lower() for col in info['columns']):
                matches.append(db_id)
        
        return matches
    
    def get_databases_with_table(self, table_name: str) -> List[str]:
        """Find all databases containing a specific table."""
        table_clean = validate_string(table_name, "table_name", allow_none=True)
        if not table_clean:
            return []
        
        table_lower = table_clean.lower()
        matches: List[str] = []
        
        for db_id, info in self.db_cache.items():
            if not info['exists']:
                continue
            
            if any(table_lower in table.lower() for table in info['tables']):
                matches.append(db_id)
        
        return matches
    
    def print_all_databases(self, limit: int = 50) -> None:
        """Pretty print all available databases."""
        available = self.get_all_databases()
        
        print(f"\n{'=' * 60}")
        print(f"AVAILABLE DATABASES ({len(available)} total)")
        print(f"{'=' * 60}")
        
        for i, db_id in enumerate(available[:limit], 1):
            info = self.db_cache[db_id]
            print(f"\n{i:3d}. {db_id:25s} | Tables: {info['table_count']:2d} | Cols: {info['column_count']:3d}")
            print(f"     Tables: {', '.join(info['tables'][:3])}")
            if len(info['tables']) > 3:
                print(f"             + {len(info['tables']) - 3} more")
        
        if len(available) > limit:
            print(f"\n... and {len(available) - limit} more databases")


db_discovery: DatabaseDiscovery = DatabaseDiscovery(schema_lookup, SPIDER_DB_DIR)

# ============================================================
# PART 4: AGENT PROMPTS
# ============================================================

MANAGER_AGENT_PROMPT: str = """You are a Database Query Manager Agent.

Your job is to:
1. Understand the user's natural language question
2. Analyze the database schema provided
3. Identify which tables and columns are relevant
4. Create a clear, step-by-step plan for generating the SQL query

DATABASE SCHEMA:
{schema}

USER QUESTION:
{question}

Respond in this exact format:
---
INTENT: [One sentence describing what the user wants]
RELEVANT_TABLES: [Comma-separated list of table names needed]
RELEVANT_COLUMNS: [Comma-separated list of column names needed]
JOINS_NEEDED: [Yes/No]
AGGREGATION: [None / COUNT / SUM / AVG / MAX / MIN]
FILTERS: [Describe any WHERE conditions needed]
EXECUTION_PLAN:
1. [First step]
2. [Second step]
---

Be precise and concise.
"""


CODER_AGENT_PROMPT: str = """You are a SQL Code Generator Agent.

Your job is to:
1. Read the execution plan from the Manager Agent
2. Generate a valid SQL query that follows the plan exactly
3. Use only the tables and columns mentioned in the schema
4. Use fully qualified column names (table.column) when there are joins to avoid ambiguity

DATABASE SCHEMA:
{schema}

MANAGER'S PLAN:
{plan}

USER QUESTION:
{question}

RULES:
- Generate ONLY the SQL query, nothing else
- Use proper SQL syntax
- Do not use tables or columns that don't exist in the schema
- Use fully qualified column names (table_name.column_name) when joining tables
- End the query with a semicolon

SQL QUERY:
"""


# ============================================================
# PART 5: SIMULATED AGENTS (FALLBACK)
# ============================================================

def simulate_manager_agent(question: str, schema: str) -> str:
    """Simulated Manager Agent for fallback."""
    question_clean = validate_string(question, "question", allow_none=True)
    schema_clean = validate_string(schema, "schema", allow_none=True)
    
    if not question_clean or not schema_clean:
        return "INTENT: Unable to analyze\nRELEVANT_TABLES: Unknown"
    
    question_lower = question_clean.lower()
    
    if "how many" in question_lower or "count" in question_lower:
        aggregation = "COUNT"
    elif "total" in question_lower or "sum" in question_lower:
        aggregation = "SUM"
    else:
        aggregation = "SELECT"
    
    table_name = "table1"
    if "Table:" in schema_clean:
        lines = schema_clean.split("\n")
        for line in lines:
            if line.startswith("Table:"):
                table_name = line.replace("Table:", "").strip()
                break
    
    return f"INTENT: {aggregation.lower()} data\nRELEVANT_TABLES: {table_name}\nJOINS_NEEDED: No\nAGGREGATION: {aggregation}"


def simulate_coder_agent(question: str, schema: str) -> str:
    """Simulated Coder Agent for fallback."""
    question_lower = question.lower()
    
    table_name = "table1"
    if "Table:" in schema:
        lines = schema.split("\n")
        for line in lines:
            if line.startswith("Table:"):
                table_name = line.replace("Table:", "").strip()
                break
    
    if "how many" in question_lower or "count" in question_lower:
        return f"SELECT COUNT(*) FROM {table_name};"
    else:
        return f"SELECT * FROM {table_name};"


# ============================================================
# PART 6: SMART OLLAMA CONNECTION - AUTO-DETECT MODELS
# ============================================================

print("\n[Detecting available Ollama models...]")

available_ollama_models = get_available_ollama_models()

print(f"\n📦 Ollama Models Downloaded ({len(available_ollama_models)}):")
for model in available_ollama_models:
    print(f"  ✓ {model}")

if not available_ollama_models:
    print("  ❌ No models found! Run: ollama serve && ollama pull qwen2.5:7b")

# Auto-select best models
manager_model = find_suitable_manager_model(available_ollama_models)
coder_model = find_suitable_coder_model(available_ollama_models)

print(f"\n[Selected models]")
print(f"  Manager Agent: {manager_model if manager_model else '❌ None'}")
print(f"  Coder Agent: {coder_model if coder_model else '❌ None'}")

LLM_AVAILABLE: bool = False
manager_chain: Optional[Any] = None
coder_chain: Optional[Any] = None

if manager_model and coder_model:
    print(f"\n[Attempting to connect to Ollama...]")
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        print("  ✓ LangChain libraries found")
        
        # Initialize the LLMs with YOUR models
        manager_llm = ChatOllama(
            model=manager_model,
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        
        coder_llm = ChatOllama(
            model=coder_model,
            temperature=0.0,
            base_url="http://localhost:11434"
        )
        
        # Create chains
        manager_prompt = ChatPromptTemplate.from_template(MANAGER_AGENT_PROMPT)
        manager_chain = manager_prompt | manager_llm | StrOutputParser()
        
        coder_prompt = ChatPromptTemplate.from_template(CODER_AGENT_PROMPT)
        coder_chain = coder_prompt | coder_llm | StrOutputParser()
        
        LLM_AVAILABLE = True
        print("\n✅ SUCCESS! Ollama connected with your models!")
        print(f"  - Manager: {manager_model}")
        print(f"  - Coder: {coder_model}")
        print(f"  - Status: USING REAL LLMs 🚀")
        
    except ImportError as e:
        print(f"\n⚠️  LangChain not installed!")
        print("  Fix: pip install langchain langchain-community")
        LLM_AVAILABLE = False
        
    except Exception as e:
        print(f"\n⚠️  Cannot connect to Ollama: {type(e).__name__}")
        print(f"  Error: {str(e)[:100]}")
        print("\n  Make sure Ollama is running: ollama serve")
        LLM_AVAILABLE = False
else:
    print("\n⚠️  Not enough models downloaded!")
    print("  You need at least 1 model for Manager and 1 for Coder")

if not LLM_AVAILABLE:
    print("\n⚠️  Falling back to SIMULATED AGENTS")
    print("  Output will be limited but functional")


# ============================================================
# PART 7: HELPER FUNCTIONS
# ============================================================

def get_schema_text(db_id: str) -> str:
    """Convert schema info to readable text format with fully qualified column names."""
    db_id_clean = validate_string(db_id, "db_id")
    
    if db_id_clean not in schema_lookup:
        return f"Schema not found for database: {db_id_clean}"
    
    info = schema_lookup[db_id_clean]
    
    lines: List[str] = []
    lines.append(f"Database: {db_id_clean}")
    lines.append("=" * 50)
    
    table_names: List[str] = info['table_names_original']
    column_names: List[tuple] = info['column_names_original']
    
    for table_idx, table_name in enumerate(table_names):
        lines.append(f"\nTable: {table_name}")
        lines.append("-" * 30)
        
        for col_idx, (tbl_idx, col_name) in enumerate(column_names):
            if tbl_idx == table_idx:
                # ✅ IMPROVED: Show fully qualified column name
                lines.append(f"  - {table_name}.{col_name}")
    
    return "\n".join(lines)


def get_db_path(db_id: str) -> str:
    """Get the SQLite database file path."""
    db_id_clean = validate_string(db_id, "db_id")
    return os.path.join(SPIDER_DB_DIR, db_id_clean, f"{db_id_clean}.sqlite")


def execute_sql(db_id: str, sql_query: str) -> Dict[str, Any]:
    """Execute SQL query on the specified database."""
    db_id_clean = validate_string(db_id, "db_id")
    sql_clean = validate_string(sql_query, "sql_query")
    
    db_path = get_db_path(db_id_clean)
    
    if not os.path.exists(db_path):
        return {
            "success": False,
            "result": [],
            "columns": [],
            "error": f"Database file not found: {db_path}"
        }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_clean)
        
        columns: List[str] = [desc[0] for desc in cursor.description] if cursor.description else []
        results: List[tuple] = cursor.fetchall()
        conn.close()
        
        return {
            "success": True,
            "result": results,
            "columns": columns,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "result": [],
            "columns": [],
            "error": str(e)
        }


# ============================================================
# PART 8: MAIN PIPELINE
# ============================================================

def run_agentic_pipeline(
    db_ids: Optional[List[str]] = None,
    db_id: Optional[str] = None,
    question: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """Run the complete agentic text-to-SQL pipeline."""
    
    try:
        question_clean = validate_question(question)
    except ValueError as e:
        print(f"\n{e}")
        raise
    
    try:
        if db_id is not None:
            db_id_clean = validate_db_id(db_id, schema_lookup)
            db_ids = [db_id_clean] if db_ids is None else db_ids
        
        db_ids_clean = validate_db_list(db_ids, schema_lookup)
    except ValueError as e:
        print(f"\n{e}")
        raise
    
    results: Dict[str, Any] = {
        "db_ids": db_ids_clean,
        "question": question_clean,
        "schemas": {},
        "manager_plan": None,
        "generated_sql": None,
        "execution_results": {},
        "final_answer": None,
        "success": False
    }
    
    print("=" * 60)
    print(f"QUESTION: {question_clean}")
    print(f"DATABASES: {db_ids_clean}")
    print(f"LLM: {'✅ Real (Ollama)' if LLM_AVAILABLE and use_llm else '⚠️  Simulated'}")
    print("=" * 60)
    
    valid_dbs: List[str] = []
    for db in db_ids_clean:
        if db in schema_lookup and os.path.exists(get_db_path(db)):
            valid_dbs.append(db)
    
    if not valid_dbs:
        final_msg = f"❌ None of the databases exist: {db_ids_clean}"
        results["final_answer"] = final_msg
        print(f"\n{final_msg}")
        return results
    
    results["db_ids"] = valid_dbs
    
    for db in valid_dbs:
        results["schemas"][db] = get_schema_text(db)
    
    combined_schema = results["schemas"][valid_dbs[0]] if len(valid_dbs) == 1 else "\n---\n".join(results["schemas"].values())
    
    print("\n[1/4] MANAGER AGENT - Analyzing question...")
    
    if use_llm and LLM_AVAILABLE and manager_chain:
        try:
            manager_plan: str = manager_chain.invoke({
                "schema": combined_schema,
                "question": question_clean
            })
        except Exception as e:
            print(f"  ⚠️  LLM error: {str(e)[:80]}")
            print("  Falling back to simulated agent...")
            manager_plan = simulate_manager_agent(question_clean, combined_schema)
    else:
        manager_plan = simulate_manager_agent(question_clean, combined_schema)
    
    results["manager_plan"] = manager_plan
    print(f"\n{manager_plan[:300]}...")
    
    print("\n[2/4] CODER AGENT - Generating SQL...")
    
    if use_llm and LLM_AVAILABLE and coder_chain:
        try:
            generated_sql: str = coder_chain.invoke({
                "schema": combined_schema,
                "plan": manager_plan,
                "question": question_clean
            })
        except Exception as e:
            print(f"  ⚠️  LLM error: {str(e)[:80]}")
            print("  Falling back to simulated agent...")
            generated_sql = simulate_coder_agent(question_clean, combined_schema)
    else:
        generated_sql = simulate_coder_agent(question_clean, combined_schema)
    
    # ✅ FIXED: Clean SQL output from LLM artifacts
    generated_sql = clean_sql_output(generated_sql)
    if not generated_sql:
        generated_sql = "SELECT 1"
    
    results["generated_sql"] = generated_sql
    print(f"\nSQL:\n{generated_sql}")
    
    print("\n[3/4] EXECUTOR - Running query...")
    
    for db in valid_dbs:
        result = execute_sql(db, generated_sql)
        results["execution_results"][db] = result
        
        if result["success"]:
            print(f"  ✓ {db}: {len(result['result'])} rows")
        else:
            print(f"  ✗ {db}: {result['error'][:60]}")
    
    print("\n[4/4] FORMATTER - Creating response...")
    
    result = results["execution_results"][valid_dbs[0]]
    if result["success"]:
        count = len(result["result"])
        if count == 0:
            final_answer = "No results found."
        elif count == 1 and len(result["result"][0]) == 1:
            final_answer = f"Answer: {result['result'][0][0]}"
        else:
            final_answer = f"Found {count} results:\n"
            for i, row in enumerate(result["result"][:5]):
                final_answer += f"  {i+1}. {row}\n"
        results["success"] = True
    else:
        final_answer = f"Error: {result['error']}"
    
    results["final_answer"] = final_answer
    print(f"\n{final_answer}")
    
    return results


# ============================================================
# PART 9: INTERACTIVE DEMO
# ============================================================

def interactive_demo() -> None:
    """Interactive demo."""
    print("\n" + "=" * 60)
    print("AGENTIC TEXT-TO-SQL - INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands: list, search <keyword>, query, quit")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if not cmd:
                continue
            elif cmd == 'quit':
                break
            elif cmd == 'list':
                db_discovery.print_all_databases()
            elif cmd.startswith('search'):
                keyword = cmd.replace('search', '').strip()
                if keyword:
                    matches = db_discovery.search_databases(keyword)
                    print(f"\nFound: {matches[:10]}")
            elif cmd == 'query':
                db_id = input("Database ID: ").strip()
                question = input("Question: ").strip()
                if db_id and question:
                    try:
                        run_agentic_pipeline(db_id=db_id, question=question, use_llm=LLM_AVAILABLE)
                    except:
                        pass
            else:
                print("Unknown command")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AGENTIC TEXT-TO-SQL v3.13 - BETTER SCHEMA CONTEXT")
    print("=" * 60)
    print(f"\nStatus: LLMs {'✅ Connected' if LLM_AVAILABLE else '⚠️  Using Fallback'}")
    
    print("\n📖 USAGE:")
    print("  1. interactive_demo()")
    print("  2. run_agentic_pipeline(db_id='aircraft', question='Compare flights and airlines', use_llm=LLM_AVAILABLE)")
    
    if available_ollama_models:
        print(f"\n💡 TIP: You have {len(available_ollama_models)} model(s) ready!")
        print("   No need to pull again - system auto-detected them!")