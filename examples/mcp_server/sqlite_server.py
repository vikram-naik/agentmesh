#!/usr/bin/env python3
"""
SQLite MCP Server for AgentMesh Tactical Loop Demo
---------------------------------------------------
This server provides a set of dependent tools for querying a SQLite database.
The tools are designed to force a multi-step workflow:

1. list_tables() -> Returns available tables
2. describe_table(table_name) -> Returns schema for a table
3. run_query(sql) -> Executes a SQL query

The dependency chain forces the agent to:
- First discover what tables exist
- Then understand their schemas
- Finally construct and run the appropriate query

Built with FastMCP for streamable HTTP transport.
"""

import sqlite3
from pathlib import Path
from fastmcp import FastMCP

# --- Database Setup ---
DB_PATH = Path(__file__).parent / "demo.db"

def init_database():
    """Initialize the demo database with sample data."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Create products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL
        )
    """)
    
    # Create sales table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            customer_region TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    
    # Create customers table (additional complexity)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            region TEXT NOT NULL,
            tier TEXT NOT NULL
        )
    """)
    
    # Check if data exists
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        # Insert products
        products = [
            (1, "Laptop Pro", "Electronics", 1299.99),
            (2, "Wireless Mouse", "Electronics", 49.99),
            (3, "USB-C Hub", "Electronics", 79.99),
            (4, "Office Chair", "Furniture", 299.99),
            (5, "Standing Desk", "Furniture", 599.99),
            (6, "Monitor Arm", "Furniture", 149.99),
            (7, "Notebook Set", "Stationery", 24.99),
            (8, "Pen Collection", "Stationery", 39.99),
        ]
        cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
        
        # Insert sales
        sales = [
            (1, 1, "West", 5, "2024-01-15"),
            (2, 1, "East", 3, "2024-01-20"),
            (3, 2, "West", 20, "2024-02-01"),
            (4, 3, "West", 15, "2024-02-10"),
            (5, 4, "North", 8, "2024-02-15"),
            (6, 5, "West", 2, "2024-03-01"),
            (7, 6, "East", 10, "2024-03-10"),
            (8, 1, "West", 7, "2024-03-15"),
            (9, 2, "South", 12, "2024-03-20"),
            (10, 7, "West", 50, "2024-04-01"),
        ]
        cursor.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", sales)
        
        # Insert customers
        customers = [
            (1, "Acme Corp", "West", "Enterprise"),
            (2, "TechStart Inc", "East", "Startup"),
            (3, "BigBox Retail", "North", "Enterprise"),
            (4, "LocalShop", "South", "SMB"),
            (5, "WestCoast LLC", "West", "Enterprise"),
        ]
        cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)
        
        conn.commit()
        print("Database initialized with sample data.")
    
    conn.close()

# Initialize database on import
init_database()

# --- FastMCP Server ---
mcp = FastMCP("sqlite_demo")

@mcp.tool()
def list_tables() -> str:
    """
    List all available tables in the database.
    This is the first step to understand what data is available.
    Returns a comma-separated list of table names.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return f"Available tables: {', '.join(tables)}"

@mcp.tool()
def describe_table(table_name: str) -> str:
    """
    Get the schema (columns and types) for a specific table.
    Use this after list_tables() to understand the structure before querying.
    
    Args:
        table_name: Name of the table to describe
    
    Returns:
        Column definitions for the table
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Verify table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if not cursor.fetchone():
        conn.close()
        return f"Error: Table '{table_name}' does not exist. Use list_tables() first."
    
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    
    schema = [f"{col[1]} ({col[2]})" for col in columns]
    return f"Table '{table_name}' columns: {', '.join(schema)}"

@mcp.tool()
def run_query(sql: str) -> str:
    """
    Execute a SQL SELECT query on the database.
    Use describe_table() first to understand the schema before writing queries.
    
    Args:
        sql: A valid SQL SELECT statement
    
    Returns:
        Query results as formatted text
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        if not rows:
            return "Query returned no results."
        
        # Format results
        result_lines = [" | ".join(columns)]
        result_lines.append("-" * len(result_lines[0]))
        for row in rows:
            result_lines.append(" | ".join(str(v) for v in row))
        
        conn.close()
        return "\n".join(result_lines)
    except Exception as e:
        conn.close()
        return f"Query error: {str(e)}"

# --- Main ---
if __name__ == "__main__":
    # Run with streamable HTTP transport on port 7002
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7002)
