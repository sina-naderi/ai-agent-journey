# phase2_week4/day1_postgresql.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── Database connection ───────────────────────────────────────────────────
# Connection string can be stored in .env for production
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "agent_db",
    "user": "postgres",
    "password": "postgres"
}

def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)

# ── Create tables ─────────────────────────────────────────────────────────
def create_tables():
    """Create the necessary tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Conversations table — stores each chat session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            user_message TEXT NOT NULL,
            assistant_message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Agent actions table — logs every tool call
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_actions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) NOT NULL,
            tool_name VARCHAR(100) NOT NULL,
            tool_input TEXT,
            tool_output TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Tables created successfully")

# ── CRUD operations ───────────────────────────────────────────────────────
def save_conversation(session_id: str, user_msg: str, assistant_msg: str):
    """Save a conversation turn to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO conversations
           (session_id, user_message, assistant_message)
           VALUES (%s, %s, %s)""",
        (session_id, user_msg, assistant_msg)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_conversation_history(session_id: str, limit: int = 10) -> list:
    """Retrieve recent conversation history for a session."""
    conn = get_connection()
    # RealDictCursor returns rows as dictionaries instead of tuples
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        """SELECT user_message, assistant_message, created_at
           FROM conversations
           WHERE session_id = %s
           ORDER BY created_at DESC
           LIMIT %s""",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    # Return in chronological order (oldest first)
    return list(reversed(rows))

def log_agent_action(session_id: str, tool_name: str,
                     tool_input: str, tool_output: str):
    """Log a tool call made by the agent."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO agent_actions
           (session_id, tool_name, tool_input, tool_output)
           VALUES (%s, %s, %s, %s)""",
        (session_id, tool_name, tool_input, tool_output)
    )
    conn.commit()
    cursor.close()
    conn.close()

def get_session_stats(session_id: str) -> dict:
    """Get statistics for a session — useful for monitoring."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Count messages
    cursor.execute(
        "SELECT COUNT(*) as count FROM conversations WHERE session_id = %s",
        (session_id,)
    )
    msg_count = cursor.fetchone()["count"]

    # Count tool calls
    cursor.execute(
        "SELECT tool_name, COUNT(*) as count FROM agent_actions "
        "WHERE session_id = %s GROUP BY tool_name",
        (session_id,)
    )
    tool_counts = cursor.fetchall()

    cursor.close()
    conn.close()

    return {
        "session_id": session_id,
        "message_count": msg_count,
        "tool_usage": {row["tool_name"]: row["count"] for row in tool_counts}
    }

# ── Tests ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create tables
    create_tables()

    # Simulate a conversation
    session = "test_session_001"

    logger.info("Saving test conversations...")
    save_conversation(session, "What is the return policy?",
                      "You can return products within 30 days.")
    save_conversation(session, "How long does shipping take?",
                      "Standard shipping is 3-5 business days.")
    save_conversation(session, "Do you have student discounts?",
                      "Yes, 20% off with valid student ID.")

    # Log some tool actions
    log_agent_action(session, "search_knowledge_base",
                     "return policy", "30 days return window")
    log_agent_action(session, "search_knowledge_base",
                     "shipping time", "3-5 business days Europe")

    # Retrieve history
    print("\n── Conversation History ──")
    history = get_conversation_history(session)
    for turn in history:
        print(f"User: {turn['user_message']}")
        print(f"Bot:  {turn['assistant_message']}")
        print(f"Time: {turn['created_at']}\n")

    # Get stats
    print("── Session Stats ──")
    stats = get_session_stats(session)
    print(f"Session: {stats['session_id']}")
    print(f"Messages: {stats['message_count']}")
    print(f"Tool usage: {stats['tool_usage']}")