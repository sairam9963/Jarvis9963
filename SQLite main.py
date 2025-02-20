import sqlite3

# Initialize Database
conn = sqlite3.connect("chat_memory.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, user_input TEXT, ai_response TEXT)")
conn.commit()

def save_to_memory(user_input, ai_response):
    """ Store conversation in the database """
    cursor.execute("INSERT INTO chat_history (user_input, ai_response) VALUES (?, ?)", (user_input, ai_response))
    conn.commit()

def get_previous_context():
    """ Retrieve last 5 messages from memory """
    cursor.execute("SELECT user_input, ai_response FROM chat_history ORDER BY id DESC LIMIT 5")
    return cursor.fetchall()
