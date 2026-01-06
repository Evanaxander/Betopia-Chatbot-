# app/rag/prompt.py

def build_prompt(context: str, question: str, history: list = None, max_history: int = 5) -> str:
    """
    Build a prompt for the Betopia chatbot that:
    - Answers only about Betopia (from PDF)
    - Engages the user naturally
    - Remembers conversation history

    Parameters:
    - context: str, relevant chunks retrieved from PDF/vector DB
    - question: str, current user question
    - history: list of tuples [(user_msg, bot_msg), ...]
    - max_history: int, how many previous turns to include

    Returns:
    - prompt: str, ready for OpenAI model
    """

    # Format conversation history
    if history:
        history = history[-max_history:]
        history_str = ""
        for i, (user_msg, bot_msg) in enumerate(history, 1):
            history_str += f"Turn {i}:\nUser: {user_msg}\nAssistant: {bot_msg}\n\n"
    else:
        history_str = "No previous conversation."

    # Build final prompt
    prompt = f"""
You are Betopia PDF Chatbot, a friendly assistant that ONLY talks about Betopia.
- Answer all questions strictly using the provided document context.
- If the user says hello, greets, or small talk, respond naturally and steer the conversation toward Betopia.
- If a question cannot be answered from the document, respond:
  "I don't know based on the provided document, but I can tell you about Betopia Limited!"

Conversation History:
{history_str}

Context (from the document):
{context}

User Question:
{question}

Answer:
"""
    return prompt
