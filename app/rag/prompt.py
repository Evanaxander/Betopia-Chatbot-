def build_prompt(context: str, question: str, history: list = None, max_history: int = 5) -> str:
    """
    Build a prompt for the Knowledge Assistant that handles multiple sources 
    and conversation history.
    """
    
    # --- Format conversation history ---
    if history:
        # Keep only the last N turns
        recent_history = history[-max_history:]
        history_str = ""
        for i, (user_msg, bot_msg) in enumerate(recent_history, 1):
            history_str += f"User: {user_msg}\nAssistant: {bot_msg}\n\n"
    else:
        history_str = "No previous conversation."

    # --- Build final prompt ---
    prompt = f"""
You are a helpful Knowledge Assistant. Use the provided context to answer the user's question accurately.
- Base your answers ONLY on the provided context.
- If the context contains information from multiple files, combine them for a complete answer.
- If the user asks about something not in the context, respond:
  "I don't have information on that in my current documents, but I can help with topics found in your uploaded files!"
- Prioritize information from the most recent file version mentioned in the source tags if there is a conflict.

Conversation History:
{history_str}

Context (from multiple sources):
{context}

User Question:
{question}

Answer:
"""
    return prompt



def build_prompt(context: str, question: str, history: list = None) -> str:
    # Format the history into a string
    history_str = ""
    if history:
        for i, (u, a) in enumerate(history):
            history_str += f"Turn {i+1}:\nUser: {u}\nAssistant: {a}\n\n"
    else:
        history_str = "None."

    return f"""

    
You are a Knowledge Assistant with access to specific documents and conversation history.

CONVERSATION HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the User Question using the DOCUMENT CONTEXT.
2. If the user uses pronouns like "my", "it", or "they", refer to the CONVERSATION HISTORY to understand what they are referring to.
3. If the answer is in the history but not the current context, you may use the history to maintain continuity.
4. If you truly cannot find the answer in either, politely say you don't know.

User Question: {question}
Answer:"""