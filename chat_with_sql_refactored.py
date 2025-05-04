
import os
import streamlit as st
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI

# Load environment variables
load_dotenv()

# --- Constants ---
LLM = GoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), api_key=os.getenv("GEMINI_API_KEY"))
# LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

SQL_QUERY_PROMPT = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Promedio de coste de soles por tipo de planta
    SQL Query: SELECT tipo, AVG(coste_soles) AS promedio_coste_soles FROM plantas GROUP BY tipo;
    Question: Plantas por minijuego
    SQL Query: SELECT m.nombre_minijuego AS Minijuego, p.nombre_planta AS Planta FROM minijuego_plantas mp JOIN minijuegos m ON mp.id_minijuego = m.id_minijuego JOIN plantas p ON mp.id_planta = p.id_planta ORDER BY m.nombre_minijuego, p.nombre_planta;
    Question: Ver todas las partidas con detalles de planta, zombi y escenario
    SQL Query: SELECT pa.id_partida, p.nombre_planta, z.nombre_zombie, e.nombre AS escenario, pa.fecha_partida, pa.resultado FROM partidas pa JOIN plantas p ON pa.id_planta = p.id_planta JOIN zombies z ON pa.id_zombie = z.id_zombie JOIN escenarios e ON pa.id_escenario = e.id_escenario;

    Your turn:
    
    Question: {question}
    SQL Query:
    """

NATURAL_RESPONSE_PROMPT = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""

SQL_EXPLANATION_PROMPT = """
    You are a data analyst. You will be given a SQL query. Explain what it does clearly and concisely in {language}.
    SQL Query: {query}
    Explanation:
    """

ERROR_EXPLANATION_PROMPT = """
    You are a database expert. A user attempted to run the following SQL query and received an error. Explain the cause of the error and how to fix it in {language}.
    SQL Query: {query}
    Error: {error}
    Explanation:
    """

# --- Database Functions ---
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(uri)

def run_dict_query(user: str, password: str, host: str, port: str, database: str, query: str):
    conn = mysql.connector.connect(
        host=host, user=user, password=password, port=port, database=database
    )
    with conn.cursor(dictionary=True) as cursor:
        cursor.execute(query)
        result = cursor.fetchall()
    conn.close()
    return result

# --- LangChain Chains ---
def get_sql_chain(db: SQLDatabase):
    prompt = ChatPromptTemplate.from_template(SQL_QUERY_PROMPT)
    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | LLM
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, sql_query: str, response: list):
    prompt = ChatPromptTemplate.from_template(NATURAL_RESPONSE_PROMPT)
    return (
        RunnablePassthrough.assign(
            schema=lambda _: db.get_table_info(),
        )
        | prompt
        | LLM
        | StrOutputParser()
    ).invoke({
        "question": user_query,
        "chat_history": chat_history,
        "query": sql_query,
        "response": response,
    })

def explain_sql_query(sql_query: str):
    prompt = ChatPromptTemplate.from_template(SQL_EXPLANATION_PROMPT)
    return (
        RunnablePassthrough.assign(query=lambda _: sql_query)
        | prompt
        | LLM
        | StrOutputParser()
    ).invoke({
        "language": "spanish"
    })

def explain_sql_error(sql_query: str, error: str):
    prompt = ChatPromptTemplate.from_template(ERROR_EXPLANATION_PROMPT)
    return (
        RunnablePassthrough.assign(query=lambda _: sql_query, error=lambda _: error)
        | prompt
        | LLM
        | StrOutputParser()
    ).invoke({
        "language": "spanish"
    })

# --- UI Config ---
def sidebar_config():
    st.sidebar.markdown("## Settings")
    st.sidebar.text_input("Host", value="localhost", key="Host")
    st.sidebar.text_input("Port", value="3306", key="Port")
    st.sidebar.text_input("User", value="root", key="User")
    st.sidebar.text_input("Password", type="password", value="root", key="Password")
    st.sidebar.text_input("Database", value="pvz", key="Database")
    
    if st.sidebar.button("Connect"):
        try:
            with st.spinner("Connecting to database..."):
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("✅ Connected to database!")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

# --- Main App ---
def render_chat_messages():
    for message in st.session_state.chat_history:
        avatar = "img/chatbot.png" if isinstance(message, AIMessage) else "img/boy.png"
        role = "ai" if isinstance(message, AIMessage) else "human"
        with st.chat_message(name=role, avatar=avatar):
            st.markdown(message.content)

def handle_user_query(user_query: str):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message(name="human", avatar="img/boy.png"):
        st.markdown(user_query)

    with st.chat_message(name="ai", avatar="img/chatbot.png"):
        try:
            # Detect if input is SQL (basic heuristic: starts with SELECT/UPDATE/etc.)
            if user_query.strip().lower().startswith(("select", "update", "insert", "delete", "with", "call")):
                cleaned_query = user_query.strip()
                result = run_dict_query(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"],
                    cleaned_query
                )
                df = pd.DataFrame(result)
                explanation = explain_sql_query(cleaned_query)

                st.markdown("### 🛠️ SQL Query:")
                st.code(cleaned_query, language="sql")
                st.markdown("### 💬 Explanation:")
                st.markdown(explanation)

                st.markdown("### 📊 Result:")
                if not df.empty:
                    st.table(df)
                else:
                    st.info("The query returned no results.")

                st.session_state.chat_history.append(AIMessage(content=explanation))
            else:
                sql_chain = get_sql_chain(st.session_state.db)
                sql_query = sql_chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "question": user_query
                }).strip()

                result = run_dict_query(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"],
                    sql_query
                )
                df = pd.DataFrame(result)
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history, sql_query, result)

                st.markdown("### 🛠️ Generated SQL Query:")
                st.code(sql_query, language="sql")
                st.markdown("### 🧪 Raw Query Result:")
                st.write(result)
                st.markdown("### 📊 Query Result:")
                if not df.empty:
                    st.table(df)
                else:
                    st.info("The query returned no results.")

                st.markdown("### 💬 Response:")
                st.markdown(response)

                # Update chat history
                st.session_state.chat_history.extend([
                    AIMessage(content=sql_query),
                    AIMessage(content=response)
                ])
        except Exception as e:
            error_message = explain_sql_error(user_query, str(e))
            st.error("❌ An error occurred while processing the query.")
            st.markdown("### ❗ Error explanation:")
            st.markdown(error_message)
            st.session_state.chat_history.append(AIMessage(content=error_message))

# --- Run App ---
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
st.title(":speech_balloon: Chat with MySQL")

sidebar_config()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database.")
    ]

if "db" in st.session_state:
    render_chat_messages()
    user_input = st.chat_input("Ask anything about your database...")
    if user_input and user_input.strip():
        handle_user_query(user_input)
