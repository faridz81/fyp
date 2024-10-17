import os
import pandas as pd
import streamlit as st
from io import StringIO
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_csv_agent
import requests

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Define the prompt template
PROMPT_TEMPLATE = (
    "You are a data assistant. Your task is to assist users in retrieving information from the provided JSON data."
    "Data is about attendance log of the student of my class. My class means that I am a lecturer. Each one json object contains one attendance of a student."
    "Count number of object with same student name, to count total attendance for each student."
    "IMPORTANT! You determine a class is distinct based on class_name. Do not use partial match for class name."
    "Use date and time in easy readable format."
    "Different punched date and time is considered different attendance, although student attends the same class name."
    "Answer with natural language, don't use json code or other code as an answer. Answer in Malay if the question is in Malay. Answer in English if the question is in English. Express count number by digit not text. Explain your answer. Be friendly."
    "Do not always use a table. If data has more than two columns, use a table to show the data."
)

def main():
    st.set_page_config(page_title="ASK YOUR CSV")
    st.header("ASK YOUR CSV")

    # Get the filename from the URL parameters
    query_params = st.query_params
    filename = query_params.get("file")

    if filename:
        # Construct the URL to fetch the CSV file
        url = f"https://fyp.smartsolah.com/{filename}"

        # Load the CSV file from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request fails
        csv_data = StringIO(response.text)

        # Create the agent with the loaded CSV
        agent = create_csv_agent(
            ChatGroq(
                model="llama3-70b-8192",
                temperature=0
            ),
            csv_data,
            verbose=True,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )

        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Create a container for the conversation history
        history_container = st.container()

        # Display conversation history
        with history_container:
            st.markdown("<div style='height: 400px; overflow-y: scroll;'>", unsafe_allow_html=True)
            for speaker, text in st.session_state.chat_history:
                if speaker == "User":
                    st.markdown(f"<div style='text-align: right;'><strong>{speaker}:</strong> {text}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: left;'><strong>{speaker}:</strong> {text}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Input field at the bottom
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            # Include the prompt in the run method
            full_prompt = f"{PROMPT_TEMPLATE}\n\nUser Question: {user_question}"
            with st.spinner(text="In progress..."):
                response = agent.run(full_prompt)

                # Store the question and response in chat history
                st.session_state.chat_history.append(("User", user_question))
                st.session_state.chat_history.append(("Assistant", response))

                # Clear the input field
                st.experimental_rerun()

    else:
        st.warning("Please provide a 'file' parameter in the URL.")

if __name__ == "__main__":
    main()
