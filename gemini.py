import os
import json
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load and return JSON data
def read_json_data(json_url):
    response = requests.get(json_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to load data from the URL.")
        return []

# Group data by (student_name, class_name) and provide counts
def group_data_by_student_class(data):
    grouped_data = {}
    for entry in data:
        student_class = (entry['student_name'], entry['class_name'])
        grouped_data.setdefault(student_class, []).append(entry)
    return grouped_data

# Convert grouped data to text chunks
def get_text_chunks(grouped_data):
    text = ""
    for (student, class_name), attendances in grouped_data.items():
        attendance_count = len(attendances)
        text += f"Student: {student}, Class: {class_name}, Attendances: {attendance_count}\n"
        for entry in attendances:
            text += f"Date: {entry['date']}, Time: {entry['time']}\n"
    
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=2000, chunk_overlap=200)  # Adjusted for better handling
    return splitter.split_text(text)

# Create a vector store with embeddings
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define conversational chain with improved prompt
def get_conversational_chain():
    prompt_template = """
    You are a data assistant helping me as a lecturer to analyze attendance data for my class.
    Data Details:
    - Each JSON entry is a separate attendance log for one student in a specific class in a specific time.
    - `student_name` represents the student's name, `class_name` identifies the class, `date`, and `time` represent the attendance log.
    
    Instructions for Analysis:
    - When counting attendance, use exact matches for `student_name` and `class_name`.
    - Each unique date and time entry for the same `class_name` counts as a separate attendance, even if it's for the same student.
    - Express counts as digits (e.g., "3" instead of "three") and ensure clarity and accuracy.
    - Present attendance details in a table format when they contain more than two columns.
    - Respond in Malay if the question is in Malay, and in English if it is in English.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        client=genai,
        temperature=0.2,   # Lowered temperature for better accuracy
        top_k=5            # Adjusted for more relevant responses
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the attendance data."}]

# Process user input and generate AI response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, top_k=5)  # Adjusted top_k for focused results

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

# Streamlit application main function
def main():
    st.set_page_config(page_title="Smart Attendance AI Assistant", page_icon="🤖")
    query_params = st.query_params
    lecturer_id = query_params.get("id", None)

    if lecturer_id:
        json_url = f"https://fyp.smartsolah.com/kehadiranApi/kehadiranMyStudent/{lecturer_id}"
        
        with st.spinner("Processing..."):
            raw_data = read_json_data(json_url)
            grouped_data = group_data_by_student_class(raw_data)
            text_chunks = get_text_chunks(grouped_data)
            if text_chunks:
                get_vector_store(text_chunks)
                st.success("Student attendance data processed successfully.")
            else:
                st.warning("No data available.")
    else:
        st.error("Please provide lecturer information.")

    st.title("Smart Attendance AI Assistant")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Feel free to ask questions about the attendance data."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
