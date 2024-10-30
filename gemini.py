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
from collections import defaultdict

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_json_data(json_url):
    response = requests.get(json_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to load data from the URL.")
        return []

def group_data_by_student_class(data):
    grouped_data = defaultdict(list)
    for entry in data:
        student_class = (entry['student_name'], entry['class_name'])
        grouped_data[student_class].append(entry)
    return grouped_data

def format_grouped_data(grouped_data):
    formatted_text = ""
    for (student_name, class_name), logs in grouped_data.items():
        log_details = "\n".join(
            f"Date and Time: {log['register_time']}, Device Location: {log['lokasi_device']}" 
            for log in logs
        )
        formatted_text += f"Student: {student_name}, Class: {class_name}, Attendance Records:\n{log_details}\n\n"
    return formatted_text

def get_text_chunks(data):
    formatted_text = format_grouped_data(group_data_by_student_class(data))
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=50000,
        chunk_overlap=5000)
    chunks = splitter.split_text(formatted_text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    I am a lecturer. You are a data assistant helping me as lecturer to analyze attendance data for my class.

    Data Details:
    - Each JSON entry is a separate attendance log for one student in a specific class.
    - `student_name` represents the student's name, `class_name` identifies the class, `register_time` represent the attendance log.
    
    Instructions for Analysis:
    - When counting attendance, use exact matches for `student_name` and `class_name`.
    - Each unique date and time entry for the same `class_name` counts as a separate attendance, even if it's for the same student.
    - Express counts as digits (e.g., "3" instead of "three") and ensure clarity and accuracy.
    - Present attendance details in a table format when they contain more than two columns.
    - Respond in Malay if the question is in Malay, and in English if it is in English.

    Context:\n {context}?\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.1, top_k=5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, top_k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    st.set_page_config(page_title="Smart Attendance AI Assistant", page_icon="🤖")

    query_params = st.query_params
    lecturer_id = query_params.get("id", None)

    if lecturer_id:
        json_url = f"https://fyp.smartsolah.com/kehadiranApi/kehadiranMyStudent/{lecturer_id}"
        
        with st.spinner("Processing..."):
            raw_data = read_json_data(json_url)
            text_chunks = get_text_chunks(raw_data)
            if text_chunks:
                get_vector_store(text_chunks)
                st.success("Data pelajar telah diproses")
            else:
                st.warning("Tiada data")
    else:
        st.error("Sila berikan maklumat pensyarah")

    st.title("Smart Attendance AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Sila bertanya berkenaan maklumat kehadiran sahaja."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Sedang Proses..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ""
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
