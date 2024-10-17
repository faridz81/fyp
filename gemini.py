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

def read_json_data(json_url):
    response = requests.get(json_url)
    if response.status_code == 200:
        return response.json()  # Return parsed JSON data
    else:
        st.error("Failed to load data from the URL.")
        return []

def get_text_chunks(data):
    text = ""
    # Convert list of dictionaries to a structured string with explanations for each key
    for entry in data:
        for key, value in entry.items():
            if key == "id_kehadiran":
                description = f"id_kehadiran: {value}. This key represents the unique attendance ID for the record."
            elif key == "student_id":
                description = f"student_id: {value}. This key represents the unique ID for each student in the record. Same student_id belongs to same person"
            elif key == "student_name":
                description = f"name: {value}. This key represents the full name of the student."
            elif key == "lecturer_name":
                description = f"lecturerName: {value}. This key represents the full name of the class lecturer."
            elif key == "class_id":
                description = f"class_id: {value}. This key represents of the class ID. You determine a class is unique based on class_id"
            elif key == "nama_subject":
                description = f"nama_subject: {value}. This key represents of the class name. "
            elif key == "info_kelas":
                description = f"info_kelas: {value}. This key represents the class section."
            elif key == "no_matriks":
                description = f"no_matriks: {value}. This key represents the student's matriculation number."
            elif key == "nama_fakulti":
                description = f"nama_fakulti: {value}. This key represents the name of the faculty the student is enrolled in."
            elif key == "nama_course":
                description = f"nama_course: {value}. This key represents the name of the course the student is taking."
            elif key == "register_time":
                description = f"register_time: {value}. This key represents the date and time when the student punched their attendance."
            elif key == "lokasi_device":
                description = f"lokasi_device: {value}. This key represents the location of the device used to register the attendance."
            else:
                description = f"{key}: {value}."
            
            text += description + "\n"
    
    # Split the text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    
    return chunks

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a JSON data assistant. Your task is to assist users in retrieving information from the provided JSON data.
    Data is about attendance log of the student of my class. My class means that I am a lecturer. 
    Student name with the most attendance object in array, got the highest attendance performance.    
    You determine a class is unique based on class_id. Do not write class_id in answers.
    Answer with natural language, don't use json code or other code as answer. Answer in Malay if question in Malay. Answer in English if Question in English. Express count number by digit not text.  Explain your answer. Be friendly.
    Your goal is to provide accurate answers based on the context of the data, including partial name matching and identifying students with high attendance.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   client=genai,
                                   temperature=0.1,
                                   top_k=10)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the attendance data."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

def main():
    st.set_page_config(
        page_title="Smart Attendance AI Assistant",
        page_icon="🤖"
    )

    # Get the lecturer ID from the GET parameter
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

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Sila bertanya berkenaan maklumat kehadiran sahaja."}]

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
            with st.spinner("Sedang Proses..."):
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
