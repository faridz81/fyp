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
                description = f"id_kehadiran: {value}. This column represents the unique attendance ID for the record."
            elif key == "name":
                description = f"name: {value}. This column represents the full name of the student."
            elif key == "nama_subject":
                description = f"nama_subject: {value}. This column represents the name of the classroom's subject."
            elif key == "info_kelas":
                description = f"info_kelas: {value}. This column represents the info of the classroom."
            elif key == "no_matriks":
                description = f"no_matriks: {value}. This column represents the student's matriculation number."
            elif key == "nama_fakulti":
                description = f"nama_fakulti: {value}. This column represents the name of the faculty the student is enrolled in."
            elif key == "nama_course":
                description = f"nama_course: {value}. This column represents the name of the course the student is taking."
            elif key == "register_time":
                description = f"register_time: {value}. This column represents the date and time when the student punch their attendance."
            elif key == "lokasi_device":
                description = f"lokasi_device: {value}. This column represents the location of the device used to register the attendance. If device is phone, means attendance take by using QR code generate by lecturer's phone"
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
    Data is about attendance log of the student.
    Student name with the most row of data log, got the highest attendance performance.
    Your goal is to provide accurate answers based on the context of the data, including partial name matching and identifying students with high attendance.


    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   client=genai,
                                   temperature=0.1,
                                   top_k=10,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a JSON URL and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question, top_k=10)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response

def main():
    st.set_page_config(
        page_title="Gemini JSON Chatbot",
        page_icon="🤖"
    )

    # Sidebar for entering JSON URL
    with st.sidebar:
        st.title("Menu:")
        json_url = st.text_input("Enter the URL of your JSON data")
        
        if st.button("Submit & Process"):
            if json_url:
                with st.spinner("Processing..."):
                    raw_data = read_json_data(json_url)
                    text_chunks = get_text_chunks(raw_data)
                    if text_chunks:  # Check if text_chunks is not empty
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.warning("No data found in the JSON.")
            else:
                st.warning("Please enter a JSON URL.")
                return  # Return to avoid further execution

    # Main content area for displaying chat messages
    st.title("Chat with JSON data using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a JSON URL and ask me a question"}]

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
            with st.spinner("Thinking..."):
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
