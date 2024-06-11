import streamlit as st
import pinecone 
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.vectorstores import Pinecone
#from langchain import PineconeVectorStore
#from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document

 
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pc = Pinecone(api_key='44e11952-d1a6-4dcb-8be2-d2f25261a9e4')

index_name="langchainvectors"
index=pc.Index(index_name)
index.describe_index_stats() 


def get_pdf_text(pdf_docs):
    # create a loader
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks
     


def get_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings_list = []
    
    for text in texts:
        res = embeddings.embed_query(text)  
        embeddings_list.append(res) 
    meta = [{'text': line} for line in texts]
    
    ids_batch = [str(n) for n in range(len(texts))]
    to_upsert = zip(ids_batch, embeddings_list, meta)
    index.upsert(vectors=to_upsert)   
    
      

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    qe=embeddings.embed_query(user_question) 
    docs = index.query(vector=qe,top_k=3,include_metadata=True)   
    doc=[]
    match=docs.matches
    for match in docs['matches']:
        a=Document(page_content=''.join(match['metadata']['text']))
        doc.append(a) 

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":doc, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
