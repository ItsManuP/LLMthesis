import streamlit
import pandas
import numpy as np
import os
import time
from PyPDF2 import PdfReader
import langchain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import text_splitter
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate, LLMChain


os.environ["OPENAI_API_KEY"] = "Insert your OpenAI key Here"


streamlit.set_page_config(page_title='Michela Murgia LLM')


streamlit.image("https://www.donnamoderna.com/content/uploads/2023/05/michela-murgia-2023.jpg",caption='Michela Murgia mentre parla della sua malattia')


streamlit.header("Se hai qualche domanda sulle opere di Michela Murgia, chiedi pure")

pdf= "documento.pdf"
if(pdf is not None):
    lettura_pdf = PdfReader(pdf)
    testo = ""
    for pagina in lettura_pdf.pages:
        testo += pagina.extract_text()
    testo_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
    chunks = testo_splitter.split_text(testo)
    
    richiesta_utente = streamlit.text_input("Il pdf contiene informazioni su alcune sue opere")
    if(richiesta_utente is not None):
        embeddings = OpenAIEmbeddings()
        documento_db = FAISS.from_texts(chunks, embeddings)
        if (richiesta_utente):
            docs = documento_db.similarity_search(richiesta_utente)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            risposta = chain.run(input_documents=docs, question=richiesta_utente)
            streamlit.write(risposta)
