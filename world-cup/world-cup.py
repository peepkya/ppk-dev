import os

import openai

import sys

sys.path.append('../..')

openai_api_key = "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"

#Setting the LLM, this code has been borrowed as is from the course, it sets the llm that will be leveraged
import datetime

current_date = datetime.datetime.now().date()

if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

print(llm_name)

#These are additional libraries that we need to install if not already done
#! pip install pypdf

#! pip install chromadb

#Importing libraries pertaining to langchain
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI

#Embedding is a complex topic. Long story short, Natural Language Processing models work on text which needs to be translated to numbers before a model can be trained as model itself is an abstraction of the input and output. Embedding is a function that converts the text to vectors. Well vector is a uni dimension matrix of real numbers. Here we are using the Open AI embedding function. In order to use the Open API we need to pass in the Open AI api key. If it did not make sense, do not worry, keep marching, you can come back later to deep dive on this.
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

#The vectors we generate has to be stored in a database, here we will be storing it in a Chroma vector database. A relational database is not suitable for this hence a vector database.
# remove old database files if any

#!rm -rf ./worldcupData/chromadb

persist_directory = 'worldcupData/chrxomadb/'

#It's finally the time to load all the files which are stored in worldcupData/1992. The langchain makes it easy to load all the files using this loader. Here each file is a match that was played in 1992 World Cup. It has the score sheet and summary of the match.
loader = PyPDFDirectoryLoader("worldcupData/1992/")

pages = loader.load()

#Once all the files have been loaded, we need to split the files so that they can processed with ease. Here we are using a function within langchain which makes splitting the files a cake walk. Once again splitting is a topic that needs to be addressed as a separate topic. But for the sake of learning, lets keep marching towards the final goal, we will come back to splitting at a later point in time.
from langchain.text_splitter import RecursiveCharacterTextSplitter

 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

splits = text_splitter.split_documents(pages)

#Before we call the LLM, we will use the feature of the langchain to identify the sections of the text that match our query. We do this because we want to minimise the content that we send to LLM as a LLM has a limitation on the length of the context window. We are going to use something called MMR but for now you can ignore it and just go ahead with the execution
#Here we are going to check "How many runs did Kris Srikkanth score against England?". Ideally the LLM should come back with the answer. Before you shoot this, you may want to check the score card and understand the complexity of the content that you are presenting. As you can see the pdf has multiple parts some of which are contextual to what we are asking and others do not have a relevance at all. The task of the LLM is to find what we are seeking and get back with the right answer. Sometimes LLM may not answer it correct so we need to go back and revisit the question we asked. That was a long passage and before you go to sleep let's execute this.
question = "How many runs did Kris Srikkanth score against England?"

#vectordb = Chroma.from_documents(
#    documents=splits,
#    embedding=embedding,
#    persist_directory=persist_directory
#)

docs = "testing" #vectordb.max_marginal_relevance_search(question)

#The docs in the previous step retrieves all that matches with our question. This reduces the size of the content that we pass to LLM.  The next sequence of steps is the process of calling the LLM.
#As you can observe its a long sequence of steps, but lets keep moving forward to see the result that LLM generates. We create a smaller vector db, initialise the LLM, build the prompt template, call the llm using the prompt and question. Once you are done, change the question and see what response you get.
smalldb = Chroma.from_documents(documents=docs,embedding=embedding)

llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=openai_api_key)

from langchain.prompts import PromptTemplate

 

# Build prompt

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum. Keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer. {context} Question: {question} Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain

qa_chain = RetrievalQA.from_chain_type( llm, retriever=smalldb.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})

print(result["result"])