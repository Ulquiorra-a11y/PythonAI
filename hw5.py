from dotenv import load_dotenv
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

load_from_url = WebBaseLoader('https://itcareerhub.de')

docs = load_from_url.load()

prompt = ChatPromptTemplate.from_template('Сделай-ка краткое описание указанной веб-страницы :).'
                                          ' Веб-страница: {context}')


# )

chain = create_stuff_documents_chain(llm=llm, prompt=prompt)


result = chain.invoke({'context': docs})

print(result)