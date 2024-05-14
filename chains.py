
from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client
from langchain.docstore.document import Document

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent, create_tool_calling_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.output_parsers  import  JsonOutputParser
from typing import List, Tuple, Dict, Type, Any

from langchain.pydantic_v1 import BaseModel, Field, create_model
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import unicodedata
import dill
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

import tiktoken

from azure.search.documents.models import VectorFilterMode
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

## INICIAR LANGSMITH Y API KEYS
dotenv_path = here() / ".env"
load_dotenv(dotenv_path=dotenv_path)


client = Client()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"API - {unique_id}"


# LEVANTAR DATOS


path = "./Manuales_migracion/"
manuales = []
for filename in os.listdir(path):
    with open(path + filename, 'rb') as file:
        manuales.append(dill.load(file))

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")


index_name = "migracion_secciones_manuales"
index_name_2 = "migracion_campos"

credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))
endpoint = os.getenv("AZURE_AI_SEARCH_SERVICE_NAME")


sc = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
sc_campo = SearchClient(endpoint=endpoint, index_name=index_name_2, credential=credential)

with open("./data/manuales_object.pkl", 'rb') as f:
    manulales = dill.load(f)

# DEFINIR TOOLS


def remover_tildes(input_str):
    # Normalizar la cadena de texto a 'NFD' para descomponer los acentos
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filtrar para quitar los caracteres de combinación (diacríticos)
    return ''.join(c for c in normalized_str if unicodedata.category(c) != 'Mn')



class desc_campo(BaseModel):
    information: str = Field(description="Information that wants to be stored in a field.")
    system = str = Field(description="Filter the search to a particular system. The possible systems are: ['Cuentas Vistas', 'Cuentas y Personas', 'Chequeras', 'Depósitos', 'Microfinanzas', 'Saldos Iniciales', 'Facultades', 'Líneas de Crédito','Garantías', 'Préstamos', 'Acuerdos de Sobregiro', 'Tarjetas de Débito', 'Descuentos', 'all']", default="all")

@tool("retrieve_fields_from_description",args_schema=desc_campo)
def retrieve_fields_from_description(information:str, system:str="all"):
    """Retrieves probable fields suitable to store the information."""
    embedding = embeddings.embed_query(information)
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=20, fields="content_vector", exhaustive=True)

    if system != "all":
        filter=f"manual eq 'Manual {system}'"
    else:
        filter=None

    results = sc_campo.search(  
        # search_text=query,  
        vector_queries= [vector_query],
        filter=filter,
        select=["content","manual", "campo", "bandeja", "tipo"],
    )

    string = ""
    bdjs = []
    for md in results:
        string += f'{{ "field": "{md["campo"]}", "description": "{md["content"]}",  "tray": "{md["bandeja"]}", "type": "{md["tipo"]}", "sistem": {md["manual"].replace("Manual ", "")} }} \n\n'
        # si md bandeja no esta en bdjs la agrego
        if md["bandeja"] not in bdjs:
            bdjs.append(md["bandeja"])
    tray_string = f"Trays: \n"
    for manual in manulales:
        for bandeja in manulales[manual].bandejas:
            if bandeja.codigo in bdjs:
                tray_string += f'{{ "tray": "{bandeja.codigo}", "description": "{bandeja.descripcion}", "system": "{manual.replace("Manual ", "")}" }} \n\n'
    return tray_string + "\n\n Suitable Fields: \n"+string


class retriever_input(BaseModel):
    question: str = Field(description="Question to answer from the manual, in spanish")
    keywords: List[str] = Field(description="Keywords to directly look for in the contents. Must be a particular element to search for (tray, field, etc)", default=[])


def create_manual_tool(description, manual_pkl):
    def decorator(func):
        func.__doc__ = description
        return tool("retrieve_information_from_" + remover_tildes(manual_pkl.title.replace(" ", "_").lower()), args_schema=retriever_input)(func)

    @decorator
    def retrieve_information(question:str, keywords:List[str] = []):


        # Pure Vector Search
        embedding = embeddings.embed_query(question)
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=4, fields="content_vector", exhaustive=True)

        results = sc.search(  
            # search_text=query,  
            vector_queries= [vector_query],
            filter=f"manual eq '{manual_pkl.title}'",
            select=["seccion", "tokens"],
        )
        retrieved_secs = []
        for r in results:
            retrieved_secs.append(r["seccion"])
        total_token = 0
        sec_kw = []
        for c in manual_pkl.contents.values():
            kw = "BNJ040"
            if kw.lower().strip() in c.lower().strip():
                sec_kw.append(manual_pkl.get_key(c, manual_pkl.contents))
        tot_sec = retrieved_secs + sec_kw
        print(retrieved_secs, sec_kw)
        retrieved_secs_filt = [x for x in tot_sec if not any([x.startswith(y) for y in tot_sec if y!=x])]

        content = ""
        for s in retrieved_secs_filt:
            new_sect = manual_pkl.get_section(s)
            token = len(encoding.encode(new_sect))
            if token + total_token < 12000:
                content += new_sect + "\n\n"
                total_token += token
            else:
                print("Section {s} was not included in the answer because it exceeded the token limit")
        return content
    return retrieve_information
# Example of creating a tool for a specific manual

manual_tools = []
for manual in manuales:
    manual_name = manual.title
    description = f'Retrieves relevant information from the manual using semantic search. The manual has information related to: {manual.description}'
    manual_tools.append(create_manual_tool(description, manual))
    # manual_tools.append(create_manual_tool(description, manual,manual_search_args))


tools = [retrieve_fields_from_description] + manual_tools




# DEFINIR AGENTE

openai = ChatOpenAI(temperature=0.0,max_tokens=4096)




chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: 

You are a helpful assistant, expert on the migration process to the software Bantotal. You must answer users question IN SPANISH. 

Instructions: 

1) All information in your answers MUST BE RETRIEVED from the use of the tools. DO NOT MAKE INFORMATION UP. 

2) In case the question can´t be answered  using the tools provided (It is not relevant to the migration process) honestly say that you can not answer that question.

3) The user must be transparent to the tools you are using to retrieve the information. If a tool needs to be used, use it without consulting the user. 

4) Be detailed in your answers but stay focused to the question. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question.

5) All tools can be used as many times as needed. If you are not able to provide a complete answer with the output of one tool, try using the same tool with different parameters or a new tool.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
) 



agent = create_openai_tools_agent(openai, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
