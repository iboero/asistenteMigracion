
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

for m in manuales:
    if m.title == "Manual Proceso de Migracion":
        manual_general = m


with open('bandejas.dill', 'rb') as file:
    bandejas = dill.load(file)
# DEFINIR TOOLS


def remover_tildes(input_str):
    # Normalizar la cadena de texto a 'NFD' para descomponer los acentos
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filtrar para quitar los caracteres de combinación (diacríticos)
    return ''.join(c for c in normalized_str if unicodedata.category(c) != 'Mn')


## FIELDS TOOL

class desc_campo(BaseModel):
    information: str = Field(description="Information that wants to be stored in a field.")
    system = str = Field(description="Filter the search to a particular system. The possible systems are: ['Cuentas Vistas', 'Cuentas y Personas', 'Chequeras', 'Depósitos', 'Microfinanzas', 'Saldos Iniciales', 'Facultades', 'Líneas de Crédito','Garantías', 'Préstamos', 'Acuerdos de Sobregiro', 'Tarjetas de Débito', 'Descuentos', 'all']", default="all")

@tool("retrieve_records_from_description",args_schema=desc_campo)
def retrieve_records_from_description(information:str, system:str="all"):
    """Migrating to Bantotal requieres the user to fill in a lot of information. This tool helps you find the records that are somehow related to the information you have."""
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
        string += f'{{ "Campo": "{md["campo"]}", "Descripción": "{md["content"]}",  "Bandeja/Tabla": "{md["bandeja"]}", "Tipo": "{md["tipo"]}", "Sistema": {md["manual"].replace("Manual ", "")} }} \n\n'
        # si md bandeja no esta en bdjs la agrego
        if md["bandeja"] not in bdjs:
            bdjs.append(md["bandeja"])
    tray_string = f"Trays: \n"
    for manual in manulales:
        for bandeja in manulales[manual].bandejas:
            if bandeja.codigo in bdjs:
                tray_string += f'{{ "Bandeja/Tabla": "{bandeja.codigo}", "Descripción": "{bandeja.descripcion}", "Sistema": "{manual.replace("Manual ", "")}" }} \n\n'
    return tray_string + "\n\n Suitable Fields: \n"+string



# SISTEM TOOL
class system_retriever_input(BaseModel):
    system: str = Field(description="System to retrieve information from. The possible systems are: ['Cuentas Vistas', 'Cuentas y Personas', 'Chequeras', 'Depósitos', 'Microfinanzas', 'Saldos Iniciales', 'Facultades', 'Líneas de Crédito','Garantías', 'Préstamos', 'Acuerdos de Sobregiro', 'Tarjetas de Débito', 'Descuentos']")
    # retrieve_sec_1: bool = Field(description="Retrieve outline of this particular sistem migration proccess", default=False)
    # retrieve_sec_2: bool = Field(description="Retrieve detailed information about the trays and records that store the information of the particular system", default=False)
    # retrieve_sec_3: bool = Field(description="Retrieve information about the transactions linked to the trays of the system", default=False)



@tool("retrieve_sistem_migration_information",args_schema=system_retriever_input)
def retrieve_sistem_migration_information(system:str, retrieve_sec_1: bool = False, retrieve_sec_2: bool = False, retrieve_sec_3: bool = False):
    """Retrieves content relevant to the migration of a particular sistems. The content consists of an outline of the migration of the system (trays and programs of the system, transactions, previous requieremnts, possible errors, among other things)"""
    total_token = 0
    content = ""
    print(remover_tildes(system.lower()))
    for m in manuales:
        print(m.title.lower())
        if m.title.lower() == "manual " + remover_tildes(system.lower()):
            manual = m
            break
    
    retrieved_secs_filt = ["1", "3"]
    # if retrieve_sec_1:
    #     retrieved_secs_filt.append("1")
    # if retrieve_sec_2:
    #     retrieved_secs_filt.append("2")
    # if retrieve_sec_3:
    #     retrieved_secs_filt.append("3")

    for s in retrieved_secs_filt:
        new_sect = manual.get_section(s)
        token = len(encoding.encode(new_sect))
        print(token)
        if token + total_token < 11000:
            content += new_sect + "\n\n"
            total_token += token
        else:
            print("Section {s} was not included in the answer because it exceeded the token limit")
    return content


# GENERAL INFORMATION TOOL
class general_retriever_input(BaseModel):
    retrieve_sec_1: bool = Field(description="Retrieve a breif introduction of the concepts related to the mirgation process", default=False)
    retrieve_sec_2: bool = Field(description="Retrieve an overview of how the migration process works. ", default=False)
    retrieve_sec_3: bool = Field(description="Retrieve a detailed guide on how to use the Bantotal GUI to put into practice the migration process.", default=False)
    retrieve_sec_4: bool = Field(description="Retrieve detailed information of the parameters that customize the migration process.", default=False)



@tool("retrieve_migration_process_information",args_schema=general_retriever_input)
def retrieve_migration_process_information(retrieve_sec_1: bool = False, retrieve_sec_2: bool = False, retrieve_sec_3: bool = False, retrieve_sec_4: bool = False):
    """Retrieves information relevant to the migration process. The informations is grouped in four setions.  """
    total_token = 0
    content = ""

    
    retrieved_secs_filt = []
    if retrieve_sec_1:
        retrieved_secs_filt.append("1")
    if retrieve_sec_2:
        retrieved_secs_filt.append("3")
    if retrieve_sec_3:
        retrieved_secs_filt.append("2")
    if retrieve_sec_4:
        retrieved_secs_filt.append("4")

    for s in retrieved_secs_filt:
        new_sect = manual_general.get_section(s)
        token = len(encoding.encode(new_sect))
        print(token)
        if token + total_token < 11000:
            content += new_sect + "\n\n"
            total_token += token
        else:
            print("Section {s} was not included in the answer because it exceeded the token limit")
    return content



class tray_retriever_input(BaseModel):
    tray: str = Field(description="Tray that wants to be retrieved the fields from.")
    system: str = Field(description="System the tray belongs to. The possible systems are: ['Cuentas Vistas', 'Cuentas y Personas', 'Chequeras', 'Depósitos', 'Microfinanzas', 'Saldos Iniciales', 'Facultades', 'Líneas de Crédito','Garantías', 'Préstamos', 'Acuerdos de Sobregiro', 'Tarjetas de Débito', 'Descuentos']", default="all")



@tool("retrieve_fields_from_tray",args_schema=tray_retriever_input)
def retrieve_fields_from_tray(tray:str, system:str="all"):
    """Retrieves the fields from a tray with a brief description of them."""
    total_token = 0
    content = ""
    tray = tray.upper()
    system = "Manual " + remover_tildes(system)
    if system != "all":
        try:
            content += f"Bandeja: {tray}\n Descripcion:{str(bandejas[system][tray]['descripcion'])} \n Registros: {str(bandejas[system][tray]['registros'])}\n"
        except:
            content += "No se encontraron registros para la bandeja solicitada"
    else:
        for s in bandejas:
            try:
                content += f"Bandeja: {tray}\n Descripcion:{str(bandejas[s][tray]['descripcion'])} \n Registros: {str(bandejas[s][tray]['registros'])}\n"
            except:
                continue
    return content






tools = [retrieve_migration_process_information, retrieve_sistem_migration_information, retrieve_fields_from_tray, retrieve_records_from_description]




# DEFINIR AGENTE

openai = ChatOpenAI(temperature=0.0,max_tokens=4096)




chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """Task: 

You are a helpful assistant, expert on the migration process to the software Bantotal. You must answer users question IN SPANISH. 

Here is a brief description of the elements of the migration process to Bantotal:

Control and Dump Programs:
There are two clearly differentiated types of programs in the System:
    Control Programs: These verify that the data to be recorded in the tables are suitable. They indicate each record with a status code (whether it is valid or not).
    Dump Programs: These take the information controlled by the Control Programs and record it in the Bantotal.

Trays:
The Control and Dump programs act on specific records, which are identified as belonging to a Tray. In other words, a Tray identifies a set of records, which may belong to one or more bantotal tables and be "views" of these.
Trays are created specifically for each client according to their needs. It is understood that a Tray gathers a set of operations of the same type of product (e.g., Savings Accounts).

Systems:
The mentioned trays are grouped into specific themes, called Systems.
         
Transaction:
In many Migrations, the result of the Dump program consists of generating data in various tables and generating an accounting entry. An accounting entry is associated with a transaction, which defines its behavior.
Each Tray is associated with a transaction, and with this, the corresponding accounting entry is generated.
         

You must follow this instructions to answer the user questions: 

1) All information in your answers MUST BE RETRIEVED from the use of the tools. DO NOT MAKE INFORMATION UP. 

2) The user must be transparent to the tools you are using to retrieve the information. If a tool needs to be used, use it without consulting the user. 

3) Be detailed in your answers but do NOT be redundant. Add all details that are useful to provide a complete answer, but do not add details beyond the scope of the question or are not present in the retrierved text.

4) All tools can be used as many times as needed. If you are not able to provide a complete answer with the output of one tool, keep using tools until you can provide a complete answer.

5) If for any reason the question can´t be answered, or is irrelevant to the topic, honestly say that you can not answer that question. UNDER NO CIRCUNSTANCE MAKE INFORMATION UP

"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
) 



agent = create_openai_tools_agent(openai, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
