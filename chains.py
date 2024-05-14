
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

db = Chroma(persist_directory="C:/Users/iboero/asistentes/generacion_datos/procesamiento_manuales/manuales_database",embedding_function=embeddings)
db_campo = Chroma(persist_directory="C:/Users/iboero/asistentes/asistente_migracion/asistente_migracion_v1/data/db_campos", embedding_function=embeddings)


with open("C:/Users/iboero/asistentes/asistente_migracion/asistente_migracion_v1/data/manuales_object.pkl", 'rb') as f:
    manulales = dill.load(f)

# DEFINIR TOOLS



## CREAR TOOLS DE MANUALS



# class manual_search_args(BaseModel):
#     question: str = Field(description="Question to answer from the manual in SPANISH")


# prompt_elegir_secciones = ChatPromptTemplate.from_messages([
#     ("system", """
# Task: Given the index and question below, identify which sections/subsections of the index are relevant to answer the question.

# Instructions:
#     1) Select relevant sections or subsections based on the question's context. If a section or subsection is selected, do not separately select any of its child subsections, as they are included.
#     2) Limit your selection to a maximum of 5 sections or subsections. There is no minimum requirement.
#     3) Output your answer as a JSON blob with two keys:
#        - "sections_name": a list containing the names of the relevant sections/subsections
#        - "sections_number": a list containing the numbers of the relevant sections/subsections, formatted as STRINGS

# Example Input:
# Index:
#     1. Installation
#         1.1 Prerequisites
#         1.2 Setup Instructions
#     2. Configuration
#         2.1 Modify Config Files
#         2.2 Verify Installation
#         2.2.1 System Check
# Question: "What sections detail the initial installation steps?"

# Example Output:
# {{
#     "sections_name": ["Installation"],
#     "sections_number": ["1"]
# }}

# Index:
# {index}

# Question: {question}
# """)])




# def filter_sections(sections):
#     # Sort sections to ensure parents come before their children
#     sections_sorted = sorted(sections, key=lambda x: list(map(int, x.split('.'))))
    
#     filtered_sections = []
    
#     # Loop through each section
#     for section in sections_sorted:
#         # Generate all possible parent sections
#         parts = section.split('.')
#         possible_parents = ['.'.join(parts[:i]) for i in range(1, len(parts))]
        
#         # Check if any possible parent is in the filtered sections
#         if not any(parent in filtered_sections for parent in possible_parents):
#             filtered_sections.append(section)

#     return filtered_sections


# def create_manual_tool(description, manual_pkl,args_schema):
#     def decorator(func):
#         func.__doc__ = description
#         return tool("retrieve_information_from_" + manual_pkl.title.replace(" ", "_").lower(), args_schema=args_schema)(func)

#     @decorator
#     def retrieve_information(question:str):
#         # Elijo el manual específico
#         selected_manual = manual_pkl
#         indice = selected_manual.get_index()
#         chain = prompt_elegir_secciones | ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0) | JsonOutputParser()
#         secc = chain.invoke({"index":indice, "question":question})
#         filtered_sections = filter_sections(secc["sections_number"])
        
#         # Recupero el contenido de las secciones
#         content = ""
#         for sec in filtered_sections:
#             content += selected_manual.get_section(sec) + '\n\n'
#         return content

#     return retrieve_information



# def class_from_section(sections: List[Tuple[int, str]]) -> Type[BaseModel]:
#     level_1_sections = {name.replace(' ', '_').lower(): (bool, Field(default=False,description=f"Set True to return the content of section '{name}'. Else set to False"))
#                         for level, name in sections if level == 1}
#     class_dict = {}
#     for name in level_1_sections.keys():
#          field = (bool, Field(description=f"Requiered field. Set True to return the content of section '{name}'. Else set to False"))
#          class_dict[name] = field
#     ManualSections = create_model('ManualSections', **class_dict)    
#     return ManualSections



# def create_manual_tool_2(description, manual_pkl,args_schema):
#     def decorator(func):
#         func.__doc__ = description
#         return tool("retrieve_information_from_" + manual_pkl.title.replace(" ", "_").lower(), args_schema=args_schema)(func)

#     @decorator
#     def retrieve_information(**args_schema):
#         # Elijo el manual específico
#         selected_manual = manual_pkl
#         sections = []
#         total_tokens = 0
#         for idx, ret_sec in enumerate(args_schema.values()):
#             if ret_sec:
#                 sections.append(str(idx+1))
#         # Recupero el contenido de las secciones
#         content = ""
#         for sec in sections:
#             ret_sec = selected_manual.get_section(sec) + '\n\n'
#             tokens = len(encoding.encode(ret_sec))
#             if total_tokens + tokens < 13000:
#                 content += ret_sec
#                 total_tokens += tokens
#         return content

#     return retrieve_information
# # Example of creating a tool for a specific manual

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
    if system in ['Cuentas Vistas', 'Cuentas y Personas', 'Chequeras', 'Depósitos', 'Microfinanzas', 'Saldos Iniciales', 'Facultades', 'Líneas de Crédito','Garantías', 'Préstamos', 'Acuerdos de Sobregiro', 'Tarjetas de Débito', 'Descuentos']:
        filt = {"manual":"Manual " + system}
    else:
        filt = {}
    campos =  db_campo.similarity_search(information,k=len(db_campo.get()["ids"]), filter=filt)[:10]
    string = ""
    bdjs = []
    for c in campos:
        md = c.metadata
        string += f'{{ "field": "{md["campo"]}", "description": "{c.page_content}",  "tray": "{md["bandeja"]}", "type": "{md["tipo"]}", "sistem": {md["manual"].replace("Manual ", "")} }} \n\n'
        # si md bandeja no esta en bdjs la agrego
        if md["bandeja"] not in bdjs:
            bdjs.append(md["bandeja"])
    string += f"\n\n Tray info: \n"
    for manual in manulales:
        for bandeja in manulales[manual].bandejas:
            if bandeja.codigo in bdjs:
                string += f'{{ "tray": "{bandeja.codigo}", "description": "{bandeja.descripcion}", "system": "{manual.replace("Manual ", "")}" }} \n\n'
    return string


class retriever_input(BaseModel):
    question: str = Field(description="Question to answer from the manual, in spanish")
    keywords: List[str] = Field(description="Keywords to directly look for in the contents. Must be a particular element to search for (tray, field, etc)", default=[])


def create_manual_tool(description, manual_pkl):
    def decorator(func):
        func.__doc__ = description
        return tool("retrieve_information_from_" + remover_tildes(manual_pkl.title.replace(" ", "_").lower()), args_schema=retriever_input)(func)

    @decorator
    def retrieve_information(question:str, keywords:List[str] = []):
        total_token = 0
        retrieved_secs = [x.metadata["section"] for x in db.similarity_search(question, filter={"title":manual_pkl.title})]
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
