import os 
from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document as docs
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableLambda
import dill as pkl
from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client
import prompts as p
import dill as pkl

## INICIAR LANGSMITH Y API KEYS
dotenv_path = here() / ".env"
load_dotenv(dotenv_path=dotenv_path)


client = Client()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = 'Asistente Migracion'#f"Migracion - {unique_id}"


## LEVANTAR BASE DE DATOS Y LLM
db_ret = Chroma(persist_directory="data/db_campos", embedding_function=OpenAIEmbeddings())
filename = 'data/dict_arbol_secciones.pkl'
with open(filename, 'rb') as file:
    trees = pkl.load(file)

llm = ChatOpenAI(temperature=0)
#llm = ChatAnthropic(model_name="claude-3-haiku-20240307",temperature=0)


## CHAIN PARA CAMPOS
def parse_campos(i_dict):
    manual = i_dict["manual"]
    pregunta = i_dict["pregunta"]
    campos =  db_ret.similarity_search(pregunta,filter={"manual":manual},k=10)
    string = ""
    for c in campos:
        md = c.metadata
        string += f'{{ "registro": "{md["campo"]}", "descripcion_registro": "{c.page_content}",  "bandeja_del_registro": "{md["bandeja"]}", "tipo_del_registro" "{md["tipo"]}" }} \n\n'
    return string



prompt_campos = p.prompt_campos


filename = 'data/manuales_object.pkl'
with open(filename, 'rb') as file:
    manuales_obj = pkl.load(file)

def get_desc_bandejas(manual):
    manual = manuales_obj[manual]
    string = ""
    for b in manual.bandejas:
        string +=  b.codigo + ': ' + b.descripcion + '\n'
    return string


chain_campo = {"campos":{"pregunta": itemgetter("pregunta"), "manual": itemgetter("manual")}| RunnableLambda(parse_campos),"desc_bandejas":itemgetter("manual") | RunnableLambda(get_desc_bandejas), "pregunta":itemgetter("pregunta")} | prompt_campos | llm | StrOutputParser()


### CHAIN GENERAL

prompt_rag = p.prompt_rag
prompt_seccion = p.prompt_seccion

def content_from_section(dicto):
    nro_seccion = dicto["seccion"]["numero_seccion"]
    manual = dicto["manual"]
    num = nro_seccion.split('.')
    seed = trees[manual]
    for x  in num:
        seed = seed.childrens[int(x)-1]
    content = f"Seccion {nro_seccion}: {seed.section} \n\n {seed.content} + \n\n"
    for idx,child in enumerate(seed.childrens):
        content += f"Seccion {nro_seccion}.{idx+1}: {child.section} \n\n {child.content} + \n\n"
    return content

def get_indice(manual):
    return trees[manual].print_index_w_content()


chain_get_section = {"pregunta":itemgetter("pregunta"),"secciones": itemgetter("manual") | RunnableLambda(get_indice)} | prompt_seccion | llm | JsonOutputParser()

chain_get_answer = {"pregunta": itemgetter("pregunta"), "contenido": {"seccion":chain_get_section, "manual": itemgetter("manual")} | RunnableLambda(content_from_section)} | prompt_rag | llm | StrOutputParser()
