#IMPORTS 

from flask import Flask, request, render_template_string, jsonify,Response, stream_with_context, request
import asyncio
from langchain_core.output_parsers import  StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import csv
from chains import chain_campo, chain_get_answer
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate


# LEVANTAR SERVIDOR Y HTML
app = Flask(__name__)

def importar_html_como_string(archivo_html, codificacion='utf-8'):
    with open(archivo_html, 'r', encoding=codificacion) as archivo:
        contenido = archivo.read()
    return contenido

HTML = importar_html_como_string("pagina_web.html"  )

#llm_type = "anthropic"
llm_type = "openai"

def format_chat_history_anthropic(chat_history):
    template = ""
    for message in chat_history:
        if message[0] == "system":
            template += message[1] + '\n\n'
        if message[0] == "human":
            template += "<question>\n"
            template += message[1] + '\n'
            template += "</question>\n"
        if message[0] == "ai":
            template += "<answer>\n"
            template += message[1] + '\n'
            template += "</answer>\n"
    template += "Standalone Question:"
    return PromptTemplate.from_template(template)
def format_chat_history_openai(chat_history):
    return ChatPromptTemplate.from_messages(chat_history)


if llm_type == "anthropic":
    llm = ChatAnthropic(model_name="claude-3-haiku-20240307",temperature=0)
    format_chat_history = format_chat_history_anthropic
else:
    llm = ChatOpenAI(temperature=0.0)
    format_chat_history = format_chat_history_openai

# VARIABLES GLOBALES
selected_option = None
chat_history = [("system", "Given the following question-answer sequence in spanish, rewrite the LAST question in a standalone format. That mean, in a way that the standalone question can be fully understood without the question/answer history. Rewrite it in spanish, not english. In case there is only one question, leave the question as is. DO NOT TRY TO ANSWER THE QUESTION, JUST REWRITE IT'S STANDALONE VERSION")]
ultima_respuesta = ""
ultima_pregunta = ""
ultima_seccion = ""
campo = 0
historial = 0


# FUNCIONES ASOCIADAS A JS

# Levantar servidor
@app.route('/')
def home():
    return render_template_string(HTML)


## Manejar Respuesta
def generate_data(message):
    if historial:
        chain = format_chat_history(chat_history) | llm | StrOutputParser()
        message = chain.invoke({})
    input_graph = {"pregunta":message,"manual":manual}
    mensaje_completo = ""
    if campo:
        for chunk in chain_campo.stream(input_graph):
            mensaje_completo += chunk
            chunk_with_placeholder = chunk.replace("\n", "||")
            yield f"data: {chunk_with_placeholder}\n\n"
    else:
        for chunk in chain_get_answer.stream(input_graph):
            mensaje_completo += chunk
            chunk_with_placeholder = chunk.replace("\n", "||")
            yield f"data: {chunk_with_placeholder}\n\n"
    chat_history.append(("ai",mensaje_completo))
    yield "data: done\n\n"

@app.route('/send', methods=['POST'])
def send():
    global chat_history
    global ultima_pregunta
    global ultima_respuesta
    global ultima_seccion
    data = request.json
    ultima_pregunta = data['message']
    chat_history.append(("human",ultima_pregunta))
    return jsonify({'status': 'success'}), 200


@app.route('/stream')
def stream():
    global ultima_pregunta
    return Response(stream_with_context(generate_data(ultima_pregunta)), mimetype='text/event-stream')


## Borrar historial de chat
@app.route('/new_chat', methods=['POST'])
def handle_new_chat():
    global chat_history
    chat_history = [("system", "Given the following question-answer sequence in spanish, rewrite the LAST question in a standalone format. That mean, in a way that the standalone question can be fully understood without the question/answer history. Rewrite it in spanish, not english. In case there is only one question, leave the question as is. DO NOT TRY TO ANSWER THE QUESTION, JUST REWRITE IT'S STANDALONE VERSION")]
    return jsonify({'status': 'new chat started'})  # Respuesta opcional
# hola

## Cambiar Sistema
@app.route('/update_option', methods=['POST'])
def update_option():
    global selected_option
    global manual
    data = request.json
    selected_option = data['option']
    manual = "Manual "+ selected_option
    print("Opción actualizada a:", selected_option)
    return jsonify({'status': 'success'})


## Cambiar a modo Buscar Campo
@app.route('/update_campo', methods=['POST'])
def update_campo():
    global campo
    data = request.json
    campo = data['toggle']
    print("Toggle actualizado a:", campo)
    return jsonify({'status': 'success'})


## Habilitar Historial de Chat
@app.route('/update_historial', methods=['POST'])
def update_historial():
    global historial
    data = request.json
    historial = data['toggle']
    print("historial actualizado a:", historial)
    return jsonify({'status': 'success'})


## Manejar Feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    global chat_history
    global ultima_pregunta
    global ultima_respuesta
    global ultima_seccion
    data = request.json
    feedback = data['feedback']
    positive = int(data["positive"])*2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta","Respuesta","Seccion Recuperada","Positivo","Comentario", "Historial de chat"])
        writer.writerow([ultima_pregunta,ultima_respuesta,ultima_seccion, positive, feedback, chat_history])
    print([ultima_pregunta,ultima_respuesta, feedback, chat_history])
    return jsonify({'status': 'Feedback recibido'})


# CORRER SERVIDOR
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


