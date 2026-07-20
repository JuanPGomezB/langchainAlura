from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from my_models import GEMINI_FLASH, COHERE_COMMAND
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from my_helper import encode_image

llm = ChatGoogleGenerativeAI(
    model=GEMINI_FLASH,
    api_key=GEMINI_API_KEY,
)

respuesta=llm.invoke("Cuáles canales colombianos de Youtube me recomiendas para saber más sobre teléfonos inteligentes")

texto_respuesta = respuesta.content[0]["text"]

print(f"Gemini: {texto_respuesta}")

llm2=ChatCohere(
    model=COHERE_COMMAND,
    cohere_api_key=COHERE_API_KEY
)

respuesta2=llm2.invoke([HumanMessage(content="Cuáles canales colombianos de Youtube me recomiendas para saber más sobre teléfonos inteligentes")])

print(f"Cohere: {respuesta2.content}")

# imagen = encode_image('./datos/ejemplo_grafico.jpg')

# pregunta = "Describe la imagen: "

# mensaje = HumanMessage(
#   content=[
#     {
#       "type": "text",
#       "text": pregunta
#     },
#     {
#       "type": "image_url",
#       "image_url": {
#         "url": f"data:image/png;base64,{imagen}"
#       }
#     }
#   ]
# )

# respuesta = llm.invoke([mensaje])

# print(respuesta)