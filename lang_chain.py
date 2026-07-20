from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image

llm = ChatGoogleGenerativeAI(
    model=GEMINI_FLASH,
    api_key=GEMINI_API_KEY,
)

imagen = encode_image('./datos/ejemplo_grafico.jpg')

pregunta = "Describe la imagen: "

mensaje = HumanMessage(
  content=[
    {
      "type": "text",
      "text": pregunta
    },
    {
      "type": "image_url",
      "image_url": {
        "url": f"data:image/png;base64,{imagen}"
      }
    }
  ]
)

respuesta = llm.invoke([mensaje])

print(respuesta)