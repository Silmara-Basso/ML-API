from fastapi import FastAPI, UploadFile, File, Form
from sil_modelo import sil_pipeline_modelo
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Define uma rota raiz que retorna uma mensagem
@app.get("/")
def inicio():
    return {"sil": "Projeto5"}

# Raiz: "http://localhost:3310/"
# API:  "http://localhost:3310/api"


@app.post("/api")
async def api(text: str = Form(...), image: UploadFile = File(...)):

    # Lê e abre o conteúdo da imagem enviada
    image_contents = await image.read()
    image = Image.open(io.BytesIO(image_contents))

    # Chama a função do modelo, passando o texto e a imagem processada, e armazena o resultado
    resultado = sil_pipeline_modelo(text, image)

    # Retorna o resultado processado pelo modelo em um dicionário JSON
    return {"Resposta do Modelo": resultado}

# Inicia o servidor WSGI Uvicorn com a API 
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3310)

