import requests

# URL da API
url = "http://localhost:3310/api"

# Payload (texto de entrada para o LLM)
#payload = {'text': (None, 'Which color is the car in the image?')}
#payload = {'text': (None, 'What is in the top of the building?')}
payload = {'text': (None, 'What is the dog doing in the image?')}

# Imagem enviada para o modelo
#file = [('image', open('imagem1.jpg','rb'))]
#file = [('image', open('imagem2.jpg','rb'))]
file = [('image', open('imagem3.jpg','rb'))]

# Faz a requisição à API e armazena a resposta
headers = {'accept': 'application/json'}
resposta = requests.request("POST", 
                            url, 
                            headers = headers, 
                            data = payload, 
                            files = file)

print(resposta.text)
