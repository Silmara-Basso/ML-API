# Importa a classe ViltProcessor para processamento de imagem e texto, e ViltForQuestionAnswering para o modelo de QA
from transformers import ViltProcessor, ViltForQuestionAnswering

# Importa a biblioteca PIL para manipulação de imagens
from PIL import Image

# Carrega o processador pré-treinado específico para tarefas de QA 
sil_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Carrega o modelo pré-treinado para responder a perguntas baseadas em imagens e texto
sil_model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

# Define uma função pipeline para processar texto e imagem e obter uma resposta
def sil_pipeline_modelo(text:str, image:Image):
    encoding = sil_processor(image, text, return_tensors = "pt")
    outputs = sil_model(**encoding)
    logits = outputs.logits
    index = logits.argmax(-1).item()

    # Retorna a etiqueta associada ao índice de maior pontuação como a resposta
    return sil_model.config.id2label[index]
