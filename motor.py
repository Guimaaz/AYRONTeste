from langchain_ollama import ChatOllama

def motor():
    print("Iniciando a ignição do Llama-3...")
    llm = ChatOllama(
        model="llama3",
    )
    pergunta = "Responda em uma frase: O que é o projeto NORYA?"
    #pergunta para testar o modelo, provavelmente sem o RAG ele alucinará em alguma resposta sobre siglas ou ongs 
    print(f"Você perguntou: {pergunta}")

    resposta = llm.invoke(pergunta)

    print("Llama-3 respondeu:")
    print(resposta.content)

if __name__ == "__main__":
    motor()


