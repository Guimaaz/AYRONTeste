from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import PROMPT_JUIZ, PROMPT_ARQUITETO

def chat_norya():
    print("\n SISTEMA NORYA: TESTE DE PROTOCOLOS ")
    motor_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    banco_vetorial = Chroma(persist_directory="./vector_db", embedding_function=motor_embeddings)
    buscador = banco_vetorial.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="llama3", temperature=0.3)


    prompt_juiz = PromptTemplate.from_template(PROMPT_JUIZ)
    juiz_chain = prompt_juiz | llm | StrOutputParser()

    prompt_arquiteto = PromptTemplate.from_template(PROMPT_ARQUITETO)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": buscador | format_docs, "input": RunnablePassthrough()}
        | prompt_arquiteto
        | llm
        | StrOutputParser()
    )

    print("[ STATUS: ONLINE ] - Arquiteto pronto para o trabalho.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ")
        if pergunta.lower() in ['sair', 'exit', 'parar']: break
        print("AYRON avaliando escopo...", end="\r")
        decisao = juiz_chain.invoke({"input": pergunta}).strip().upper()
        print(" " * 50, end="\r") 
        if decisao == "SIM" or "SIM" in decisao:
            print("AYRON consultando manuais...", end="\r")
            resultado = rag_chain.invoke(pergunta)
            print(" " * 50, end="\r")
            print(f"AYRON: {resultado}\n")
        else:
            print("AYRON: Sou um modelo especializado pela NORYA, meu conhecimento é focado e direcionado para ajudar e estruturar suas ideias.\n")

if __name__ == "__main__":
    chat_norya()