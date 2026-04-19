from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter # especificamente para separar por markdown com marcação # 

#passo 1
def contrucao_banco_vetorial () :
    print("Lendo arquivo markdown ")
    split = [("#", "Titulo_principal"), ("##", "Titulo_Secundario")]
    markdown_splitter = MarkdownHeaderTextSplitter(split) 

#passo 2
    try : 
        with open("AYRONORYA.md", "r", encoding="utf=8") as md_open : 
            texto_puro = md_open.read()
    except Exception as e :
        print(f"erro, não foi possivel achar o arquivo, {e}" )

#passo 3
    try : 
        chunking_docs = markdown_splitter.split_text(texto_puro)
        print(f"documento passou pelo processo de chunk com sucesso, numero de chunks : {len(chunking_docs)}")
    except Exception as e:
        print(f"houve um erro no processo de chunking {e}")
#passo 4 
    print("acionando modelo de embedding (bert), pode levar algum tempo")
    motor_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("modelo all-MiniLM-L6-v2 implementado com sucesso ")

#passo 5 
    pasta_banco = "./vector_db"

#essa função from_documents, cria o banco chroma preenchendo com as informações abaixo 
    banco_vetorial = Chroma.from_documents( 
            documents=chunking_docs,
            embedding=motor_embeddings,
            persist_directory=pasta_banco
    )

if __name__ == "__main__":
    contrucao_banco_vetorial()