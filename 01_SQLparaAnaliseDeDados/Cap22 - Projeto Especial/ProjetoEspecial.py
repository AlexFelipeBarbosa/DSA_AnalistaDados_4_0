# Instalando os pacotes 
# pip install -q langchain==0.1.13
# pip install -q pymupdf==1.23.19
# pip install -q huggingface-hub==0.20.3
# pip install -q faiss-cpu==1.7.4
# pip install -q sentence-transformers==2.2.2
# pip install -q openai==1.14.3

import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import langchain
import textwrap
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import warnings
from dotenv import load_dotenv
import time
#import openai


warnings.filterwarnings('ignore')



# 3 - Função para carregar o PDF
def dsa_carrega_pdf(file_path):
  
  # Cria uma instancia da classe PyMuPDFLoader, passando o caminho do PDF como argumento
  loader = PyMuPDFLoader(file_path=file_path)
  
  # Usa o metodo 'load' do objeto 'loader' para carregar o conteudo do PDF
  # Isso retorna um objeto ou uma estrutura de dados contendo as paginas do PDF com seu conteudo
  docs = loader.load()
  
  # Retorna o conteudo carregado do PDF
  return docs

# 4 - Função para dividir os documentos em varios pedaços (chunks)
def dsa_split_docs(documents, chunk_size = 1000, chunk_overlap = 200):

  # Cria uma instância da classe RecursiveCharacterTextSplitter
  # Esta classe divide textos longos em pedaços menores (chunks)
  # 'chunk_size' define o tamanho de cada pedaço e 'chunk_overlap' define a sobreposição entre pedaços consecutivos
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
  
  # Utiliza o método 'split_documents' do objeto 'text_splitter' para dividir o documento fornecido
  # 'documents' é uma variavel que contém o texto ou conjunto de textos a serem divididos
  chunks = text_splitter.split_documents(documents= documents)
  
  # Retorna os pedaços de texto resultantes da divisão
  return chunks


# 5 - Carrega o modelo de embeddings
def dsa_carrega_embedding_model(model_path, normalize_embedding=True):
  
  # Retorna uma instância da classe HuggingFaceEmbeddings.
  # 'model_name' é o identificador do modelo de embeddings a ser carregado.
  # 'model_kwargs' é um dicionario de argumentos adicionais para a configuração do modelo, neste caso, definindo o dispositivo para 'cpu'
  # 'encode_kwargs' pe um dicionario de argumentos para o metodo de codificação, aqui especificando se os embeddings devem ser normalizados.
  return HuggingFaceEmbeddings(model_name = model_path,
                               model_kwargs = {'device':'cpu'},
                               encode_kwargs = {'normalize_embeddings': normalize_embedding})

  
# 6 - Função para criar embeddings
def dsa_cria_embeddings(chunks, embedding_model, storing_path = 'modelo/vectorstore'):
  
  # Cria um 'vectorstore' (um indice FAISS) a partir dos documentos fornecidos.
  # 'chunks' é uma lista de segmentos de texto e 'embedding_model' é o modelo utilizado para converter um texto em embeddings.
  vectorstore = FAISS.from_documents(chunks, embedding_model)
  
  # Salva o 'vectorstore' criado em um caminho local especificado por 'storing_path'.
  # Isso permite a persistencia do indice FAISS para o uso futuro.
  vectorstore.save_local(storing_path) 
  
  # Retorna o 'vectorstore' criado, que contém os embeddings e pode ser usado para operações de busca e comparação de similaridade
  return vectorstore


# 7 - Criando a Chain
def dsa_load_qa_chain(retriever, llm, prompt):
  
  # Retorna uma instância da classe RetrievalQA.
  # Esta função lista com a cadeia de processos envolvidos em um sistema de Question Answering (QA).
  # 'llm' refere-se ao modelo de linguagem de grande escala (como modelo GPT ou BERT).
  # 'retriever' é um componente usado para recuperar informações relevantes (como um mecanismo de busca ou um retriever de documentos).
  # 'chain_type' define o tipo de cadeia ou estratégia usada no processo de QA. Aqui, está definido como 'stuff', um placeholder para um tipo real.
  # 'return_source_documents' um booleano que, quando True, indica que os documentos fonte (ou seja, os documentos de onde as respostas são extraidas) devem ser retornados juntamente com as respostas.
  # 'chain_type_kwargs' é um dicionario de argumentos adicionais especificos para o tipo de cadeia escolhido. Aqui, está passando 'prompt' como argumento.
  return RetrievalQA.from_chain_type(llm = llm,
                                      retriever = retriever,
                                      chain_type = 'stuff',
                                      return_source_documents = True,
                                      chain_type_kwargs = {'prompt':prompt})
  
  
 # 8 - Função para obter as respostas do LLM (Large Language Model) 
def dsa_get_response(query, chain):
   
   
  # Invoca a 'chain' (cadeia de processamento, um pipeline de Question Answering) com a 'query' fornecida.
  # 'chain' é uma função que recebe uma consulta e retorna uma resposta, utilizando LLM.
  response = chain({'query': query})
  
  # Utiliza a biblioteca textwrap para formatar a resposta. 'textwrap.fill' quebra o texto da resposta em linhas de largura especificada (100 caracteres neste caso) ,
  # tornando mais facil a leitura em ambientes como o Jupyter Notebook.
  wrapped_text = textwrap.fill(response['result'], width=100)
  
  # Imprime o texto formatado
  print(wrapped_text)

# 10 - Definindo a API da OpenAI
load_dotenv()
api_key = os.getenv("API_KEY")
llm_api = OpenAI(api_key=api_key)


# 11 - Carrega o modelo Embedding
embed = dsa_carrega_embedding_model(model_path = "all-MiniLM-L6-v2")


# 12 - Carrega o arquivo PDF
docs = dsa_carrega_pdf(file_path = 'PythonParaDataScience.pdf') 


# 13 - Divide o arquivo em Chunks
documents = dsa_split_docs(documents = docs)


# 14 - Cria o vectorstore
vectorstore = dsa_cria_embeddings(documents, embed)


# 15 - Converte o vectorstore para um retriever
retriever = vectorstore.as_retriever()


# 16 - Cria o template
template = """
### System:
Vocé é um assistente pessoal. Você tem que responder as perguntas do usuário \
usando apenas o contexto fornecido a você. Se você não sabe a resposta, \
apenas diga que você não sabe. Não tente inventar uma resposta.

### Context:
{context} 

### User:
{question}

### Response: 
"""

# 17 - Criando o prompt a partir do template
prompt = PromptTemplate.from_template(template)


# 18 - Criando a chain (pipeline)
dsa_chain = dsa_load_qa_chain(retriever, llm_api, prompt)



# Aguardar quando houver "estouro de cota"
def safe_get_response(query, chain):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chain({'query': query})
            return response
        except OpenAI.error.RateLimitError as e:
            print(f"Rate limit exceeded. Attempt {attempt + 1}/{max_retries}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)  # Exponencial backoff
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return None



# 19 - Interagindo com o assistente
response = safe_get_response("Quais são os tipos de dados no Python?", dsa_chain)
if response:
    dsa_get_response("Quais são os tipos de dados no Python?", dsa_chain)