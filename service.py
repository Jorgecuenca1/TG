import locale
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain.prompts import PromptTemplate
from textwrap import fill
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredPDFLoader
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Install dependencies
def install_dependencies():
    import subprocess
    import sys
    requirements = [
        "transformers==4.37.2",
        "optimum==1.12.0",
        "auto-gptq",
        "langchain==0.1.9",
        "sentence_transformers==2.4.0",
        "unstructured",
        "pdf2image",
        "pdfminer.six==20221105",
        "unstructured-inference",
        "faiss-gpu==1.7.2",
        "pikepdf==8.13.0",
        "pypdf==4.0.2",
        "pillow_heif==0.15.0"
    ]
    for requirement in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

# Uncomment to install dependencies
# install_dependencies()

# Load Llama 2 model
model_name = "TheBloke/Llama-2-13b-Chat-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
gen_cfg = GenerationConfig.from_pretrained(model_name)
gen_cfg.max_new_tokens = 512
gen_cfg.temperature = 0.0000001
gen_cfg.return_full_text = True
gen_cfg.do_sample = True
gen_cfg.repetition_penalty = 1.11

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=gen_cfg
)

llm = HuggingFacePipeline(pipeline=pipe)

# Test LLM with Llama 2 prompt structure and LangChain PromptTemplate
template = """
<s>[INST] <<SYS>>
You are an AI assistant. You are truthful, unbiased and honest in your response.

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

{text} [/INST]
"""

prompt = PromptTemplate(input_variables=["text"], template=template)
text = "Explain artificial intelligence in a few lines"
result = llm(prompt.format(text=text))
print(fill(result.strip(), width=100))

locale.getpreferredencoding = lambda: "UTF-8"

# RAG from web pages
web_loader = UnstructuredURLLoader(urls=["https://en.wikipedia.org/wiki/Solar_System"], mode="elements", strategy="fast")
web_doc = web_loader.load()
updated_web_doc = filter_complex_metadata(web_doc)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunked_web_doc = text_splitter.split_documents(updated_web_doc)
print(f"Number of chunks: {len(chunked_web_doc)}")

# Create vector database with FAISS
embeddings = HuggingFaceEmbeddings()
db_web = FAISS.from_documents(chunked_web_doc, embeddings)

# Use RetrievalQA chain
prompt_template = """
<s>[INST] <<SYS>>
Use the following context to Answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question. Don't try to make up an answer.

<</SYS>>

{context}

Question: {question} [/INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
Chain_web = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_web.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

# Queries
queries = [
    "When was the solar system formed?",
    "Explain in detail how the solar system was formed.",
    "Why do the planets orbit the Sun in the same direction that the Sun is rotating?",
    "What are the planets of the solar system composed of? Give a detailed response.",
    "How does the tranformers architecture work?"
]

for query in queries:
    result = Chain_web.invoke(query)
    print(fill(result['result'].strip(), width=100))

# RAG from PDF Files
# Uncomment the following lines to download and load PDF files
#!gdown "https://github.com/muntasirhsn/datasets/raw/main/Solar-System-Wikipedia.pdf"
pdf_loader = UnstructuredPDFLoader("Solar-System-Wikipedia.pdf")
pdf_doc = pdf_loader.load()
updated_pdf_doc = filter_complex_metadata(pdf_doc)

# Split the document into chunks
chunked_pdf_doc = text_splitter.split_documents(updated_pdf_doc)
print(f"Number of chunks: {len(chunked_pdf_doc)}")

# Create the vector store
db_pdf = FAISS.from_documents(chunked_pdf_doc, embeddings)

# RAG with RetrievalQA
Chain_pdf = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_pdf.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

for query in queries:
    result = Chain_pdf.invoke(query)
    print(fill(result['result'].strip(), width=100))
