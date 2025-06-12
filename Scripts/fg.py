import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMChain

# ---------- Streamlit Page Configuration ---------- #
st.set_page_config(page_title="\U0001F4DA Smart Document QA", layout="wide")

st.markdown("""
    <style>
        .title-container {
            width: 100%;
            padding: 2rem 1rem 1rem 1rem;
            text-align: center;
        }

        .title-text {
            font-size: 2.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }

        .subtitle-text {
            font-size: 1.1rem;
            font-weight: 400;
            color: #34495e;
        }
    </style>

    <div class="title-container">
        <div class="title-text">\U0001F4DA Smart Document QA</div>
        <div class="subtitle-text">Ask your documents smart questions in India History!</div>
    </div>
""", unsafe_allow_html=True)

# ---------- Load Vector DB and LLM ---------- #
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.load_local("/content/Index", embedding_model, allow_dangerous_deserialization=True)

model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Ensure model runs on CPU to prevent CUDA OOM
model.to("cpu")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, device=-1)
llm = HuggingFacePipeline(pipeline=pipe)

# ---------- Prompt Template ---------- #
prompt = PromptTemplate(
    template="""
You are a helpful assistant. Use the following context to answer the question concisely in 4 to 5 lines.
If you don't know the answer, just say you don't know. Don't make anything up.

Context:
{context}

Question:
{question}

Helpful Answer:
""",
    input_variables=["context", "question"]
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# Define Retriever Tool
retriever_tool = Tool(
    name="Document Retriever",
    func=lambda q: retriever.get_relevant_documents(q),
    description="Useful for fetching relevant context from historical documents."
)

# QA Chain
qa_chain = LLMChain(llm=llm, prompt=prompt)

def qa_agent_func(question):
    context_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    return qa_chain.run({"context": context, "question": question})

# Verifier Tool
def verify_answer(answer):
    if "Helpful Answer:" in answer:
        answer = answer.split("Helpful Answer:")[1].strip()
    return answer.strip() if len(answer.strip()) > 20 and "I don't know" not in answer else "Answer could not be verified reliably."

# ---------- Input & Output ---------- #
user_query = st.text_input("Enter your question")

if st.button("Get Answer") and user_query:
    initial_answer = qa_agent_func(user_query)
    final_answer = verify_answer(initial_answer)

    st.success("Answer:")
    st.markdown("**Question:**")
    st.write(user_query)
    st.markdown("**Answer:**")
    st.write(final_answer)
