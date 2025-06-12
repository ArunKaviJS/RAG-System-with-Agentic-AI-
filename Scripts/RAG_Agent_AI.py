import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

# ---------- Streamlit Page Configuration ---------- #
st.set_page_config(page_title="📚 Smart Document QA", layout="wide")


# Custom HTML and CSS injected with st.markdown


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
            color: #2c3e50; /* Professional dark navy */
            margin-bottom: 0.5rem;
        }

        .subtitle-text {
            font-size: 1.1rem;
            font-weight: 400;
            color: #34495e; /* Subtle dark gray */
        }
    </style>

    <div class="title-container">
        <div class="title-text">📚 Smart Document QA</div>
        <div class="subtitle-text">Ask your documents smart questions in India History!</div>
    </div>
""", unsafe_allow_html=True)

# ---------- Load Vector DB and LLM ---------- #


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = FAISS.load_local(r"index", embedding_model, allow_dangerous_deserialization=True)

model_id = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# ---------- Prompt Template ---------- #
template = """
You are a helpful assistant. Use the following context to answer the question concisely in 4 to 5 lines.
If you don't know the answer, just say you don't know. Don't make anything up.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# ---------- Input & Output ---------- #
user_query = st.text_input("Enter your question")

if st.button("Get Answer") and user_query:
    response = qa_chain.run(user_query)

    # Clean the response to extract only the helpful answer
    if "Helpful Answer:" in response:
        answer = response.split("Helpful Answer:")[1].strip()
    else:
        answer = response.strip()

    st.success("Answer:")
    st.write(answer)

    with st.expander("🧠 Raw LLM Response"):
        st.write(response)
