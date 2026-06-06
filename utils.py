from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import tempfile
import os


class DashScopeEmbeddings(Embeddings):
    """DashScope Embeddings 封装（兼容 LangChain 接口）"""

    def __init__(self, api_key, model="text-embedding-v3"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model

    def embed_documents(self, texts):
        all_embeddings = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

    def embed_query(self, text):
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


def process_pdf_to_retriever(uploaded_file, api_key):
    """
    解析 PDF 并构建 FAISS 向量库（只需调用一次）。
    返回 (retriever, num_chunks)。
    """
    file_content = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "！", "？", "，", "、", ""],
        )
        texts = text_splitter.split_documents(docs)

        if not texts:
            raise ValueError("PDF 解析后没有可用的文本内容")

        embeddings_model = DashScopeEmbeddings(api_key=api_key)
        db = FAISS.from_documents(texts, embeddings_model)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        return retriever, len(texts)
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def ask_question(question, retriever, api_key, chat_history=None):
    """
    基于已构建的向量库进行问答。
    返回 (answer_stream, retrieved_docs)：
      - answer_stream: 可被 st.write_stream() 消费的生成器
      - retrieved_docs: 检索到的相关文档列表
    """
    llm = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.5,
        request_timeout=60,
        max_retries=3,
        streaming=True,
    )

    # 检索相关文档
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 构建历史消息（限制最近 10 轮 = 20 条消息）
    history_messages = []
    if chat_history:
        history_messages = chat_history[-20:]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个专业的问答助手。请结合以下检索到的上下文信息，用中文回答用户的问题。"
            "如果你不知道答案，就说'根据提供的文档，无法确定确切答案'，不要编造信息。\n\n"
            "以下是相关的上下文信息：\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    # 返回流式输出和参考文档
    stream = chain.stream({
        "context": context,
        "chat_history": history_messages,
        "question": question,
    })

    return stream, retrieved_docs
