import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from utils import process_pdf_to_retriever, ask_question

st.set_page_config(page_title="AI智能PDF问答工具", page_icon="📑")
st.title("📑 AI智能PDF问答工具")

# --- 常量 ---
MAX_FILE_SIZE_MB = 50
MAX_HISTORY_ROUNDS = 10  # 最多保留 10 轮对话

with st.sidebar:
    openai_api_key = st.text_input("请输入DashScope API密钥（通义千问）：", type="password")
    st.markdown("[获取DashScope API Key](https://dashscope.console.aliyun.com/apiKey)")

# --- 初始化 session_state ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "file_id" not in st.session_state:
    st.session_state["file_id"] = None

# --- 上传 PDF 并构建向量库（只处理一次）---
uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")

if uploaded_file:
    # 文件大小校验
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"文件大小 ({uploaded_file.size / 1024 / 1024:.1f}MB) 超过限制 ({MAX_FILE_SIZE_MB}MB)")
        uploaded_file = None
    else:
        st.info(f"文件: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

if uploaded_file and openai_api_key:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    # 文件变化时重新构建向量库
    if st.session_state["file_id"] != file_id:
        with st.spinner("正在解析PDF并构建向量库（首次需要等待）..."):
            start = time.time()
            retriever, num_chunks = process_pdf_to_retriever(uploaded_file, openai_api_key)
            elapsed = time.time() - start
        st.session_state["retriever"] = retriever
        st.session_state["file_id"] = file_id
        st.session_state["chat_history"] = []  # 换文件时清空历史
        st.success(f"文档处理完成！{num_chunks} 个片段，耗时 {elapsed:.1f}秒")
    else:
        st.success("向量库已就绪，直接提问即可")

# --- 提问区域 ---
question = st.text_input(
    "对PDF的内容进行提问：",
    disabled=st.session_state["retriever"] is None,
    placeholder="上传PDF后即可提问...",
)

# --- 执行问答 ---
if question and st.session_state["retriever"] and openai_api_key:
    try:
        with st.spinner("AI正在思考中..."):
            start = time.time()
            stream, retrieved_docs = ask_question(
                question=question,
                retriever=st.session_state["retriever"],
                api_key=openai_api_key,
                chat_history=st.session_state["chat_history"],
            )

            # 流式输出答案
            st.write("### 答案")
            answer = st.write_stream(stream)

            elapsed = time.time() - start
            st.caption(f"响应耗时 {elapsed:.1f}秒")

            # 更新对话历史（限制最大轮数）
            history = st.session_state["chat_history"]
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))
            max_messages = MAX_HISTORY_ROUNDS * 2
            if len(history) > max_messages:
                history = history[-max_messages:]
            st.session_state["chat_history"] = history

        # 显示参考来源
        with st.expander("📚 参考来源"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"**段落 {i+1}**")
                st.text(doc.page_content[:500])
                if i < len(retrieved_docs) - 1:
                    st.divider()

    except Exception as e:
        st.error(f"请求失败: {e}")

elif question and not openai_api_key:
    st.warning("请先输入DashScope API密钥")

# --- 历史消息 ---
if st.session_state["chat_history"]:
    with st.expander(f"历史消息（最近 {len(st.session_state['chat_history']) // 2} 轮）"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_msg = st.session_state["chat_history"][i]
            ai_msg = st.session_state["chat_history"][i + 1]
            st.write(f"**问:** {human_msg.content}")
            st.write(f"**答:** {ai_msg.content}")
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
