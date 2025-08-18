import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
import tiktoken
from googleapiclient.discovery import build

# --- 환경변수 로딩 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# --- 구글 검색 함수 ---
def search_google(query: str, num_results: int = 3):
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=num_results).execute()
    return [item["link"] for item in res.get("items", [])]

# --- URL에서 div.subXX_XX_XX 텍스트 추출 ---
def crawl_url(url: str):
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        match = re.search(r"sub\d{2}_\d{2}_\d{2}", url)
        if not match:
            return "❌ URL에서 'subXX_XX_XX' 패턴을 찾을 수 없습니다."
        
        class_name = match.group()
        target_div = soup.find("div", class_=class_name)

        if target_div:
            paragraphs = [p.get_text(strip=True) for p in target_div.find_all("p")]
            return "\n".join(paragraphs) if paragraphs else target_div.get_text(strip=True)

        return "❌ 해당 클래스를 가진 <div>를 찾을 수 없습니다."

    except Exception as e:
        return f"⚠️ 크롤링 에러: {str(e)}"

# --- 텍스트 분할 ---
def split_text(text: str, max_tokens: int = 300):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, chunk, tokens = [], [], 0

    for word in words:
        token_len = len(tokenizer.encode(word))
        if tokens + token_len > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, tokens = [], 0
        chunk.append(word)
        tokens += token_len

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# --- 임베딩 ---
def embed_texts(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [d.embedding for d in response.data]

# --- FAISS 인덱스 ---
def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def search_index(index, query_embedding, texts, top_k=3):
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [texts[i] for i in I[0]]

# --- GPT 응답 ---
def answer_with_context(question, context_chunks):
    context = "\n\n".join(context_chunks)
    system_msg = "아래 정보를 참고해서 질문에 답해주세요:\n" + context
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ]
    )
    return completion.choices[0].message.content

# --- Streamlit UI ---
st.set_page_config(page_title="🔍 검색 기반 GPT 챗봇", layout="wide")
st.title("🔍 검색어 기반 웹크롤링 + GPT 챗봇")

query = st.text_input("💡 검색어를 입력하세요:")
question = st.text_input("💬 질문을 입력하세요:")

if query and question:
    with st.spinner("🔎 구글에서 관련 웹페이지 검색 중..."):
        urls = search_google(query, num_results=3)
        if not urls:
            st.error("❌ 검색 결과가 없습니다.")
            st.stop()
        st.write("🔗 검색된 URL:")
        for i, u in enumerate(urls):
            st.markdown(f"- {i+1}. [{u}]({u})")

    selected_url = urls[0]  # 가장 첫 번째 URL 자동 선택

    with st.spinner("🌐 URL에서 텍스트 추출 중..."):
        extracted_text = crawl_url(selected_url)
        if "❌" in extracted_text or "⚠️" in extracted_text:
            st.error(extracted_text)
            st.stop()
        st.text_area("📄 본문 내용 미리보기", extracted_text[:3000], height=200)

    with st.spinner("🧩 텍스트 임베딩 중..."):
        chunks = split_text(extracted_text)
        if not chunks:
            st.error("❌ 텍스트가 비어 있습니다.")
            st.stop()
        embeddings = embed_texts(chunks)
        index = build_index(embeddings)
        query_embedding = embed_texts([question])[0]
        top_chunks = search_index(index, query_embedding, chunks)

    with st.spinner("🤖 GPT가 답변을 작성 중입니다..."):
        answer = answer_with_context(question, top_chunks)
        st.markdown("### 📢 GPT 답변:")
        st.write(answer)
