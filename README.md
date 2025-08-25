# 🔍 검색 기반 GPT 챗봇

이 프로젝트는 **구글 검색 → 웹페이지 크롤링 → 텍스트 임베딩 → FAISS 검색 → GPT 응답 생성** 과정을 통해  
사용자의 질문에 대해 웹에서 가져온 정보를 바탕으로 답변을 생성하는 **Streamlit 챗봇**입니다.  

---

## 🚀 주요 기능
- 🔎 **구글 검색 API**: 검색어로 관련 URL 가져오기  
- 🌐 **웹 크롤링**: URL에서 `subXX_XX_XX` 패턴에 해당하는 `<div>` 본문만 추출  
- ✂️ **텍스트 분할**: `tiktoken` 기반 토큰 단위 분할  
- 🧩 **임베딩 생성**: OpenAI `text-embedding-3-small` 모델 사용  
- 📚 **FAISS 벡터 검색**: 질문과 가장 관련 있는 문맥 추출  
- 🤖 **GPT 응답 생성**: OpenAI `gpt-4o` 모델로 최종 답변 출력  
- 🎨 **Streamlit UI**: 웹 브라우저에서 간단하게 검색어 & 질문 입력  

---

## 📂 프로젝트 구조
```
project/
│── app.py              # 메인 Streamlit 앱
│── .env                # API 키 저장 (OpenAI, Google API, CSE ID)
```

---

## 🔑 환경 변수 설정 (.env 파일)
프로젝트 루트에 `.env` 파일을 만들고 아래 내용을 입력하세요:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

- `OPENAI_API_KEY`: [OpenAI API](https://platform.openai.com/) 키  
- `GOOGLE_API_KEY`: [Google Cloud Console](https://console.cloud.google.com/)에서 발급  
- `GOOGLE_CSE_ID`: [Google Custom Search Engine](https://programmablesearchengine.google.com/)에서 생성  

---

## 현재 문제점
```
https://github.com/leehhm/chatbot/blob/main/chatbot.png?raw=true
```
