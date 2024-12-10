from openai import OpenAI
import streamlit as st
import random
import re
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()

# 채용 데이터 파일 읽기
with open('jobs.txt', encoding='utf-8') as f:
    file = f.read()

# 텍스트 전처리
chfile = file.replace('\n', '\n\n')

# 텍스트 분할
splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400)
lines = splitter.split_text(chfile)

# OpenAI Embedding 모델 초기화
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# FAISS 벡터 DB 생성
db = FAISS.from_texts(texts=lines, embedding=embeddings)

# 검색기 설정
retriever = db.as_retriever(search_kwargs={'k': 100})

# 채용 공고에서 회사 이름, 직무, URL 추출하는 함수
def extract_company_and_title(page_content):
    company_pattern = r"회사이름: ([^,]+)"
    job_title_pattern = r"공고이름: ([^,]+)"
    url_pattern = r"url: ([^,]+)"
    
    company_name = re.search(company_pattern, page_content)
    job_title = re.search(job_title_pattern, page_content)
    url = re.search(url_pattern, page_content)
    
    return company_name.group(1) if company_name else None, job_title.group(1) if job_title else None, url.group(1) if url else None

# 채용 공고 검색 및 채팅창에 반영하는 함수
def search_jobs_and_update_chat(selected_job, selected_exp, selected_loc, start_index=0, limit=5):
    query = f"{selected_loc}에서 {selected_exp}을 채용하는 {selected_job} 공고 알려줘"
    results = retriever.get_relevant_documents(query)
    
    # 중복되지 않는 결과 필터링
    displayed_results = st.session_state.get("displayed_results", [])
    new_results = [result for result in results if result not in displayed_results]
    
    # 새롭게 출력할 공고
    output = new_results[start_index:start_index + limit]
    
    if output:
        # 결과를 메시지로 변환
        job_messages = []
        for result in output:
            company_name, job_title, url = extract_company_and_title(result.page_content)
            job_messages.append(f"- 회사이름: {company_name}, 공고이름: {job_title}, [URL]({url})")
        
        job_message_content = "\n".join(job_messages)
        
        # 출력된 결과 기록
        st.session_state["displayed_results"].extend(output)
    else:
        job_message_content = "더 이상 표시할 공고가 없습니다."
    
    # 채팅창에 공고 반영
    st.session_state["messages"].append({
        "role": "assistant",
        "content": f"다음은 선택하신 조건에 맞는 공고입니다:\n\n{job_message_content}"
    })

# 사이드바: 직무, 경력, 지역 선택
with st.sidebar:
    # 직무 선택
    job_options = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', 
                   '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']
    selected_job = st.selectbox("직무를 선택하세요:", job_options, key="selected_job")

    # 경력 선택
    exp_options = ['신입', '경력']
    selected_exp = st.selectbox("신입 또는 경력을 선택하세요:", exp_options, key="selected_exp")

    # 지역 선택
    loc_options = ['서울', '경기', '인천', '대전', '광주', '대구', '울산', '부산', '강원', 
                   '세종', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '해외']
    selected_loc = st.selectbox("지역을 선택하세요:", loc_options, key="selected_loc")

    # 채용 공고 검색 버튼
    if st.button("채용 공고 검색"):
        # 선택된 값 채팅창에 반영
        st.session_state["messages"].append({
            "role": "user",
            "content": f"제가 선택한 직무는 {selected_job}, 경력은 {selected_exp}, 지역은 {selected_loc}입니다."
        })
        # 공고 표시 초기화
        st.session_state["displayed_results"] = []
        search_jobs_and_update_chat(selected_job, selected_exp, selected_loc)

# 앱 제목 및 설명
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "원하시는 직무, 신입/경력, 지역을 선택해주세요"}]

# 기존 채팅 메시지 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 더보기 버튼
if "displayed_results" in st.session_state and len(st.session_state["displayed_results"]) > 0:
    if st.button("더 보기"):
        search_jobs_and_update_chat(
            selected_job,
            selected_exp,
            selected_loc,
            start_index=len(st.session_state["displayed_results"])
        )

# 사용자 입력 처리
if prompt := st.chat_input():
    # OpenAI API 키가 없으면 메시지 표시
    if "openai_api_key" not in st.secrets:
        st.info("OpenAI API Key가 필요합니다. 환경 변수를 통해 설정하세요.")
        st.stop()

    client = OpenAI(api_key=st.secrets["openai_api_key"])
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # OpenAI API를 통해 응답 생성
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    # 응답 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
