import streamlit as st
import random
import re
import time
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, ConversationChain

# 환경변수 로드
load_dotenv()

# 파일 읽기
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

# 챗봇 메모리 초기화
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# 출력된 공고 목록을 추적할 변수
output_history = []
user_input_data = {}  # 사용자 입력 저장

# 채팅 기록을 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 사용자 입력 받기
def get_user_input():
    job = st.session_state.job
    exp = st.session_state.exp
    loc = st.session_state.loc
    return job, exp, loc

# 공고에서 회사 이름, 직무, URL 추출하는 함수
def extract_company_and_title(page_content):
    company_pattern = r"회사이름: ([^,]+)"
    job_title_pattern = r"공고이름: ([^,]+)"
    url_pattern = r"url: ([^,]+)"
    
    company_name = re.search(company_pattern, page_content)
    job_title = re.search(job_title_pattern, page_content)
    url = re.search(url_pattern, page_content)
    
    return company_name.group(1) if company_name else None, job_title.group(1) if job_title else None, url.group(1) if url else None

# 직무에 맞는 공고 우선 출력하는 함수
def prioritize_job_title(results, job):
    job_related = []
    non_job_related = []
    
    for result in results:
        company_name, job_title, url = extract_company_and_title(result.page_content)
        
        if job_title and job.lower() in job_title.lower():
            job_related.append(result)
        else:
            non_job_related.append(result)
    
    return job_related + random.sample(non_job_related, min(5, len(non_job_related)))

# 챗봇 함수
def chatbot():
    global user_input_data  # 사용자 입력 데이터 저장
    if not user_input_data or st.button("새로운 입력을 원하시면 클릭하세요"):
        # 사용자로부터 직무, 경력, 지역 입력 받기
        job = st.session_state.job
        exp = st.session_state.exp
        loc = st.session_state.loc
        user_input_data = {"job": job, "exp": exp, "loc": loc}
    else:
        # 이미 입력된 데이터 사용
        job = user_input_data["job"]
        exp = user_input_data["exp"]
        loc = user_input_data["loc"]
    
    question = f'{loc}에서 {exp}을 채용하는 {job} 공고 알려줘'
    
    # 검색된 공고 가져오기
    results = retriever.get_relevant_documents(question)
    
    # 직무가 제목에 포함된 공고를 우선적으로 가져오기
    results = prioritize_job_title(results, job)
    
    global output_history
    if output_history:
        # 이전에 출력된 결과는 제외하고 새로운 공고만 출력
        results = [result for result in results if result not in output_history]
    
    output = results[:5]  # 새로운 공고 중에서 5개 출력
    output_history.extend(output)  # 출력된 공고를 기록
    
    # 결과 출력
    for result in output:
        company_name, job_title, url = extract_company_and_title(result.page_content)
        st.write(f"회사이름: {company_name}, 공고이름: {job_title}, URL: {url}")
    
    # 더보기 여부 처리
    more_query = st.button("더 보기를 원하시면 클릭하세요")
    if more_query:
        chatbot()  # 더보기 버튼 클릭 시 계속해서 결과 출력

# 페이지 레이아웃 설정
st.title("채용 공고 탐색 챗봇")

# 왼쪽 영역: 직무, 경력, 지역 설정
col1, col2 = st.columns([2, 3])

with col1:
    exp_options = ['신입', '경력']
    loc_options = ['서울', '경기', '부산', '대구', '인천', '대전']
    job_options = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']

    st.selectbox('경력', exp_options, key='exp')
    st.selectbox('지역', loc_options, key='loc')

# 직무 버튼
with col2:
    for job in job_options:
        if st.button(job):
            st.session_state.job = job
            # 채팅 기록 초기화 및 인사말 추가
            st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요! 경력과 선호 지역, 검색을 원하는 직무를 선택해주세요."}]
            st.write(f"질문: {st.session_state.chat_history[0]['content']}")
            chatbot()

# 채팅 기록 출력 및 업데이트
if st.button("채팅 시작"):
    # 예시: 사용자가 버튼을 눌렀을 때 채팅 기록 추가
    st.session_state.chat_history.append({"role": "assistant", "content": "안녕하세요! 경력과 선호 지역, 검색을 원하는 직무를 선택해주세요."})

# 채팅 창
st.text_area("채팅", height=200, value="\n".join([msg['content'] for msg in st.session_state.chat_history]), key="chat_display")