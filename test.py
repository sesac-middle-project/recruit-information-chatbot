import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 확인
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.info("OpenAI API Key가 필요합니다. 환경 변수를 통해 설정하세요.")
    st.stop()

# 초기 설정
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# 사이드바: 직무, 경력, 지역 선택 및 조회 버튼 추가
with st.sidebar:
    st.title("💬 Job Search Chatbot")
    st.header("검색 조건")
    
    job_options = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자',
                   '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']
    selected_jobs = st.multiselect("직무 선택 (복수 선택사항):", job_options)
    
    exp_options = ['신입', '경력']
    selected_exp = st.selectbox("경력 선택 (단일 선택):", exp_options)
    
    loc_options = ['서울', '경기', '인천', '대전', '광주', '대구', '울산', '부산', '강원',
                   '세종', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '해외']
    selected_locs = st.selectbox("지역 선택 :", loc_options)
    
    search_button = st.button("🔍 조회")

# 캐시된 리소스를 위한 함수
@st.cache_resource
def initialize_vector_db(file_content):
    # 텍스트 전처리
    chfile = file_content.replace('\n', '\n\n')
    
    # 텍스트 분할기 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=400)
    lines = splitter.split_text(chfile)
    
    # OpenAI Embeddings 초기화
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    
    # FAISS 벡터 DB 생성
    db = FAISS.from_texts(texts=lines, embedding=embeddings)
    return db.as_retriever(search_kwargs={'k': 5})

# 채용 데이터 파일 읽기
try:
    with open('jobs.txt', encoding='utf-8') as f:
        file = f.read()
except FileNotFoundError:
    st.error("⚠️ 'jobs.txt' 파일을 찾을 수 없습니다. 데이터를 업로드하세요.")
    st.stop()

# 세션 상태에 retriever가 저장되지 않은 경우 초기화
if 'retriever' not in st.session_state:
    try:
        st.session_state.retriever = initialize_vector_db(file)
    except Exception as e:
        st.error(f"⚠️ 벡터 데이터베이스 초기화 중 오류가 발생했습니다: {e}")
        st.stop()

# RAG 파이프라인 구성 
if 'retriever' in st.session_state and search_button:
    retriever = st.session_state.retriever
    
    # 대화 메모리 설정
    memory = ConversationBufferMemory()
    
    # LLM 설정 (OpenAI GPT-4)
    llm = OpenAI(temperature=0.1, model_name="gpt-4")
    
    # 프롬프트 템플릿 설정
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Given the following context, answer the question: {context} Question: {question}"
    )
    
    # 선택된 직무 각각에 대해 검색 수행
    if selected_jobs:
        for job in selected_jobs:
            st.write(f"### {job} 관련 공고")
            # 선택된 경력과 지역 쿼리 생성
            exp_query = f"경력: {selected_exp}"
            loc_query = f"지역: {selected_locs}"
            query = f"직무: {job} {exp_query} {loc_query}"
            
            # 검색
            search_results = retriever.get_relevant_documents(query)
            
            # 상위 5개 결과 출력
            if search_results:
                for idx, result in enumerate(search_results[:5], start=1):
                    st.markdown(f"**{idx}.** {result.page_content}")
            else:
                st.write("⚠️ 검색 결과가 없습니다.")
    else:
        st.write("⚠️ 검색할 직무를 선택해주세요.")
