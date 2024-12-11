from openai import OpenAI
import streamlit as st
import re
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

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

# 시스템 메시지 및 Prompt 템플릿
system_message = """
    너는 입력된 question 속에 포함된 직무, 경력, 선호 지역에 맞는 채용 공고를 다섯 개씩 출력하는 탐색 AI야.
    검색된 context들 중 입력된 직무와 같은 카테고리인 채용 공고를 찾아오면 돼.
    공고이름에 직무가 들어가는 공고를 우선으로 출력하고, 그 이외에는 랜덤으로 출력해줘.
    만약 사용자가 더 많은 공고를 필요로 한다면, 같은 카테고리에서 이전에 네가 가져온 공고들을 제외하고 나머지 공고들 중 다섯 개를 뽑아서 가져와서 출력해줘.
    출력 형식은 하나의 공고마다 회사이름, 공고이름, URL를 출력해주면 돼.

    # context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{question}'),
])

# LLM 및 메모리 초기화
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# 공고 데이터 처리 함수
def extract_company_and_title(page_content):
    company_pattern = r"회사이름: ([^,]+)"
    job_title_pattern = r"공고이름: ([^,]+)"
    url_pattern = r"url: ([^,]+)"
    
    company_name = re.search(company_pattern, page_content)
    job_title = re.search(job_title_pattern, page_content)
    url = re.search(url_pattern, page_content)
    
    return company_name.group(1) if company_name else None, job_title.group(1) if job_title else None, url.group(1) if url else None

# 채용 공고 검색 및 출력 함수
def search_jobs_and_update_chat(question, start_index=0, limit=5):
    # FAISS 검색
    results = retriever.get_relevant_documents(question)

    # 중복된 결과 제거
    displayed_results = st.session_state.get("displayed_results", [])
    new_results = [result for result in results if result not in displayed_results]

    # 새롭게 출력할 공고
    output = new_results[start_index:start_index + limit]
    job_messages = []

    for result in output:
        company_name, job_title, url = extract_company_and_title(result.page_content)
        if company_name and job_title and url:
            job_messages.append(f"- 회사이름: {company_name}, 공고이름: {job_title}, [URL]({url})")
    
    if job_messages:
        st.session_state["displayed_results"].extend(output)
        return "\n".join(job_messages)
    else:
        return "더 이상 표시할 공고가 없습니다."

# 일반 질문 처리 함수
def process_user_question(question):
    context = "\n".join([result.page_content for result in retriever.get_relevant_documents(question)])
    
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda _: context))
        | RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: memory.chat_memory.messages))  # 수정된 부분
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"question": question})
    memory.save_context({"input": question}, {"output": response})
    return response

# Streamlit 앱 구성
st.title("💬 Chatbot - 채용 공고 탐색 AI")
st.caption("LLM 및 FAISS를 활용한 채용 공고 탐색")

# 초기 메시지 설정
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요! 궁금한 점을 입력해주세요."}]

if "displayed_results" not in st.session_state:
    st.session_state["displayed_results"] = []

# 사이드바: 조건 선택
with st.sidebar:
    job_options = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', 
                   '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']
    exp_options = ['신입', '경력']
    loc_options = ['서울', '경기', '인천', '대전', '광주', '대구', '울산', '부산', '강원', 
                   '세종', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '해외']

    selected_job = st.selectbox("직무를 선택하세요:", job_options, key="selected_job")
    selected_exp = st.selectbox("신입/경력을 선택하세요:", exp_options, key="selected_exp")
    selected_loc = st.selectbox("지역을 선택하세요:", loc_options, key="selected_loc")

    if st.button("채용 공고 검색"):
        st.session_state["messages"].append({
            "role": "user",
            "content": f"제가 선택한 직무는 {selected_job}, 경력은 {selected_exp}, 지역은 {selected_loc}입니다."
        })
        question = f"{selected_loc}에서 {selected_exp}을 채용하는 {selected_job} 공고 알려줘"
        st.session_state["question"] = question

        # 채용 공고 검색
        response = search_jobs_and_update_chat(question)
        st.session_state["messages"].append({"role": "assistant", "content": response})

# 기존 대화 기록 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# "더 보기" 버튼
if "question" in st.session_state and len(st.session_state["displayed_results"]) > 0:
    if st.button("더 보기"):
        question = st.session_state["question"]
        response = search_jobs_and_update_chat(
            question,
            start_index=len(st.session_state["displayed_results"])
        )
        st.session_state["messages"].append({"role": "assistant", "content": response})

# 사용자 질문 입력 처리
user_input = st.chat_input("질문을 입력하세요:")
if user_input :
    st.session_state["messages"].append({"role": "user", "content": user_input})
    if "더 보기" not in user_input:  # 일반 질문 처리
        response = process_user_question(user_input)
    else:  # "더 보기" 요청 무시
        response = "현재 더보기는 버튼으로만 지원됩니다."
    st.session_state["messages"].append({"role": "assistant", "content": response})
