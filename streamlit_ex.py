import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import re
from dotenv import load_dotenv

# from langchain_teddynote import logging
# logging.langsmith("GPT-4o")

# 환경 변수 로드
load_dotenv()

st.set_page_config(layout="wide")

@st.cache_resource
def preprocessing():
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
    retriever = db.as_retriever(search_kwargs={'k': 800})

    return retriever


# 메모리 초기화 (직접 chat_history 설정 제거)
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

retriever = preprocessing()

# 시스템 메시지 템플릿 설정
system_message = """
    너는 사용자가 입력한 질문에 따라 직무, 경력, 선호 지역에 맞는 채용 공고를 출력하는 AI야. 아래 지침에 따라 작업을 수행해:

    1. question에서 직무, 경력 수준(예: 신입, 경력), 선호 지역을 추출해.
    2. context에서 제공된 채용 공고들 중 question의 요구사항에 맞는 공고를 검색해.
    - question에 포함된 직무, 경력, 지역은 꼭 입력된 값과 동일한 것만 검색해야 돼.
    3. 우선순위 기준
    - 공고 제목에 직무 키워드가 포함된 경우를 우선적으로 출력해.
    - 경력 수준에 신입, 경력이 모두 포함된 경우, 신입일 때와 경력일 때 모두 출력해도 돼.
    4. 중복 방지
    - 앞의 답변을 저장해두고 다음에 조회할때 답변한 값들은 제외시켜줘 근데 저장한 답변을 출력하지마 이미 표로 출력중이야.
    5. 채용 공고를 가져온 사이트는 jobkorea, saramin, jumpit, wanted로 네 개가 있어
    - 예를 들어, 다섯 개의 채용 공고를 출력한다면 jobkorea 1개, saramin 2개, jumpit 1개, wanted 1개처럼 골고루 들어가면 좋아.
    6. 출력 형식은 Streamlit의 `st.markdown()` 함수와 호환되는 Markdown 표 형식이어야 하며, 다음과 같은 형식을 사용해:


    | idx | 회사이름 | 공고이름 | 지역 | 사이트 | URL |
    |-----|----------|----------|------|-------|-----|
    | 1   | 회사이름1 | 공고이름1 | 지역1 | 사이트1 | [URL](해당 url) |
    | 2   | 회사이름2 | 공고이름2 | 지역2 | 사이트2 | [URL](해당 url) |
    | ... | ...      | ...      | ...  | ... | ... |
    
    # memory: {chat_history}
    # context: {context}
    # question: {question}
    # answer:
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system_message),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{question}'),
])

coordination = """
    너는 직무에 대한 설명을 친절하게 해주는 잡 코디야.
    사용자가 어떠한 직무에 대해 설명해달라고 입력하면,
    해당 직무의 이름, 주요 업무 및 책임, 필요한 역량(학력, 경험, 기술 등)을 사용자가 알아보기 깔끔하게 설명해줘!
    만약 사용자가 특정 내용에 대해 질문을 한다면 네가 알고 있는만큼 설명해주면 돼.
    절대 없는 내용을 창조하면 안돼. 모르는 내용일 경우 '알 수 없는 정보입니다.' 라고 출력하면 돼.

    만약 사용자가 직무를 제시하면서 이전에 출력됐던 공고를 모두 출력해달라고 하면 memory에 저장된 해당 직무의 공고들을 전부 밑의 형식으로 출력해줘.
    만약 사용자가 이전에 물어봤던 모든 공고를 출력해달라고 한다면 직무 이름을 꼭 표에 넣고 같이 출력해줘야 돼.
    출력 형식은 표 형식으로 출력되며, 각 공고는 번호(idx), 회사이름, 공고이름, 직무, URL을 포함해야 해.
    URL은 Streamlit에서 지원 가능한 형태로 링크를 생성해야 하며, 다음과 같이 출력하면 돼:

    | idx | 회사이름 | 공고이름 | 지역 | 직무 | URL |
    |-----|----------|----------|------|-----|-----|
    | 1   | 회사이름1 | 공고이름1 | 지역1 | 직무1 | [URL](해당 url) |
    | 2   | 회사이름2 | 공고이름2 | 지역2 | 직무2 | [URL](해당 url) |
    | ... | ...      | ...      | ...  | ... | ... |

    Streamlit에서 이 표를 출력할 때는 `st.markdown()` 함수와 Markdown 테이블 형식을 활용하면 돼.
    사용자가 선택할 수 있는 직무는 데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', 
    '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트' 가 존재해.

    # memory: {chat_history}
    # question: {question}
    # answer: 
"""

prompt_qa = ChatPromptTemplate.from_messages([
    ('system', coordination),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{question}'),
])

# 검색 및 출력 함수
def search_jobs_with_llm(question):
    # 검색된 결과를 LLM에게 전달하기 위한 context 생성
    results = retriever.get_relevant_documents(question)
    context = "\n".join([result.page_content for result in results])
    llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
    # 메모리에서 대화 기록 로드
    memory_variables = st.session_state.memory.load_memory_variables({})
    chat_history = memory_variables.get('chat_history', [])

    # LLM에게 검색된 결과와 질문을 전달하여 공고 출력
    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(lambda _: context))
        | RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: chat_history))
        | prompt
        | llm
    )
    
    # 응답 생성
    response = chain.invoke({'question': question})
    
    # 응답을 문자열로 변환 (content 추출)
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)

    # 메모리에 대화 기록 저장
    st.session_state.memory.save_context(
        {'input': question},
        {'output': response_text},
    )

    return response_text

def process_user_question(question):
    llm = ChatOpenAI(model='gpt-4o', temperature=0.7)
    memory_variables = st.session_state.memory.load_memory_variables({})
    chat_history = memory_variables.get('chat_history', [])
    chain = (
        RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: chat_history))
        | prompt_qa
        | llm
    )

    response = chain.invoke({'question': question})

    # 응답을 문자열로 변환 (content 추출)
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)

    # 메모리에 대화 기록 저장
    st.session_state.memory.save_context(
        {'input': question},
        {'output': response_text},
    )

    return response_text

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "원하시는 직무, 신입/경력, 지역을 선택해주세요\n\n혹은 직무에 대한 질문을 채팅에 입력해주세요"}]

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
    selected_loc = st.multiselect("지역을 선택하세요:", loc_options, key="selected_loc")

    # 채용 공고 검색 버튼
    if st.button("🔍 조회"):
        if selected_loc:
            loc_text = ", ".join(selected_loc)
        else:
            loc_text = "모든 지역"
        query = f"{loc_text}에서 {selected_exp}인 {selected_job} 직무를 채용하는 공고 알려줘"
        st.session_state["messages"].append({
            "role": "user",
            "content": query
        })
        
        # LLM에 검색 요청
        response = search_jobs_with_llm(query)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response
        })

# 앱 제목 및 설명
st.title("Job Search Chatbot 💭")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# 기존 채팅 메시지 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 질문 입력 처리
user_input = st.chat_input("직무에 대한 질문을 입력하세요:")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    response = process_user_question(user_input)
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response
    })
    st.chat_message("assistant").write(response)