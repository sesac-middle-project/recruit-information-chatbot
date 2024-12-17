import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import json

class StreamlitUI:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Job Search Chatbot 💭")
        st.caption("🚀 A Streamlit chatbot powered by OpenAI")
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "원하시는 직무, 신입/경력, 지역을 선택해주세요\n\n혹은 직무에 대한 질문을 채팅에 입력해주세요"}]
            self.user_input = None
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        if 'input_count' not in st.session_state:
            st.session_state.input_count = 0

    def sidebar_options(self):
        """직무, 경력, 지역 선택 사이드바"""
        with st.sidebar:
            job_options = ['데이터 분석가', '데이터 엔지니어', 'AI 개발자', '챗봇 개발자', 
                        '클라우드 엔지니어', 'API 개발자', '머신러닝 엔지니어', '데이터 사이언티스트']
            selected_job = st.selectbox("직무를 선택하세요:", job_options, key="selected_job")
            selected_exp = st.selectbox("경력 수준을 선택하세요:", ["신입", "경력"], key="selected_exp")
            loc_options = ['서울', '경기', '인천', '대전', '광주', '대구', '울산', '부산', '강원', 
                            '세종', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '해외']
            selected_loc = st.multiselect("지역을 입력하세요:", loc_options, key="selected_loc")
            button = st.button("🔍 조회")

        return selected_job, selected_exp, selected_loc, button
    
    def user_input_box(self):
        """사용자 입력창"""
        # input_count를 기준으로 고유한 key 생성
        input_key = f"chat_input_{st.session_state.input_count}"
        
        # 채팅 입력 박스
        user_input = st.chat_input("직무 관련 질문을 입력해주세요: ", key=input_key)
        
        # 입력이 있을 때마다 카운트 증가
        if user_input:
            st.session_state.input_count += 1
        
        return user_input
    
    def save_memory(self, question, response):
        st.session_state.memory.save_context(
            {'input': question},
            {'output': response},
        )
        memory = st.session_state.memory.load_memory_variables({})['chat_history']
        chat_history = []
        for message in memory:
            # message 객체가 'HumanMessage'인지 'AIMessage'인지 확인하여 'role' 값 지정
            if isinstance(message, HumanMessage):
                role = 'user'  # 사용자의 메시지일 경우
            elif isinstance(message, AIMessage):
                role = 'ai'  # AI의 메시지일 경우
            else:
                role = 'unknown'  # 알 수 없는 경우
            
            chat_history.append({
                'role': role,
                'content': message.content
            })
        
        # JSON 파일로 저장
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)

    def save_messages(self, question, response):
        st.session_state["messages"].append({
            "role": "user", "content": question
        })
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response
        })

    def display_response(self):
        """응답 출력"""
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])