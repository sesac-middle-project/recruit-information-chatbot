from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
import json

class ChatAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm_qa = ChatOpenAI(model='gpt-4o', temperature=0.7)
        self.system_messages = """
            너는 사용자가 입력한 질문에 따라 직무, 경력, 선호 지역에 맞는 채용 공고를 출력하는 AI야. 아래 지침에 따라 작업을 수행해:

            1. question에서 직무, 경력 수준(예: 신입, 경력), 선호 지역을 추출해.
            - 경력 무관이면 신입에 포함해.
            2. context에 포함된 채용 공고들 중 question의 요구사항에 맞는 공고를 검색해.
            - question에 포함된 직무, 경력, 지역과 context의 직무, 경력, 지역이 무조건 동일한 것만 선택해야 해.
            - 경력 여부가 경력 무관이거나 신입과 경력이 모두 포함되었다면 신입일 때도 경력일 때도 모두 포함해.
            3. 중복 공고 출력 방지
            - 한 번 출력할 때 같은 공고를 여러 번 출력하지마.
            - context의 모든 공고들 중 memory에 이미 저장되어 있는 공고라면 이전에 이미 출력되었던 중복 공고이므로 해당 공고를 출력하지마.
            4. 가져오는 행의 개수는 10개 이하로 가져와.
            - 10개씩만 가져오지 말고, 가지고 있는 공고의 개수가 10개보다 적다고 판단되면 그 개수만 가져와.
            - memory에 저장되어 있는 공고는 그냥 다시 출력을 하지마. 없으면 그냥 5개, 6개, 7개씩 가져와도 돼.
            5. 출력 형식은 Streamlit의 `st.markdown()` 함수와 호환되는 Markdown 표 형식이어야 하며, 다음과 같은 형식을 사용해:
            - 중복 공고이거나 공고가 더 이상 없다면 그냥 남은 공고들만 출력해.

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
        self.coordination = """
            너는 직무에 대한 설명을 친절하게 해주는 잡 코디야.
            사용자가 어떠한 직무에 대해 설명해달라고 입력하면,
            해당 직무의 이름, 주요 업무 및 책임, 필요한 역량(학력, 경험, 기술 등)을 사용자가 알아보기 깔끔하게 설명해줘!
            만약 사용자가 특정 내용에 대해 질문을 한다면 네가 알고 있는만큼 설명해주면 돼.
            절대 없는 내용을 창조하면 안돼. 모르는 내용일 경우 '알 수 없는 정보입니다.' 라고 출력하면 돼.

            만약 사용자가 직무를 제시하면서 이전에 출력됐던 공고를 모두 출력해달라고 하면 memory에 저장된 해당 직무의 공고들을 전부 밑의 형식으로 출력해줘.
            만약 사용자가 이전에 물어봤던 모든 공고를 출력해달라고 한다면 직무 이름을 꼭 표에 넣고 같이 출력해줘야 돼.
            만약 중복된 공고가 있다면 하나만 남기고 나머지는 전부 빼고 출력해줘.
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

    def create_chain(self, query, retriever=None):
        """Prompt 템플릿 및 chain 생성"""
        try:
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
        except FileNotFoundError:
            chat_history = []
        results = retriever.get_relevant_documents(query)
        context = "\n".join([result.page_content for result in results])

        prompt = ChatPromptTemplate.from_messages([
            ('system', self.system_messages),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{question}'),
        ])

        chain = (
            RunnablePassthrough.assign(context=RunnableLambda(lambda _: context))
            | RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: chat_history))
            | prompt
            | self.llm
        )
        return chain

    def create_chat_chain(self):
        """Chat Prompt 템플릿 및 chain 생성"""
        try:
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
        except FileNotFoundError:
            chat_history = []
        prompt_qa = ChatPromptTemplate.from_messages([
                ('system', self.coordination),
                MessagesPlaceholder(variable_name='chat_history'),
                ('human', '{question}'),
        ])

        chain = (
            RunnablePassthrough.assign(chat_history=RunnableLambda(lambda _: chat_history))
            | prompt_qa
            | self.llm_qa
        )
        return chain


    def generate_response(self, question, chain):
        """질문을 바탕으로 응답 생성"""
        response = chain.invoke({'question': question})
        response_text = response.content if hasattr(response, 'content') else str(response)
        self.memory.save_context({'input': question}, {'output': response_text})
        return response_text