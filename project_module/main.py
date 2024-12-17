from data_processor import DataProcessor
from chat_assistant import ChatAssistant
from streamlit_ui import StreamlitUI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def main():
    processor = DataProcessor()
    assistant = ChatAssistant()
    ui = StreamlitUI()

    filenames = ['datexts.txt', 'detexts.txt', 'aitexts.txt', 'cbtexts.txt',
                 'cetexts.txt', 'apitexts.txt', 'metexts.txt', 'dstexts.txt']
    metadatas = [[{'role': '데이터 분석가'}], [{'role': '데이터 엔지니어'}], [{'role': 'AI 개발자'}], [{'role': '챗봇 개발자'}],
                [{'role': '클라우드 엔지니어'}], [{'role': 'API 개발자'}], [{'role': '머신러닝 엔지니어'}], [{'role': '데이터 사이언티스트'}]]
    files = processor.load_and_split_files(filenames)
    processor.build_vectorstore(files, metadatas)
    retriever = processor.get_retriever()

    selected_job, selected_exp, selected_loc, button = ui.sidebar_options()
    user_input = ui.user_input_box()

    if button:
        if selected_loc:
            loc_text = ", ".join(selected_loc)
        else:
            loc_text = "모든 지역"
        query = f"{loc_text}에서 {selected_exp}인 {selected_job} 직무를 채용하는 공고 알려줘"
        chain = assistant.create_chain(query, retriever=retriever)
        response = assistant.generate_response(query, chain)
        ui.save_memory(query, response)
        ui.save_messages(query, response)

    if user_input:
        chain = assistant.create_chat_chain()
        response = assistant.generate_response(user_input, chain)
        ui.save_memory(user_input, response)
        ui.save_messages(user_input, response)
    
    ui.display_response()

if __name__ == "__main__":
    main()
