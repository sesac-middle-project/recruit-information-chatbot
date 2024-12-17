from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

class DataProcessor:
    def __init__(self, embedding_model='text-embedding-3-large', chunk_size=7500):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = None
    
    def load_and_split_files(self, filenames):
        """파일을 불러오고 텍스트를 분할합니다."""
        files = []
        for file in filenames:
            with open(file, encoding='utf-8') as f:
                content = f.read().replace('\n', '\n\n')
                files.append(content)
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size)
        return [splitter.split_text(file) for file in files]

    def build_vectorstore(self, files, metadatas):
        """FAISS 벡터 데이터베이스를 생성합니다."""
        alldb = FAISS.from_texts(texts=files[0], embedding=self.embeddings, metadatas=metadatas[0] * len(files[0]))
        for i in range(1, len(files)):
            db = FAISS.from_texts(texts=files[i], embedding=self.embeddings, metadatas=metadatas[i] * len(files[i]))
            alldb.merge_from(db)
        self.vectorstore = alldb

    def get_retriever(self, k=15):
        """벡터 DB에서 검색기 설정"""
        return self.vectorstore.as_retriever(search_kwargs={'k': k})
