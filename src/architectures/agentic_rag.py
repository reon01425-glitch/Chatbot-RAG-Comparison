import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_PATH = "./indo_finetuned_embedding"

# Set up global references for tools to access
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@tool
def cari_dokumen_sop(query: str) -> str:
    """Cari informasi relevan seputar SOP Fakultas Sains dan Matematika Universitas Diponegoro dari basis data dokumen."""
    docs = db.similarity_search(query, k=3)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

class AgenticRAG:
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")
        self.tools = [cari_dokumen_sop]
        
        # System prompt for ReAct or OpenAI Tools agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Anda adalah asisten layanan mahasiswa Fakultas Sains dan Matematika Universitas Diponegoro. "
                       "Bantu menjawab pertanyaan seputar SOP resmi kampus berdasarkan informasi yang diperoleh melalui alat/tools yang tersedia. "
                       "Jawablah dengan bahasa Indonesia yang jelas, singkat, dan sopan. "
                       "Jika informasi tidak ditemukan dari alat bantu, katakan secara jujur bahwa informasi tersebut tidak tersedia."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)

    def query(self, query_text: str) -> dict:
        try:
            # We also run a quick similarity search offline just to collect the retrieved sources/contexts for Ragas/eval evaluation
            docs = db.similarity_search(query_text, k=3)
            sources = [doc.metadata.get("id", "Unknown") for doc in docs]
            contexts = [doc.page_content for doc in docs]

            response = self.agent_executor.invoke({"input": query_text})
            answer = response["output"]
            
            return {
                "answer": answer,
                "sources": sources,
                "contexts": contexts
            }
        except Exception as e:
            return {
                "answer": f"Terjadi kesalahan saat memproses jawaban: {str(e)}",
                "sources": [],
                "contexts": []
            }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    rag = AgenticRAG()
    res = rag.query("Bagaimana cara melakukan pengajuan cuti akademik?")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
