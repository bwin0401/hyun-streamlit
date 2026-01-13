from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic  import hub
from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains  import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from sentence_transformers import SentenceTransformer


from config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    # 설정
    # save_dir = r"C:\Users\7255693\Desktop\streamlit\multi_base_faiss_index"
    save_dir = './save/multi_base_faiss_index'
    # model_path = r"C:\Users\7255693\Desktop\streamlit\multi_base"
    # model = "intfloat/multilingual-e5-base"
    model = "./multi_base"
    
    # 임베딩 모델 (공통)
    embeddings = HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 인덱스가 이미 존재하는지 확인
    if os.path.exists(save_dir):
        # ✅ 기존 인덱스 로드
        print("기존 FAISS 인덱스를 로드합니다...")
        faiss_loaded = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)

    else:
        # ✅ 최초 실행: 문서 로드 + 인덱스 생성 + 저장
        print("FAISS 인덱스를 새로 생성합니다...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        # loader = Docx2txtLoader(r"C:\Users\7255693\Desktop\streamlit\tax_with_markdown.docx")
        loader = Docx2txtLoader('./docs/tax_with_markdown.docx')
        document_list = loader.load_and_split(text_splitter=text_splitter)

        faiss_db = FAISS.from_documents(documents=document_list, embedding=embeddings)
        faiss_db.save_local(save_dir)
        faiss_loaded = faiss_db

    # ✅ 검색기 준비
    retriever = faiss_loaded.as_retriever(search_kwargs={"k": 3})
    # retriever = faiss_loaded.as_retriever(search_kwargs={"k": 5})  

    return retriever
    
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
   
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
   
   
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    return history_aware_retriever
    
def get_llm():
    os.environ["AZURE_OPENAI_API_KEY"] = "8c3d7622125953975b740c0d3aee4f0401df32da42a985de1cf2650cbcbc7f0a"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://h-chat-api.autoever.com/v2/api"
    
    AZURE_OPENAI_API_VERSION = "2024-10-21"
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
    
        # LLM 초기화
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key = os.environ["AZURE_OPENAI_API_KEY"],
        api_version=AZURE_OPENAI_API_VERSION,
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.7,
    )
    return llm



def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain



def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain

   
   
   
   
    # # prompt = hub.pull("rlm/rag-prompt")
    # # qa_chain = RetrievalQA.from_chain_type(
    # #     llm, 
    # #     retriever=retriever,
    # #     chain_type_kwargs={"prompt": prompt}
    # # )
    # return qa_chain



def get_ai_response(user_message):

    os.environ["AZURE_OPENAI_API_KEY"] = "8c3d7622125953975b740c0d3aee4f0401df32da42a985de1cf2650cbcbc7f0a"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://h-chat-api.autoever.com/v2/api"
    
    AZURE_OPENAI_API_VERSION = "2024-10-21"
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
    

    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    tax_chain = {"input": dictionary_chain} | rag_chain
    ai_response = tax_chain.invoke(
    {
        "question": user_message
    },
    config={
        "configurable": {"session_id": "abc123"}
    },
)


    return ai_response








