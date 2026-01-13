import streamlit as st

from lllm_copy import get_ai_response

st.set_page_config(page_title="CCS 챗봇", page_icon="🤖")

st.title("🤖 CCS 챗봇")
st.caption("CCS에 관련된 모든것을 답해드립니다!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print(f"before == {st.session_state.message_list}")
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="CCS 평가법에 대해 궁금한점을 말해주세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question})
    
    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            st.write(ai_response)
            st.session_state.message_list.append({"role":"ai", "content":ai_response})
    
print(f"after === {st.session_state.message_list}")



