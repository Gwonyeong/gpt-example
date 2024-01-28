import streamlit as st
import time

st.set_page_config(
    page_title='FullstackGPT',
    page_icon="💻"
)

st.title('FullstackGPT')

if "messages" not in st.session_state:
    st.session_state['messages'] = []



# st.write(st.session_state['messages'])

# st.write는 메세지를 출력해준다.
# with st.chat_message('human'):
#     st.write('Hello, I am a human 🧑‍💻')

# with st.chat_message('ai'):
#     st.write('Hello, I am an AI 🤖')

# 스피너를 만들 수 있음
# with st.status("Embedding file", expanded=True) as status:
#     time.sleep(3)
#     st.write('Getting the file')
#     time.sleep(3)
#     st.write('Embedding the file')
#     time.sleep(3)
#     st.write('Done')
#     status.update(label="Error", state='error')

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)

    if save:
        st.session_state['messages'].append({
            'message': message,
            'role': role
        })

for message in st.session_state['messages']:
    send_message(message['message'], message['role'], False)

message =st.chat_input('Send a message to the AI')

if(message):
    send_message(message, 'human')
    time.sleep(2)
    send_message(f'You said: {message}', 'ai')