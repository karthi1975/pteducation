import streamlit as st
import boto3
import json
import os

# Get AWS credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# AWS Client Configuration
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Define model ID for Cohere Command R+
model_id = "cohere.command-r-plus-v1:0"
content_type = "application/json"
accept = "*/*"

def get_bedrock_response(chat_history, message):
    # Add context prompt and ask for bullet points
    context_prompt = "Provide a brief, bullet-pointed answer tailored for a TBI or spinal cord injury patient: \n"
    request_body = {
        "chat_history": chat_history,
        "message": context_prompt + message
    }
    body_json = json.dumps(request_body)

    response = client.invoke_model(
        modelId=model_id,
        contentType=content_type,
        accept=accept,
        body=body_json
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        result = response.get('body').read()
        return json.loads(result)['text']
    else:
        return f"Error: {response['ResponseMetadata']}"

# Streamlit UI setup
# Display logos side by side
col1, col2 = st.columns(2)
with col1:
    st.image("logo.svg", width=150)
with col2:
    st.image("logo-neilsen.svg", width=150)

st.title("Education Assistant")
st.write("This chatbot provides education and support for TBI and spinal cord injury patients.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Using form to allow 'Enter' key submission
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask your question here...", key='user_input')
    submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        st.session_state.chat_history.append({"role": "USER", "message": user_input})
        response = get_bedrock_response(st.session_state.chat_history, user_input)
        st.session_state.chat_history.append({"role": "CHATBOT", "message": response})

        # Display the latest chat immediately below the input form
        st.write(f"USER: {user_input}")
        st.write(f"CHATBOT: {response}")

# Display previous chat history in order
st.write("### Chat History")
for chat in st.session_state.chat_history[:-2]:
    st.write(f"{chat['role']}: {chat['message']}")

# CSS styling for send button alignment
st.markdown(
    """
    <style>
    .stTextInput > div > div {
        display: flex;
        align-items: center;
    }
    .stTextInput > div > div > input {
        width: 80%;
    }
    .stTextInput > div > div > button {
        width: 15%;
        margin-left: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)