import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("gpt", pad_token_id=tokenizer.eos_token_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("assistant"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    encoded_input = tokenizer(prompt, return_tensors='pt')
    response = model.generate(**encoded_input,
                              max_new_tokens=50,
                              do_sample=True,
                              top_k=50
                              # temperature=0.6
                              )

    # st.text(response)
    # msg = tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    msg = tokenizer.decode(response[0], skip_special_tokens=True)
    # logits = response.logits[0, -1, :]
    # softmax = tf.math.softmax(logits, axis=-1)
    # argmax = tf.math.argmax(softmax, axis=-1)
    # st.text("", "[", tokenizer.decode(argmax), "]")
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
