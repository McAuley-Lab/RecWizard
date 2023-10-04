import sys
import streamlit as st
import requests
import json
from collections import defaultdict
from utils import strip_debug, create_graph, plotly_plot, plotly_timeline, img_to_html


try: 
    PORT = int(sys.argv[1])
except:
    PORT = 8000

@st.cache_data
def convert_to_json(chat_history):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return json.dumps(chat_history)


def chat_actions(model_name): 
    st.session_state["chat_history"][model_name].append(
        {"role": "user", "content": st.session_state["chat_input"]},
    )

def dict_validator():
    for x in ["gen_args", "rec_args"]:
        x = st.session_state[x]
        try:
            json.loads(x)
        except:
            st.sidebar.error(f"{x} is not a valid dict. Please check.")

st.markdown(img_to_html('recwizard.png', 'RecWizard'), unsafe_allow_html=True)
st.write("#### A Plug-n-Play Toolkit for Conversational Recommendation")
model_name = st.selectbox(label="Select Model", options=requests.get(f"http://localhost:{PORT}/listmodels").json())
st.subheader(model_name, divider="rainbow")
with st.spinner("Loading model..."):
    model_loading_status = requests.get(f"http://localhost:{PORT}/loadmodel", {"model_name": model_name}).json()
    print(model_loading_status, model_name)

mode = st.sidebar.radio(label="Logging mode", options=["info", "debug"])
plot_type = None
show_details = None
PLOT_OPTIONS = ["timeline", "graph"]

if mode == "debug":
    plot_type = st.sidebar.radio(label="Plot Type", options=PLOT_OPTIONS)
    show_details = st.sidebar.checkbox("Show Details")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = defaultdict(list)

if model_loading_status == "failure":
    st.exception(RuntimeError('Failed to load model. Something went wrong.'))
else:
    gen_args = st.sidebar.text_input("gen_args (dict)", value="{}", on_change=dict_validator, key="gen_args")
    rec_args = st.sidebar.text_input("rec_args (dict)", value="{}", on_change=dict_validator, key="rec_args")
    if "chatgpt" in model_name.lower():
        chatgpt_model = st.sidebar.radio(label="ChatGPT model", options=["gpt-4", "gpt-3.5"])
    prompt = st.chat_input("Enter your message",
                           on_submit=chat_actions,
                           args=(model_name,),
                           key="chat_input",
                          )
    for i in st.session_state["chat_history"][model_name]:
        with st.chat_message(name=i["role"], avatar='recwizard.png' if i["role"]=="assistant" else None):
            if 'content' in i:
                st.write(i["content"])
            if 'debug_message' in i:
                if i["type"] == "json":
                    st.json(i["debug_message"], expanded=False)
                else:
                    st.write(i["debug_message"])

    if prompt:
        prompt = "User: " + prompt
        history = st.session_state["chat_history"][model_name]
        if len(history) > 2:
            past = []
            for chat in history:
                if "content" in chat:
                    if chat["role"] == "user":
                        past.append("User: " + chat["content"])
                    else:
                        past.append("System: " + chat["content"])
            past = "<sep>".join(past)
            prompt = past + "<sep>" + prompt

        with st.spinner("generating..."):
            gen_args = json.loads(gen_args)
            rec_args = json.loads(rec_args)
            if "chatgpt" in model_name.lower():
                gen_args["model_name"] = chatgpt_model

            result = requests.get(f"http://localhost:{PORT}/predict", {"model_name": model_name,
                                                                    "query": prompt,
                                                                    "mode": mode,
                                                                    "gen_args": gen_args,
                                                                    "rec_args": rec_args,
                                                                    }).json()
        if result:
            with st.chat_message(name="ai", avatar="recwizard.png"):
                output = result["response"]["output"]
                if "links" in result["response"]:
                    for name, link in result["response"]["links"].items():
                        if link:
                            output = output.replace(name, f"[{name}]({link})")
                print(output)
                formatted_resp = output.replace("System: ", "").replace("User: ", "").replace("<|endoftext|>", "")
                st.write(formatted_resp)
                st.session_state["chat_history"][model_name].append(
                    {
                        "role": "assistant",
                        "content": formatted_resp,
                    },
                )
                if result["graph"]:
                    # Create a directed graph using NetworkX
                    st.toast(f'## Total time: {round(result["graph"][-1]["time"], 3)}s')
                
                    if plot_type == "graph":
                        G = create_graph(result)
                        fig = plotly_plot(G)
                        st.plotly_chart(fig, use_container_width=False)

                    elif plot_type == "timeline":
                        fig = plotly_timeline(result["graph"])
                        st.plotly_chart(fig, use_container_width=False)

                    st.session_state["chat_history"][model_name].append(
                        {
                            "role": "assistant",
                            "debug_message": fig,
                            "type": "figure"
                        },
                    )
                    if show_details:
                        st.write("details:")
                        st.json(result["graph"], expanded=False)
                        st.session_state["chat_history"][model_name].append(
                            {
                                "role": "assistant",
                                "debug_message": result["graph"],
                                "type": "json"
                            },
                        )
    if st.sidebar.button("Refresh Chat"):
        del st.session_state["chat_history"][model_name]
        st.rerun()
    st.sidebar.download_button("Download Chat History",
                           convert_to_json(strip_debug(st.session_state["chat_history"], model_name)),
                           file_name="chat_history.json")
