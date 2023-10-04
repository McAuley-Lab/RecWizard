import networkx as nx
import matplotlib.pyplot as plt
import copy
import base64
from pathlib import Path
import streamlit as st

def strip_debug(chat_history, model_name):
    chat_history = copy.deepcopy(chat_history[model_name])
    chat_history = [c for c in chat_history if "debug_message" not in c]
    return chat_history


def create_graph(response):
    G = nx.DiGraph()
    # Add nodes and edges to the graph
    for curr_step, next_step in zip(response["graph"], response["graph"][1:]):
        G.add_node(curr_step["node"], attr={"name": curr_step["node"],
                                            "time": round(curr_step["time"], 3)})
        G.add_node(next_step["node"], attr={"name": next_step["node"],
                                            "time": round(next_step["time"], 3)})
        G.add_edge(curr_step["node"], next_step["node"])
    return G


def matplotlib_plot(G):
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.kamada_kawai_layout(G)
    labels = nx.get_node_attributes(G, 'attr')
    nx.draw(G, pos, labels=labels, with_labels=False, node_size=400,
            node_color="skyblue", font_size=8, font_color="black",
            font_weight="bold", ax=ax, arrows=True)
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        ax.text(x, y + 0.1,
                s=f"{data['attr']['name']}: {round(data['attr']['time'], 3)}s",
                bbox=dict(facecolor='red', alpha=0.1),
                horizontalalignment='center')

    return fig


def plotly_timeline(graph):
    import pandas as pd
    import plotly.express as px
    from datetime import datetime

    nodes = graph
    nodes.sort(key=lambda x: x["start_time"])
    time0 = nodes[0]["start_time"]
    # subtract time0 from all timestamps
    for node in nodes:
        node["start_time"] -= time0
        node["end_time"] -= time0

    df = pd.DataFrame(nodes)
    # rename "node" to "method"
    df = df.rename(columns={"node": "Function Call", "time": "time (in seconds)"})
    fig = px.bar(df,
                    base="start_time",
                    x="time (in seconds)",
                    y="Function Call",
                    color="time (in seconds)",
                    hover_name="Function Call",
                    hover_data={
                        "start_time": False,
                        "end_time": False,
                        "time (in seconds)": ":.3f",
                        "Function Call": False,
                    },
                    color_continuous_scale="bluered",
                    color_continuous_midpoint=2,
                    height=min(170 + 30 * len(nodes), 600),
                    orientation='h'
                    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plotly_plot(G):
    import plotly.graph_objects as go

    pos = nx.kamada_kawai_layout(G)

    # Create a Plotly figure for the graph
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='lines+markers+text',
        textposition='top right',
        hoverinfo='none',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                xanchor='left',
                titleside='right'
            )
        ),
        showlegend=False
    )

    node_text = []
    for node, data in G.nodes(data=True):
        node_text.append(f"{data['attr']['name']}: {round(data['attr']['time'], 3)}s")

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(l=20, r=20, t=0, b=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    camera = dict(eye=dict(x=2.0, y=2.0))
    fig.update_layout(scene_camera=camera)
    return fig


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, text):
    st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        padding-left: 25px !important;
    }
    .logo-img {
        float:right;
        width:100px;
        height:100px;
        
    }
    </style>
    """,
    unsafe_allow_html=True
)
    img_html = f'''<div class="container">
                <img class="logo-img" src='data:image/png;base64,{img_to_bytes(img_path)}'><h1 class="logo-text">{text}</h1>
                </div>'''
    return img_html
