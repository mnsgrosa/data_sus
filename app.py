import streamlit as st
from src.agentic.agent import StatisticalAgent
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import glob
import numpy as np
from langchain_core.messages import AIMessage, ToolMessage

@st.cache_resource
def get_agent():
    return StatisticalAgent()

if st.session_state.get('agent') is None:
    st.session_state['agent'] = get_agent()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if "summaries" not in st.session_state:
    st.session_state['summaries'] = {}

if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

if 'agent_state' not in st.session_state:
    st.session_state['agent_state'] = {
        "messages": [],
        "report": [],
        "struct": {},
        "summary": [],
        "stat_report": [],
        "figures": []
    }

for msg in st.session_state.chat_history[-5:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if "figures" in msg and msg["figures"]:
            for idx, fig in enumerate(msg["figures"]):
                st.plotly_chart(fig, use_container_width=True, key=f"fig_{msg['timestamp']}_{idx}")

new_figures = []

with st.sidebar:
    st.markdown('# Srag agent')
    st.markdown('## Tools available: Csv downloader, Statistical report generator, Temporal graphical report generator, Numerical data summarizer, Data dictionary fetcher')
    st.markdown('---')
    st.markdown('## Columns available')
    st.markdown('''
    EVOLUCAO, UTI, DT_NOTIFIC, SG_UF_NOT, VACINA_COV, HOSPITAL, SEM_NOT <br>
    for more information ask the agent about the data dictionary
    ''')
    st.markdown('---')
    st.markdown('## Summaries about the columns')

    for year in st.session_state.summaries.keys():
        st.markdown(f'## Summary from {year}')
        for column in st.session_state.summaries[year]:
            st.markdown(f'### {column}')
            st.table(st.session_state.summaries[year][column])
        st.markdown('---')

if prompt:= st.chat_input('Chat with arxiv mcp'):
    st.session_state.chat_history.append({'role': 'user', 'content': prompt, "timestamp": len(st.session_state.chat_history)})
    with st.chat_message('user'):
        st.markdown(prompt)

    x = None
    y = None

    with st.chat_message('assistant'):
        with st.spinner('Thinking...', show_time=True):
            try:
                result = st.session_state.agent.run(
                    prompt,
                    initial_state = st.session_state['agent_state']
                )

                st.session_state['agent_state'] = result

                item_msg = False
                table_msg = False
                dict_msg = False
                store_msg = False


                for message in reversed(result['messages']):
                    if isinstance(message, ToolMessage):
                        if message.name == 'generate_temporal_graphical_report':
                            item = json.loads(message.content)
                            data = pd.DataFrame({
                                'x' : item.get('x'),
                                'y' : item.get('y')
                            })
                            chart = st.line_chart(data, x = 'x', y = 'y')
                            item_msg = True
                        elif message.name == 'generate_statistical_report':
                            items = json.loads(message.content)
                            table_msg = True
                            table = st.table(pd.DataFrame(items))
                        elif message.name == 'summarize_numerical_data':
                            items = json.loads(message.content)
                            year_dict = {}
                            for year in items.keys():
                                column_dict = {}
                                for column in items[year].keys():
                                    value_dict = {}
                                    value_dict['median'] = items[year][column]['median']
                                    for freq in items[year][column]['freq'].keys():
                                        value_dict[f'freq_{freq}'] = items[year][column]['freq'][freq]
                                    column_dict[column] = value_dict
                                st.session_state.summaries[year] = column_dict
                        elif message.name == 'get_data_dict':
                            items = json.loads(message.content)
                            dict_msg = True
                            dict_ct = st.table(pd.DataFrame(items))
                            st.session_state['struct'] = items
                        elif message.name == 'store_csvs':
                            store_msg = True

                    
                    if isinstance(message, AIMessage):
                        if item_msg:
                            st.write(f"{message.content}")
                            item_msg = False
                        elif table_msg:
                            st.write(f"{message.content} + {table}")
                            table_msg = False
                        elif dict_msg:
                            st.write(f"{message.content} + {dict_ct}")
                            dict_msg = False
                        elif store_msg:
                            st.write(f"{message.content}")
                            store_msg = False
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": message.content,
                            "figures": new_figures,
                            "timestamp": len(st.session_state.chat_history)
                        })
                        st.session_state.agent_state['messages'].append(message.content)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)