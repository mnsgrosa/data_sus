import streamlit as st
from src.agentic.agent import StatisticalAgent
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json
import glob

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

if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

if 'agent_state' not in st.session_state:
    st.session_state['agent_state'] = {
        "messages": [],
        "report": [],
        "insights": [],
        "struct": {},
        "summary": [],
        "stat_report": [],
        "figures": []
    }

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        if "figures" in msg and msg["figures"]:
            for idx, fig in enumerate(msg["figures"]):
                st.plotly_chart(fig, use_container_width=True, key=f"fig_{msg['timestamp']}_{idx}")

new_figures = []

if prompt:= st.chat_input('Chat with arxiv mcp'):
    st.session_state.messages.append({'role': 'user', 'content': prompt, "timestamp": len(st.session_state.chat_history)})
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

                st.session_state.agent_state = result

                last_message = None
                for msg in reversed(result['messages']):
                    if isinstance(msg, st.session_state.agent.llm_tool_caller.__class__.__bases__[0]):
                        last_message = msg.content
                        break

                response_text = last_message or "Task completed."

                for message in result['messages']:
                    if message.type == 'tool':
                        if message.name == 'generate_temporal_graphical_report':
                            item = json.loads(message.content)
                            px.line(x = item.get('x'), y = item.get('y')).write_image('./line_chart.png')
                            st.image('./line_chart.png')
                        elif message.name == 'generate_statistical_report':
                            ans = {
                                'death_count': [],
                                'death_rate': [],
                                'new_cases': [],
                                'cases_uti': [],
                                'cases_hospitalized': [],
                                'perc_uti': [],
                                'perc_vaccinated': []
                            }

                            items = json.loads(message.content)
                            st.write(items)
                            for item in items:
                                ans['death_count'].append(item[0])
                                ans['death_rate'].append(item[1])
                                ans['new_cases'].append(item[2])
                                ans['cases_uti'].append(item[3])
                                ans['cases_hospitalized'].append(item[4])
                                ans['perc_uti'].append(item[5])
                                ans['perc_vaccinated'].append(item[6])
                            
                            st.write(ans)

                    
                # if result.get("summary"):
                #     with st.expander("📊 Summary Statistics"):
                #         for summary in result["summary"]:
                #             st.json(summary)
                
                # if result.get("report"):
                #     with st.expander("📄 Statistical Reports"):
                #         if fig:
                #             st.line_chart(x = x, y = y)
                
                # st.session_state.chat_history.append({
                #     "role": "assistant",
                #     "content": response_text,
                #     "new_figures": fig,
                #     "timestamp": len(st.session_state.chat_history)
                # })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)