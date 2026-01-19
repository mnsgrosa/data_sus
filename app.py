import httpx
import pandas as pd
import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def chat_page():
    for message in st.session_state.chat_history[-5:]:
        with st.chat_message(message["role"]):
            if message["data"]:
                st.line_chart(message["data"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Srag chat"):
        with st.spinner("Thinking...", show_time=True):
            answer = None
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt, "data": None}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "http://localhost:8000/prompt", json={"prompt": prompt}
                )

            if response.status_code != 200:
                st.error(f"Error: {response.status_code}")
            else:
                answer = response.json()

            if answer:
                ans = answer.get("content", "")
                data = answer.get("data", [])

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": ans, "data": None}
                )

                if data:
                    x = data.get("x", [])
                    y = data.get("y", [])
                    if x and y:
                        df = pd.DataFrame({"x": x, "y": y})
                        chart = st.line_chart(df, x="x", y="y")
                        ans = ans + f"{chart}"

                    else:
                        ans = ans + f"{pd.DataFrame(data)}"

                with st.chat_message("assistant"):
                    st.markdown(answer)


if __name__ == "__main__":
    chat_page()
