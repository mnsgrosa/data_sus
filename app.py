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
        data_df = pd.DataFrame()
        chart = False
        with st.spinner("Thinking...", show_time=True):
            answer = None
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt, "data": None}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "http://localhost:8000/prompt", params={"user_input": prompt}
                )

            if response.status_code != 200:
                st.error(f"Error: {response.status_code}")
            else:
                answer = response.json()

            if answer:
                ans = answer.get("content", "")
                data = answer.get("data", [])
                tool_name = answer.get("tool_name", "")

                if "summarize" in tool_name:
                    rows = []
                    for year, columns in data.items():
                        for col_name, col_data in columns.items():
                            rows.append(
                                {
                                    "Year": year,
                                    "Column": col_name,
                                    "Metric": "Median",
                                    "Value": col_data.get("median"),
                                }
                            )

                            if "freq" in col_data:
                                for category_code, count in col_data["freq"].items():
                                    rows.append(
                                        {
                                            "Year": year,
                                            "Column": col_name,
                                            "Metric": f"Category {category_code}",
                                            "Value": count,
                                        }
                                    )

                    data_df = pd.DataFrame(rows)

                elif "statistical" in tool_name:
                    st.json(data)
                elif "graphical" in tool_name:
                    data_df = pd.DataFrame(({"x": data["x"], "y": data["y"]}))
                    chart = True

                with st.chat_message("assistant"):
                    if not data_df.empty and not chart:
                        st.dataframe(data_df)
                    if chart:
                        st.line_chart(data_df, x="x", y="y")
                    st.markdown(ans)


if __name__ == "__main__":
    chat_page()
