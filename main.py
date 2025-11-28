from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:

        agent = create_csv_agent(
            OpenAI(temperature=0), csv_file, verbose=True,allow_dangerous_code=True
)

        user_question = st.text_input("Ask a question about your CSV: ")
        CSV_PROMPT_PREFIX = """First set the pandas display options to show all the columns,get the column names, then answer the question.
"""

        CSV_PROMPT_SUFFIX = """
        - **ALWAYS** before giving the Final Answer, try another method.
        Then reflect on the answers of the two methods you did and ask yourself
        if it answers correctly the original question.
        If you are not sure, try another method.
        FORMAT 4 FIGURES OR MORE WITH COMMAS.
        - If the methods tried do not give the same result,reflect and
        try again until you have two methods that have the same result.
        - If you still cannot arrive to a consistent result, say that
        you are not sure of the answer.
        - If you are sure of the correct answer, create a beautiful
        and thorough response using Markdown.
        - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
        ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
        - **ALWAYS**, as part of your "Final Answer", explain how you got
        to the answer on a section that starts with: "\n\nExplanation:\n".
        In the explanation, mention the column names that you used to get
        to the final answer.
        """

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                QUERY = CSV_PROMPT_PREFIX + user_question + CSV_PROMPT_SUFFIX
                st.write(agent.run(QUERY))


if __name__ == "__main__":
    main()