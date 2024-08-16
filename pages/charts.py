import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
from numpy import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="pyplot")

import pandas as pd
import textwrap
import re
from APIKEY import GOOGLE_API_KEY, GOOGLE_API_KEY2, GOOGLE_API_KEY3, GOOGLE_API_KEY4
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import HarmCategory, HarmBlockThreshold
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY2

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    top_p=0.85,
    top_k=40,
    n=1,
    max_retries=1,
    timeout=0.30,
    tools='code_execution',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE ,
        HarmCategory.HARM_CATEGORY_HARASSMENT : HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE}
    )

############### ############### ###################################
def handle_query(df, column_names, data_types_list, query):

    # If the "Get Answer" button is clicked
    if query is not None and query!=" ":

        st.session_state.messagess.append({"role": "user", "content": query})
        # with st.chat_message("user"):
        #     st.markdown(query)
        
        # Ensure the query is not empty
        if query and query.strip() != "":
            # Define the prompt content
            prompt_content = f"""
            The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
            
            The DataFrame has the following columns: {column_names}
            The data type of each column are as follows: {data_types_list}
            
            Before plotting, ensure the data is ready:
            1. Intelligently find the key value from the query about which the data is to be shown in the chart.
            2. Check if columns that are supposed to be numeric are recognized as such.
            3. While getting row data, use "df[df['key_column_name'].str.lower().str.contains('key_value')]" to match with the particular row key value.
            4. Check if the final df contains all the columns and rows for chart creation.
            5. Only show the numeric column data in the charts.
            6. Use specific text columns to show data regarding to them on chart.
            
            Use package Pandas and Matplotlib ONLY.
            Provide SINGLE CODE BLOCK with a solution using Pandas and Matplotlib plots in a single figure to address the following query:
            
            {query}

            - ONLY USE THE CHART WHICH IS STATED IN THE QUERY TO CREATE.
            - ALWAYS CREATE A "fig" OBJECT FOR GRAPH.
            - WHEN GETTING THE DATA FOR ROWS CHECK YOU ARE GETTING IT CORRECTLY AS MENTIONED IN THE FILE.
            - USE SINGLE CODE BLOCK with a solution. 
            - Do NOT EXPLAIN the code 
            - DO NOT COMMENT the code. 
            - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
            - DEBUG YOUR CODE ALWAYS BEFORE GIVING FINAL ANSWER
            - The code block must start and end with ```
            
            - Example code format ```code```
        
            - Colors to use for background and axes of the figure : #0E1117
            - Colors to use for axes labels and legend labels: 'white'
            - Try to use the combination of following color palette for coloring the plots : "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"
            
            """

            with st.spinner("📟 Generating plot..."):
                response = llm.invoke(prompt_content)
                result = response.content.strip()
                # st.write(result)

            execute_code(result, df, query)

    elif query==" ":
        st.warning("Please enter a Query")


def extract_code_from_markdown(md_text):
    # Extract code between the delimiters
    code_blocks = re.findall(r"```(python)?(.*?)```", md_text, re.DOTALL)

    # Strip leading and trailing whitespace and join the code blocks
    code = "\n".join([block[1].strip() for block in code_blocks])

    return code


def execute_code(response_text: str, df: pd.DataFrame, query):
    # Extract code from the response text
    code = extract_code_from_markdown(response_text)
    code = code + "\nst.pyplot(fig)"
    
    if code:
        st.session_state.messagess.append({"role": "assistant", "content": code})
      
    else:
        st.write(response_text)
        st.session_state.messagess.append({"role": "assistant", "content": code})
 

########### ########### ############ ########### ############
if "messagess" not in st.session_state:
    st.session_state.messagess = [{'role': 'assistant', 'content': """Hello! 👋  Welcome to Your AI Financial Assistant! I'm here to help you analyze and understand your financial documents."""}]
if "key" not in st.session_state:
    st.session_state.key = str(random.randint(1000, 1000000000))
######## ############## ############# ############ ############
st.title("Chattergy")

options = st.radio("Data Usage", index=1,options=["Chat", "Visualize"], horizontal=True, label_visibility="collapsed")

query = st.chat_input("Enter your Prompt:")

if options == "Chat":
    switch_page("chat")
    df = None
else:
    data_upload = st.file_uploader("Upload a data file", type=["csv", "xlsx", "xls"], label_visibility='collapsed', key=st.session_state.key)
    
    if data_upload:
        # Check the type of file uploaded and read accordingly
        if data_upload.name.endswith('.csv'):
            df = pd.read_csv(data_upload)
            data_types_list = df.dtypes.tolist()
        elif data_upload.name.endswith('.xlsx') or data_upload.name.endswith('.xls'):
            df = pd.read_excel(data_upload)
            data_types_list = df.dtypes.tolist()
        else:
            df = None
            data_types_list=None

        # If data is uploaded successfully
        if df is not None:
            # Create an expander to optionally display the uploaded data
            with st.expander("Show Data"):
                st.write(df)

            # Extract column names for further processing
            column_names = ", ".join(df.columns)
            # st.write(":blue-background[:red[Columns]]: ", column_names)

            # Check if the uploaded DataFrame is not empty
            if not df.empty:
                # Handle the OpenAI query and display results
                handle_query(df, column_names, data_types_list, query)
            else:
                # Display a warning if the uploaded data is empty
                st.warning("The given data is empty.")

######### ############## ############## ############ ############


for message in st.session_state.messagess:
    if message["role"]=="user":
        with st.chat_message(message["role"]):
            # st.markdown("user")
            st.markdown(message["content"])
    else:
        content = message["content"]
        if "st.pyplot(fig)" not in content:
            with st.chat_message(message["role"]):
                st.markdown(content)
        else:
            try:
                with st.chat_message(message["role"]):
                    exec(textwrap.dedent(content))
                    st.write("Note : Right click on the image to download the image")
            except Exception as e:
                error_message = str(e)
                st.error(f"📟 Apologies, failed to execute the code due to the error: {error_message}")

########## ############ ############# ########### ############ ########
if st.button(label="Clear File") and 'key' in st.session_state.keys():
   st.session_state.pop('key')
   st.rerun()



