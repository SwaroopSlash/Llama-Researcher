import sys
import asyncio
import streamlit as st
import os
from dotenv import load_dotenv
import base64
# Import the necessary LangChain and LLM libraries
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from langchain_community.embeddings import HuggingFaceEmbeddings
from workflow import ResearchAssistantWorkflow

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API"))



# Define the main async function for running the workflow
async def run_workflow(topic):
    # Load environment variables from .env file
    load_dotenv()

    # Initialize the ChatGroq LLM and embedding models
    llm = Gemini()
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize the research assistant workflow with LLM and embedding model
    workflow = ResearchAssistantWorkflow(llm=llm, embed_model=embed_model, verbose=True, timeout=240.0)

    # Run the workflow with the user-provided query/topic
    report_file = await workflow.run(query=topic)
   
    
    return report_file

# Streamlit app logic
st.title("Research Provider: Your AI assistant")

# Input field to accept the topic for the research assistant workflow
topic = st.text_input("Enter the research topic for the AI assistant to generate report:")

# Button to trigger the workflow
if st.button("Generate Report"):
    if topic:

        with st.spinner("Generating report... this may take a few moments."):

        # Run the asynchronous workflow and get the report file and final report text
            report_file = asyncio.run(run_workflow(topic))
        
        # Display the PDF and final report
        if report_file:
            st.success("Report generated successfully!")

            # Show the PDF in an iframe within Streamlit
            with open(report_file, "rb") as file:
                pdf_data = file.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                        
            # Provide a download button for the PDF
            st.download_button(
                label="Download Report PDF",
                data=pdf_data,
                file_name="report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("No report generated.")
    else:
        st.error("Please enter a topic.")
