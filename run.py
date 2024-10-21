import sys
import asyncio
import subprocess
import os
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
#from llama_index.utils.workflow import draw_all_possible_flows
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from workflow import ResearchAssistantWorkflow
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API"))



async def main():
    
    llm = Gemini()
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    workflow = ResearchAssistantWorkflow(llm=llm, embed_model=embed_model, verbose=True, timeout=240.0)

    # draw_all_possible_flows(workflow, filename="research_assistant_workflow.html")
    topic = sys.argv[1]
    report_file = await workflow.run(query=topic)

    os.startfile(report_file)


if __name__ == "__main__":
    asyncio.run(main())
