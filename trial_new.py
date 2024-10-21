import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from langchain_groq import ChatGroq

from llama_index.llms.gemini import Gemini
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API"))

# Initialize the LLM (can be replaced with Groq, etc.)
llm = ChatGroq(model="llama3-70b-8192", api_key = "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"),

# Streamlit interface for uploading JD and Resume
st.title("Interview Assistant Agent")

uploaded_jd = st.file_uploader("Upload Job Description (JD)", type=["txt", "pdf"])
uploaded_resume = st.file_uploader("Upload Candidate Resume", type=["txt", "pdf"])

# Store the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if uploaded_jd and uploaded_resume:
    jd_content = uploaded_jd.read().decode("latin-1")
    resume_content = uploaded_resume.read().decode("latin-1")

    # Displaying contents
    st.subheader("Job Description")
    
    
    st.subheader("Candidate Resume")
    

    # Template for generating questions
    prompt_template = """
    Based on the following job description and candidate resume, generate {num_questions} interview questions:
    
    Job Description: {job_description}
    
    Candidate Resume: {resume}
    
    Previous Answer: {user_answer}
    
    Generate a follow-up question if applicable, considering the answer.
    """

    # Number of questions input
    num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)

    # Set up the prompt with the updated template
    prompt = PromptTemplate(
        input_variables=["job_description", "resume", "user_answer", "num_questions"],
        template=prompt_template,
    )

    # Create an LLM chain
    question_generator_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    # Start interview
    if st.button("Start Interview"):
        initial_questions = question_generator_chain.run(
            job_description=jd_content,
            resume=resume_content,
            user_answer="",  # No user answer for initial questions
            num_questions=num_questions
        )
        
        st.subheader("Initial Questions")
        st.write(initial_questions)
        st.session_state.conversation_history.append({"questions": initial_questions})

    # Ask user for an answer and generate follow-up
    user_answer = st.text_input("Your answer")

    if user_answer:
        # Generate follow-up question based on the answer
        follow_up_question = question_generator_chain.run(
            job_description=jd_content,
            resume=resume_content,
            user_answer=user_answer,
            num_questions=1  # One follow-up question at a time
        )
        
        st.session_state.conversation_history.append({"answer": user_answer, "follow-up": follow_up_question})
        
        st.subheader("Follow-up Question")
        st.write(follow_up_question)

# Display conversation history
st.subheader("Conversation History")
for entry in st.session_state.conversation_history:
    if "questions" in entry:
        st.write(f"Questions: {entry['questions']}")
    if "answer" in entry:
        st.write(f"Answer: {entry['answer']}")
        st.write(f"Follow-up Question: {entry['follow-up']}")
