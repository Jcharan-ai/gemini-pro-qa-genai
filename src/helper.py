from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from src.prompt import TEMPLATE, TEMPLATE2


load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

llm = ChatGoogleGenerativeAI(genai_api_key=os.getenv("GENAI_API_KEY"), model='gemini-1.5-pro')




quize_generation_prompt = PromptTemplate(
    input_variables=['text','number','subject','tone','response_json'],
    template=TEMPLATE
)

quiz_chain = LLMChain(llm=llm, prompt=quize_generation_prompt, output_key='quiz', verbose=True)




quize_evaluation_prompt = PromptTemplate(
    input_variables=['subject','quiz'], 
    template=TEMPLATE2
)

review_chain = LLMChain(llm=llm, prompt=quize_evaluation_prompt, output_key='review', verbose=True)


# Combining both quiz_chain and review_chain into a single chain  
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain], 
    input_variables=['text','number','subject','tone','response_json'],
    output_variables=['quiz','review'],
    verbose=True
)