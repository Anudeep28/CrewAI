# Warning control
from pyexpat.errors import messages
import warnings
warnings.filterwarnings('ignore')

import os
from utils import get_openai_api_key, pretty_print_result, get_gemini_api_key
from utils import get_serper_api_key, get_openai_base_url, get_aws_access_key, \
get_aws_secret_access_key, get_aws_region, get_aws_model_id

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


#with open('./fake_resume.md', encoding='utf-8') as infile:
#read_resume = FileReadTool(file_path='./fake_resume.md') # type: ignore
#semantic_search_resume = MDXSearchTool(mdx='./fake_resume.md')

# Serper API
SERPER_API_KEY = get_serper_api_key() # type: ignore

######### When using openai
# import openai
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.base_url = os.environ['OPENAI_BASE_URL']
###########################

############################## For using gemini ###############

def get_gemini_llm():
    api_key = get_gemini_api_key()  # replace with your actual API key if not stored in environment variables
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key) # type: ignore
    return llm_gemini
#################################################################

#news_scarpe_tool = ScrapeWebsiteTool(website_url="https://docs.crewai.com/how-to/Customizing-Agents/#key-attributes-for-customization")


############################## When using chatopenai function
def get_local_llm():
    llm = ChatOpenAI(
        api_key=get_openai_api_key(),
        #model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model = "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        # model = "bartowski/Phi-3-medium-128k-instruct-GGUF",
        # model = "bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF",
        #model = "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
        #model = "QuantFactory/TextBase-7B-v0.1-GGUF",
        base_url= get_openai_base_url(),
        temperature=0.1,

    )
    return llm
###############################################################

################## AWS CLient ############################
# from langchain_aws import BedrockChat
# # Replace with your actual AWS Access Key ID and Secret Access Key

# aws_access_key_id = get_aws_access_key()
# aws_secret_access_key = get_aws_secret_access_key()

# # Initialize the language model
# llm = BedrockChat(model_id=get_aws_model_id(), 
#                   model_kwargs={'temperature':0.1}, 
#                   #aws_access_key_id=aws_access_key_id, 
#                   #aws_secret_access_key=aws_secret_access_key,
#                   region_name=get_aws_region()) # type: ignore
#import os
###########
# from litellm import completion
# import subprocess

# os.environ["AWS_ACCESS_KEY_ID"] = get_aws_access_key()
# os.environ["AWS_SECRET_ACCESS_KEY"] = get_aws_secret_access_key()
# os.environ["AWS_REGION_NAME"] = get_aws_region()

# # Construct the PowerShell command
# ps_command = f"$env:AWS_ACCESS_KEY_ID = '{get_aws_access_key()}'; $env:AWS_SECRET_ACCESS_KEY = '{get_aws_secret_access_key()}'; $env:AWS_DEFAULT_REGION = '{get_aws_region()}'"

# # Run the PowerShell command
# subprocess.run(["powershell", "-Command", ps_command], shell=True)


# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(
#         base_url="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
#         model = get_aws_model_id(),
#         temperature=0.1
# )