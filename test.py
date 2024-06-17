# import os
# from langchain_openai import ChatOpenAI
from utils import get_serper_api_key, get_openai_base_url, get_aws_access_key, \
get_aws_secret_access_key, get_aws_region, get_aws_model_id
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
# from langchain.schema import HumanMessage, SystemMessage

# chat = ChatOpenAI(
#     base_url="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
#     model = get_aws_model_id(),
#     temperature=0.1
# )

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that im using to make a test request to."
#     ),
#     HumanMessage(
#         content="test from litellm. tell me why it's amazing in 1 sentence"
#     ),
# ]
# response = chat.invoke(messages)

# print(response)






# ########################################
# import os
# from litellm import completion

# os.environ["AWS_ACCESS_KEY_ID"] = get_aws_access_key()
# os.environ["AWS_SECRET_ACCESS_KEY"] = get_aws_secret_access_key()
# os.environ["AWS_REGION_NAME"] = get_aws_region()

# response = completion(
#   model=f"bedrock/{get_aws_model_id()}",
#   messages=[{ "content": "Hello, how are you?","role": "user"}]
# )

# print(response)

################################################
from crewai_tools import FileReadTool
read_resume = FileReadTool(file_path='./fake_resume.md')
print(read_resume)