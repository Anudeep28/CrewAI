# Warning control
from pyexpat.errors import messages
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew


import os
from utils import get_openai_api_key, pretty_print_result, get_gemini_api_key
from utils import get_serper_api_key, get_openai_base_url, get_aws_access_key, \
get_aws_secret_access_key, get_aws_region, get_aws_model_id


# Tools to be imported
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
#with open('./fake_resume.md', encoding='utf-8') as infile:
#read_resume = FileReadTool(file_path='./fake_resume.md') # type: ignore
#semantic_search_resume = MDXSearchTool(mdx='./fake_resume.md')

# Serper API
os.environ["SERPER_API_KEY"] = get_serper_api_key() # type: ignore

######### When using openai
# import openai
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.base_url = os.environ['OPENAI_BASE_URL']
###########################

############################## For using gemini ###############
from langchain_google_genai import ChatGoogleGenerativeAI
import os

api_key = get_gemini_api_key()  # replace with your actual API key if not stored in environment variables
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key) # type: ignore
#################################################################

#news_scarpe_tool = ScrapeWebsiteTool(website_url="https://docs.crewai.com/how-to/Customizing-Agents/#key-attributes-for-customization")

############################## When using chatopenai function
from langchain_openai import ChatOpenAI
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
###########################################################
# Creating Agents for the work
# Agent 1
# Agent 1: Researcher
researcher = Agent(
    role="Tennis coaching Researcher",#"Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "Tennis coaching plans to help create a perfect weekday coaching plan",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Tennis Researcher, your prowess in "
        "navigating and extracting critical "
        "Extract information only from top 5 relevant urls"
        "information of professional tennis coaching techniques is unmatched."
        "Your skills help pinpoint the necessary "
        "training drills, techniques and plans necessary for Professional tennis training "
        "coaching, forming the foundation for "
        "effective professional tennis training."
    ),
    llm=llm_gemini,
)

 # Agent 2
# Agent 2: Profiler
profiler = Agent(
    role="Personal Planner for Coaches",
    goal="Do increditble research on Effective Tennis Training Practices "
         "to help prepare a well thought out plan for coaching over the week",
    tools = [scrape_tool, search_tool],
             #read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional Tennis plan, laying the "
        "groundwork for personalized training."
    ),
    llm=llm_gemini,
)

# # Agent 3: 
# # Agent 3: Resume Strategist
# resume_strategist = Agent(
#     role="Resume Strategist for Engineers",
#     goal="Find all the best ways to make a "
#          "resume stand out in the job market.",
#     tools = [scrape_tool, search_tool,
#              read_resume, semantic_search_resume],
#     verbose=True,
#     backstory=(
#         "With a strategic mind and an eye for detail, you "
#         "excel at refining resumes to highlight the most "
#         "relevant skills and experiences, ensuring they "
#         "resonate perfectly with the job's requirements."
#     ),
#     llm=llm, # using gemini for it
# )

# # Agent 4: Interview Preparer
# interview_preparer = Agent(
#     role="Engineering Interview Preparer",
#     goal="Create interview questions and talking points "
#          "based on the resume and job requirements",
#     tools = [scrape_tool, search_tool,
#              read_resume, semantic_search_resume],
#     verbose=True,
#     backstory=(
#         "Your role is crucial in anticipating the dynamics of "
#         "interviews. With your ability to formulate key questions "
#         "and talking points, you prepare candidates for success, "
#         "ensuring they can confidently address all aspects of the "
#         "job they are applying for."
#     ),
#     llm=llm,
# )



# Creating Venue Pydantic Object
# Create a class VenueDetails using Pydantic BaseModel.
# Agents will populate this object with information about different venues by creating different instances of it.

# from pydantic import BaseModel
# # Define a Pydantic model for venue details 
# # (demonstrating Output as Pydantic)
# class VenueDetails(BaseModel):
#     name: str
#     address: str
#     capacity: int
#     booking_status: str

# Creating Tasks
# By using output_json, you can specify the structure of the output you want.
# By using output_file, you can get your output in a file.
# By setting human_input=True, the task will ask for human feedback (whether you like the results or not) before finalising it.
# Creating tasks for the agents
# Task for Data Analyst Agent: Analyze Market Data


# Task for Profiler Agent: Compile Comprehensive Profile
research_task = Task(
    description=(
        "Compile a detailed professional tennis coaching plan for ({Coach_name}) "
        #"using the GitHub ({github_url}) URLs, and personal write-up "
        "Utilize tools provided to research and extract "
        "synthesize information from the sources."
        " The plan should be for an minimum 2 hour coaching strategy each day"
    ),
    expected_output=(
        "A comprehensive profile document that includes Tennis coaching plan for each day, "
        "Detailed rough plan for Monday, Tuesday, Wednesday, Thursday, Friday and Saturday "
        
    ),
    agent=researcher,
    #async_execution=True
)

# Task for Resume Strategist Agent: Align Resume with Job Requirements
tennis_strategy_task = Task(
    description=(
        "Using the extracted comprehensive information obtained from "
        "previous tasks, tailor a well thought out plan to highlight the most "
        "tennis coaching practices each day. Employ tools to adjust and enhance the "
        "for any irrelevant plan or technique. Make sure this is the best plan but "
        "don't make up any information. Update every part, but do not take long time "
        #"inlcuding the initial summary, work experience, skills, "
        "All to better reflect the best professional plan "
        #"abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated coaching plan document that effectively give a 2 hour every day coachig plan "
        #"qualifications and experiences relevant to the job."
    ),
    output_file="tailored_tennis_plan.md",
    context=[research_task],
    agent=profiler
)



# Creating the Crew
# The Process class helps to delegate the workflow to the Agents (kind of like a Manager at work)
# In the example below, it will run this hierarchically.
# manager_llm lets you choose the "manager" LLM you want to use.
# Define the crew with agents and tasks
job_application_crew = Crew(
    agents=[researcher,
            profiler],

    tasks=[research_task,
           tennis_strategy_task],

    verbose=True
)

# Running the Crew
# Set the inputs for the execution of the crew.

# Example data for kicking off the process
job_application_inputs = {
    'Coach_name': 'Anudeep Patil',
}
# Note 1: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

# Note 2:

# Since you set human_input=True for some tasks, the execution will ask for your input before it finishes running.
# When it asks for feedback, use your mouse pointer to first click in the text box before typing anything.
### this execution will take a few minutes to run
result = job_application_crew.kickoff(inputs=job_application_inputs)

print(result)

# Display the generated marketing_report.md file.
# Note: After kickoff execution has successfully ran, wait an extra 45 seconds for the marketing_report.md file to be generated. If you try to run the code below before the file has been generated, your output would look like:

# marketing_report.md
# If you see this output, wait some more and than try again.

