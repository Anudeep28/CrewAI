# Warning control
from pyexpat.errors import messages
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from llm_utils import get_local_llm, get_gemini_llm, SERPER_API_KEY

import os

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
read_resume = FileReadTool(file_path='./fake_resume.md') # type: ignore
semantic_search_resume = MDXSearchTool(mdx='./fake_resume.md')

# Serper API
os.environ["SERPER_API_KEY"] = SERPER_API_KEY # type: ignore

######### When using openai
#llm = get_gemini_llm()
llm = get_local_llm()
###########################################################
# Creating Agents for the work
# Agent 1
# Agent 1: Researcher
researcher = Agent(
    role="Resume content Researcher",
    goal="Make sure to do amazing analysis on "
         "requirements of the resume to help modify the resume",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a resume content Researcher, your prowess in "
        "navigating and extracting critical "
        "information on the provided points/context is unmatched."
        "Your skills help recognise the necessary "
        "information and skills needed to be included in the resume "
        "sought by employers, forming the foundation for "
        "effective application tailoring."
    ),
    llm=llm,
)

 # Agent 2
# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Financial services",
    goal="Do increditble research on job applicant "
         "to help the resumes aligned with the points/context provided and market",
    tools = [scrape_tool, search_tool,
             read_resume],# semantic_search_resume],
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    ),
    llm=llm,
)

# Agent 3: 
# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Financial sector",
    goal="Find all the best ways to make a "
         "resume align with points/context provided and the job market.",
    tools = [scrape_tool, search_tool,
             read_resume],# semantic_search_resume],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resume to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the provided points/context."
    ),
    llm=llm, # using gemini for it
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Financial services Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and provided points/context",
    tools = [scrape_tool, search_tool,
             read_resume],#, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    ),
    llm=llm,
)



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
# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the points/context provided ({job_posting_url}) "
        "to extract key skills, experiences, and projects "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements. Look at max top 5 sources"
    ),
    expected_output=(
        "A structured list of extracted information on provided points/context, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed professional profile "
        "using the personal write-up and resume imported "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)

# You can pass a list of tasks as context to a task.
# The task then takes into account the output of those tasks in its execution.
# The task will not run until it has the output(s) from those tasks.

# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and provided points/context and extracted information from "
        "previous tasks, Update or Add the relevant information to the resume to the "
        "work experience areas. Employ tools to adjust, Quantify the accomplishments where ever possible. "
        "Mostly add or modify the existing sentences in work experience section of the resume. " 
        "Make sure the resume is aligned with the points/context and extracted information. "
        "Donot take much time doing it, Max 3 tries to refine the context"
        "Don't make up any information only use the information provided. "
        "Update only the skills and Work experience of Crisil section of the resume, "
    ),
    expected_output=(
        "An updated or modified resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the points/context."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)

# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and provided points/context and extracted information. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)



# Creating the Crew
# The Process class helps to delegate the workflow to the Agents (kind of like a Manager at work)
# In the example below, it will run this hierarchically.
# manager_llm lets you choose the "manager" LLM you want to use.
# Define the crew with agents and tasks
job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist],
            #interview_preparer],

    tasks=[research_task,
           profile_task,
           resume_strategy_task],
           #interview_preparation_task],

    verbose=True
)

# Running the Crew
# Set the inputs for the execution of the crew.

# Example data for kicking off the process
job_application_inputs = {
    'job_posting_url': """Credit Risk Model Development, Credit Risk on-going model monitoring,
      IFRS9 understanding, model reviews as per SR 11-7. Understanding of PD, LGD and EAD models and EL. Understanding of Basel-1,2,3 and exposure to Credit Risk""",
    'personal_writeup': """Anudeep is an accomplished Credit Risk Expert, Leader with 7 years of experience, specializing in
    managing remote and in-office teams, and expert in multiple
    programming languages and Financial Knowledge. He holds an MTech degree in 
    Mathematical Modeling and Simulation and a strong background in 
    AI and data science."""
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


