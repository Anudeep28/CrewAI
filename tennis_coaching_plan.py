import os
from crewai import Agent, Task, Crew
from llm_utils import get_gemini_llm, get_local_llm, SERPER_API_KEY


# Tools to be imported

from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

os.environ["SERPER_API_KEY"] = SERPER_API_KEY # type: ignore

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
###########################################################
llm_gemini = get_gemini_llm()
llm = get_local_llm()
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

