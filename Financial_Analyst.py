# Warning control
from pyexpat.errors import messages
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew


import os
from utils import get_openai_api_key, pretty_print_result
from utils import get_serper_api_key, get_openai_base_url



from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool, \
                         ScrapeWebsiteTool

# Serper API
os.environ["SERPER_API_KEY"] = get_serper_api_key() # type: ignore

######### When using openai
# import openai
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.base_url = os.environ['OPENAI_BASE_URL']
###########################


#news_scarpe_tool = ScrapeWebsiteTool(website_url="https://docs.crewai.com/how-to/Customizing-Agents/#key-attributes-for-customization")

# When using chatopenai function
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=get_openai_api_key(),
    #model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model = "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
    # model = "bartowski/Phi-3-medium-128k-instruct-GGUF",
    # model = "bartowski/Llama-3-8B-Instruct-Gradient-1048k-GGUF",
    #model = "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
    base_url= get_openai_base_url(),
    temperature=0.1,

)

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Creating Agents for the work
# Agent 1
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time "
         "to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent "
              "uses statistical modeling and machine learning "
              "to provide crucial insights. With a knack for data, "
              "the Data Analyst Agent is the cornerstone for "
              "informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

 # Agent 2
trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based "
         "on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial "
              "markets and quantitative analysis, this agent "
              "devises and refines trading strategies. It evaluates "
              "the performance of different approaches to determine "
              "the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

# Agent 3: 
execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies "
         "based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, "
              "and logistical details of potential trades. By evaluating "
              "these factors, it provides well-founded suggestions for "
              "when and how trades should be executed to maximize "
              "efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks "
         "associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models "
              "and market dynamics, this agent scrutinizes the potential "
              "risks of proposed trades. It offers a detailed analysis of "
              "risk exposure and suggests safeguards to ensure that "
              "trading activities align with the firm’s risk tolerance.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
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
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for "
        "the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to "
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market "
        "opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
    llm=llm,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on "
        "the insights from the Data Analyst and "
        "user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} "
        "that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)


# Task for Trade Advisor Agent: Plan Trade Execution
execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the "
        "best execution methods for {stock_selection}, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Detailed execution plans suggesting how and when to "
        "execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

# Task for Risk Advisor Agent: Assess Trading Risks
risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading "
        "strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks "
        "and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)

# Creating the Crew
# The Process class helps to delegate the workflow to the Agents (kind of like a Manager at work)
# In the example below, it will run this hierarchically.
# manager_llm lets you choose the "manager" LLM you want to use.
# Define the crew with agents and tasks
from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
financial_trading_crew = Crew(
    agents=[data_analyst_agent, 
            trading_strategy_agent, 
            execution_agent, 
            risk_management_agent],
    
    tasks=[data_analysis_task, 
           strategy_development_task, 
           execution_planning_task, 
           risk_assessment_task],
    
    manager_llm=llm,#ChatOpenAI(model="gpt-3.5-turbo", 
                #           temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)

# Running the Crew
# Set the inputs for the execution of the crew.

# Example data for kicking off the process
financial_trading_inputs = {
    'stock_selection': 'RPTECH.NS',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

# Note 1: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

# Note 2:

# Since you set human_input=True for some tasks, the execution will ask for your input before it finishes running.
# When it asks for feedback, use your mouse pointer to first click in the text box before typing anything.
### this execution will take some time to run
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

print(result)

# Display the generated marketing_report.md file.
# Note: After kickoff execution has successfully ran, wait an extra 45 seconds for the marketing_report.md file to be generated. If you try to run the code below before the file has been generated, your output would look like:

# marketing_report.md
# If you see this output, wait some more and than try again.

