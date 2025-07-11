import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from states import State
from prompts import *
from tools import *

API_KEY = "**"

# llm define
llm = ChatOpenAI(
    model="gpt-4o-mini",  # $0.60 / 1M input tokens
    api_key=API_KEY, 
    temperature=0.0
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def extract_answer(text):
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    
    return text

def create_planner_node(state: State):
    logger.info("*** Running Create Planner node ***")
    # Creates a messages array with system prompt and formatted user prompt, create a plan
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)

    # Converts the response to JSON and extracts the plan
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    plan = json.loads(extract_json(extract_answer(response['content'])))

    # Updates the state's message history with the plan
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("*** Running Update Planner node ***")
    plan = state['plan']
    goal = plan['goal']
    # Extends the message history with system prompt and update plan prompt
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan = plan, goal=goal))])
    messages = state['messages']
    while True: # handle potential JSON parsing errors
        try:
            response = llm.invoke(messages)

            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            plan = json.loads(extract_json(extract_answer(response['content'])))

            state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            messages += [HumanMessage(content=f"Json format error:{e}")]
            
def execute_node(state: State):
    logger.info("*** Running execute_node ***")
  
    plan = state['plan']

    # Debug: Log the plan structure
    logger.info(f"Plan structure: {json.dumps(plan, indent=2, ensure_ascii=False)}")

    steps = plan['steps']
    current_step = None
    current_step_index = 0
    
    # Find first pending step
    for i, step in enumerate(steps):
        status = step['status']
        if status == 'pending':
            current_step = step
            current_step_index = i
            break
        
    logger.info(f"Executing STEP:{current_step}")
    
    ## 此处只是简单跳转到report节点，实际应该根据当前STEP的描述进行判断
    if current_step is None or current_step_index == len(steps)-1:
        return Command(goto='report')
    
    messages = state['observations'] + [
        SystemMessage(content=EXECUTE_SYSTEM_PROMPT), 
        HumanMessage(content=EXECUTION_PROMPT.format(
            user_message=state['user_message'], 
            step=current_step['description']
        ))
    ]
    
    tool_result = None
    llm_with_tools = llm.bind_tools([create_file, str_replace, shell_exec])
    while True:
        # tell the LLM which tools it can call
        response = llm_with_tools.invoke(messages)
        # Add the AI response to messages BEFORE processing tool calls
        messages.append(response)
        response = response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)

        # what response looks like after the above lines: 
        # {
        #     "content": "",
        #     "additional_kwargs": { ... },
        #     "response_metadata": { ... },
        #     "type": "ai",
        #     "id": "run--0c0d08e9-0277-4883-9ffa-023cac2bb799-0",
        #     "example": false,
        #     "tool_calls": [ ... ],
        #     "invalid_tool_calls": [],
        #     "usage_metadata": { ... }
        # }
        # 
        # Inside tool_calls: 
        # [
        #     {
        #         "name": "shell_exec",
        #         "args": {
        #             "command": "cat Students Social Media Addiction.csv"
        #         },
        #         "id": "call_U13MrEwTcuolip7tdaA2GMDm",
        #         "type": "tool_call"
        #     }
        # ]

        tools = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec}

        # Processes tool calls in the response (either from structured tool_calls or from content)     
        if response['tool_calls']:
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                # Add tool result message
                tool_message = ToolMessage(
                    content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", 
                    tool_call_id=tool_call['id']
                )
                messages.append(tool_message)
        
        elif '<tool_call>' in response['content']:
            # logger.info("Found manual tool call in content")
            tool_call = response['content'].split('<tool_call>')[-1].split('</tool_call>')[0].strip()
            tool_call = json.loads(tool_call)
            
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_result = tools[tool_name].invoke(tool_args)
            logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
            tool_message = ToolMessage(
                content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", 
                tool_call_id=tool_call.get('id', 'manual_call')
            )
            messages.append(tool_message)
        else:    
            break
        
    logger.info(f"Current Step summary: {extract_answer(response['content'])}")
    
    # Store only the final result, not the tool execution details
    step_summary = extract_answer(response['content'])
    state['messages'].append(AIMessage(content=step_summary))
    state['observations'].append(HumanMessage(content=f"Completed step: {current_step['title']}"))
    state['observations'].append(AIMessage(content=step_summary))
    
    return Command(goto='update_planner', update={'plan': plan})
    

    
def report_node(state: State):
    """Report node that write a final report."""
    logger.info("*** Running report_node ***")
    
    messages = state['observations'] + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    llm_with_tools = llm.bind_tools([create_file, shell_exec])
    while True:
        response = llm_with_tools.invoke(messages)
        # Add the AI response to messages BEFORE processing tool calls
        messages.append(response)
        response = response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)
        tools = {"create_file": create_file, "shell_exec": shell_exec} 
        # Processes tool calls in the response
        if response['tool_calls']:    
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                # Add tool result message
                tool_message = ToolMessage(
                    content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", 
                    tool_call_id=tool_call['id']
                )
                messages.append(tool_message)
        else:
            break

    return {"final_report": response['content']}
