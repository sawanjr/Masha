from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from utils.nodes import create_qa_agent , llm
from utils.tools import create_handoff_tool, assign_to_RAG_agent, assign_to_general_agent, create_general_agent, create_psychiatrist_agent, create_react_agent
from typing import Dict, Any
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import List
import uuid
from langchain_core.messages import convert_to_messages

# --- AGENT SETUP ---
checkpointer = MemorySaver()


qa_agent = create_qa_agent(llm)
general_agent = create_general_agent(llm)
psychiatrist_agent = create_psychiatrist_agent(llm)



#Create the wrapper that extract the "counseling_note" and pass it:
def call_qa_and_extract_note(state: MessagesState) -> Dict[str, Any]:  # Return a dictionary to update the state
  rag_agent_result = qa_agent.invoke(state)

  # Find the counseling note within the RAG agent's output
  counseling_note = next(
        (
            msg.content for msg in reversed(rag_agent_result["messages"])
            if isinstance(msg, AIMessage) and "counseling_note" in msg.content
        ),
        "No counseling note provided."
    )

  # Return a dictionary to update the MessagesState directly
  return {"messages": rag_agent_result["messages"], "counseling_note": counseling_note}



supervisor_agent = create_react_agent(
    model=llm,
    tools=[assign_to_RAG_agent, assign_to_general_agent],  # ❌ Removed assign_to_psychiatrist_agent
    prompt=(
        "You are a supervisor that routes user queries to the appropriate agent. Follow this logic:\n"
        "- For mental health, emotions, or distress, call 'assign_to_RAG_agent'.\n"
        "- For all other topics, call 'assign_to_general_agent'.\n"
        "- if the user respond or ask follow up question where you think there is no need to fetch anytin from databse then do not call assign_to_RAG_agent , always call in this scenerio assign_to_general_agent\n"
        "- If the user explicitly wants to end or no further action is needed, respond indicating conversation is ending.\n"
        "Call one tool at a time and do not perform any work yourself."
    )
)

# --- ROUTING LOGIC ---
def route_from_supervisor(state: MessagesState) -> str:
    messages: List[BaseMessage] = state["messages"]
    last_message = messages[-1]
    tool_call = None

    # Check for tool calls directly from the state
    if "tool_calls" in state and state["tool_calls"]:
        tool_call = state["tool_calls"][0]["name"]  # Get tool name

    if tool_call == "assign_to_RAG_agent":
        return "RAG_agent"
    elif tool_call == "assign_to_general_agent":
        return "general_agent"
    elif tool_call == "end_conversation":
        return "end"

    return "end"

# --- GRAPH SETUP ---
supervisor_graph = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor_agent)
    .add_node("RAG_agent", call_qa_and_extract_note)
    .add_node("general_agent", general_agent)
    .add_node("psychiatrist_agent", psychiatrist_agent)

    # Start at supervisor
    .add_edge(START, "supervisor")

    # Supervisor can only send to RAG or general
    .add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "RAG_agent": "RAG_agent",
            "general_agent": "general_agent",
           
            "end": END
        }
    )

    # RAG and General always go to psychiatrist_agent next
    .add_edge("RAG_agent", "psychiatrist_agent")
    .add_edge("general_agent", "psychiatrist_agent")

    # Psychiatrist is always last
    .add_edge("psychiatrist_agent", END)

    .compile(checkpointer=checkpointer)
)


# # --- DISPLAY GRAPH ---
# try:
#     from IPython.display import display, Image
#     display(Image(supervisor_graph.get_graph().draw_mermaid_png()))
# except:
#     pass  # for non-notebook environments



def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")




conversation_id = "my-test-conversation-abcd"

# The configuration that tells LangGraph which conversation to continue.
config = {"configurable": {"thread_id": conversation_id}}

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Ending conversation.")
            break

        # This is the input for the stream. It's just the new message.
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # Stream the results, passing the config.
        # LangGraph will automatically load the history for this thread_id.
        final_chunk = None
        print("\nAssistant:")
        for chunk in supervisor_graph.stream(inputs, config=config):
            # Determine which agent produced the chunk and print the output accordingly
            if "supervisor" in chunk:
                print("Supervisor Agent Output:")
                # Check if 'tool_calls' key exists and has values
                if 'tool_calls' in chunk['supervisor'] and chunk['supervisor']['tool_calls']:
                    supervisor_output = chunk['supervisor']['tool_calls'][0]["function"]["__wrapped__.__name__"]
                    print(supervisor_output)
                    if "transfer_to_RAG_agent" in supervisor_output:
                        print("Routing to RAG Agent...\n")
                    elif "transfer_to_general_agent" in supervisor_output:
                        print("Routing to General Agent...\n")
                    elif "end_conversation" in supervisor_output:
                        print("Ending Conversation")
                else:
                    print("No tool calls found in supervisor output.")
                print("-" * 30)
            elif "RAG_agent" in chunk:
                print("RAG Agent Output:")
                # Extracting output to determine whether it's working.
                print(chunk['RAG_agent'])
                print("-" * 30)


            elif "general_agent" in chunk:
                print("General Agent Output:")
                # Extracting output to determine whether it's working.
                print(chunk['general_agent'])
                print("-" * 30)

            elif "psychiatrist_agent" in chunk:
                print("Psychiatrist Agent Output:")
                # Get the last message from the list in the chunk
                final_message = chunk['psychiatrist_agent']['messages'][-1]
                print(final_message.content, end="", flush=True)
                print("\n" + "=" * 50) #Formatting

            final_chunk = chunk # Keep track of the last chunk to get the final state


    except KeyboardInterrupt:
        print("\nEnding conversation.")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break
















# import warnings
# # Suppress deeplake version check warning at the start
# warnings.filterwarnings(
#     "ignore",
#     message="A newer version of deeplake.*",
#     category=UserWarning,
#     module=r"deeplake\.util\.check_latest_version"
# )

# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.messages import HumanMessage
# from langchain_core.retrievers import BaseRetriever
# from utils.nodes import create_qa_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.tools import create_handoff_tool
# from utils.tools import assign_to_RAG_agent
# from langgraph.prebuilt import create_react_agent
# from utils.nodes import llm

# checkpointer = InMemorySaver()
# qa_agent = create_qa_agent()

# # Psychiatrist Agent
# psychiatrist_agent = create_react_agent(
#     model=llm,
#     tools=[],
#     prompt=(
#         "You are an expert psychiatrist with extensive experience in mental health support. Your role is to receive responses from a RAG agent, which provides evidence-based information, and reframe them into empathetic, professional, and actionable advice for the user. Use a compassionate tone, validate the user's feelings, and provide clear, practical steps to address their concerns. Ensure your response is concise, supportive, and aligns with best practices in psychiatry. Do not include disclaimers such as 'I am not a mental health professional' or similar qualifiers, as you are positioned as an expert. Do not use any tools; only process the input messages and generate a final response."
#     ),
#     name="psychiatrist",
# )

# # General Assistant Agent
# general_assistant = create_react_agent(
#     model=llm,
#     tools=[],
#     prompt=(
#         "You are a general assistant tasked with answering general queries that are not related to mental health topics such as anxiety, stress, depression, or loneliness. Provide accurate, concise, and helpful responses to the user's questions. Do not use any tools; only process the input messages and generate a final response."
#     ),
#     name="general_assistant",
# )

# # Supervisor Agent
# supervisor_agent = create_react_agent(
#     model=llm,
#     tools=[assign_to_RAG_agent],
#     prompt=(
#         "You are a supervisor managing a RAG agent, a Psychiatrist agent, and a General Assistant agent.\n"
#         "Analyze the user query to determine if it is related to mental health topics such as anxiety, stress, depression, or loneliness by checking for relevant keywords.\n"
#         "If the query contains keywords like 'anxiety', 'stress', 'depression', 'lonely', or 'loneliness', assign the task to the RAG agent using the 'transfer_to_RAG_agent' tool.\n"
#         "For all other queries, route the query directly to the General Assistant agent without invoking any tools.\n"
#         "After the RAG agent responds, its response will be forwarded to the Psychiatrist agent for final processing.\n"
#         "After the General Assistant responds, no further actions are needed, and the response should be final.\n"
#         "Do not attempt to answer the query yourself or invoke tools unnecessarily.\n"
#         "Log your routing decision for debugging purposes."
#     ),
#     name="supervisor",
# )

# # Define the StateGraph
# def route_supervisor(state):
#     query = state["messages"][-1].content.lower()
#     mental_health_keywords = ["anxiety", "stress", "depression", "lonely", "loneliness"]
#     if any(keyword in query for keyword in mental_health_keywords):
#         print(f"DEBUG: Routing to RAG_agent for mental health query: {query}")
#         return "RAG_agent"
#     print(f"DEBUG: Routing to general_assistant for general query: {query}")
#     return "general_assistant"

# supervisor = (
#     StateGraph(MessagesState)
#     .add_node("supervisor", supervisor_agent)
#     .add_node("RAG_agent", qa_agent)
#     .add_node("psychiatrist", psychiatrist_agent)
#     .add_node("general_assistant", general_assistant)
#     .add_edge(START, "supervisor")
#     .add_conditional_edges(
#         "supervisor",
#         route_supervisor,
#         {
#             "RAG_agent": "RAG_agent",
#             "general_assistant": "general_assistant",
#         },
#     )
#     .add_edge("RAG_agent", "psychiatrist")
#     .add_edge("psychiatrist", END)
#     .add_edge("general_assistant", END)
#     .compile(checkpointer=checkpointer)
# )

# config = {"configurable": {"thread_id": "31", "recursion_limit": 50}}  # Increased recursion limit

# from langchain_core.messages import convert_to_messages

# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return
#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)

# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         if len(ns) == 0:
#             return
#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

# # Track if it's a fresh session
# session_started = False

# # Keep conversation going until user enters "quit"
# while True:
#     if not session_started:
#         query = input("Hi, how can I help you? (Type 'quit' to exit): ")
#         session_started = True
#     else:
#         query = input("How can I assist you further? (Type 'quit' to exit): ")

#     if query.lower() == "quit":
#         print("Goodbye!")
#         break

#     try:
#         for chunk in supervisor.stream(
#             {
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": f"{query}",
#                     }
#                 ]
#             },
#             config=config
#         ):
#             pretty_print_messages(chunk, last_message=True)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         continue

#     # Retrieve final state to get the final response
#     final_state = supervisor.get_state(config)
#     if final_state and final_state.values.get("messages"):
#         final_messages = convert_to_messages(final_state.values["messages"])
#         for msg in final_messages[-1:]:
#             if msg.type == "ai" and getattr(msg, "name", None) in ["psychiatrist", "general_assistant"]:
#                 print(f"\nFinal Response from {msg.name}:")
#                 print("\n")
#                 pretty_print_message(msg)
#                 break








############################3 final version v1 ####################################












# import warnings
# import uuid
# # Suppress deeplake version check warning at the start
# warnings.filterwarnings(
#     "ignore",
#     message="A newer version of deeplake.*",
#     category=UserWarning,
#     module=r"deeplake\.util\.check_latest_version"
# )

# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing import Literal, Optional
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.retrievers import BaseRetriever
# from utils.nodes import create_qa_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.tools import create_handoff_tool
# from utils.tools import assign_to_RAG_agent
# from langgraph.prebuilt import create_react_agent
# from utils.nodes import llm

# # Extend MessagesState to include user_name
# class ExtendedState(MessagesState):
#     user_name: Optional[str] = None

# checkpointer = InMemorySaver()
# qa_agent = create_qa_agent()

# # Psychiatrist Agent
# psychiatrist_agent = create_react_agent(
#     model=llm,
#     tools=[],
#     prompt=(
#         "You are an expert psychiatrist with extensive experience in mental health support. Your role is to receive responses from a RAG agent, which provides evidence-based information, and reframe them into empathetic, professional, and actionable advice for the user. If the user's name is available, address them by name for a personalized response. Use a compassionate tone, validate the user's feelings, and provide clear, practical steps to address their concerns. Ensure your response is concise, supportive, and aligns with best practices in psychiatry. Do not include disclaimers such as 'I am not a mental health professional' or similar qualifiers, as you are positioned as an expert. Do not use any tools; only process the input messages and generate a final response."
#     ),
#     name="psychiatrist",
# )

# # Supervisor Agent
# supervisor_agent = create_react_agent(
#     model=llm,
#     tools=[assign_to_RAG_agent],
#     prompt=(
#         "You are a supervisor managing a RAG agent and a Psychiatrist agent.\n"
#         "For any user query, assign the task to the RAG agent using the 'transfer_to_RAG_agent' tool.\n"
#         "After the RAG agent responds, the response will be forwarded to the Psychiatrist agent for final processing.\n"
#         "Do not invoke any tools or take further actions after receiving the Psychiatrist agent's response.\n"
#         "Do not attempt to answer the query yourself under any circumstances."
#     ),
#     name="supervisor",
# )

# # Welcome Agent to handle name collection
# welcome_agent = create_react_agent(
#     model=llm,
#     tools=[],
#     prompt=(
#         "You are a welcoming assistant. Your role is to check if the user's name is stored. "
#         "If no name is stored (user_name is None), respond with: 'Hello! May I have your name, please?' "
#         "If the name is provided, store it and respond with: 'Thank you, [name]! How can I help you today?' "
#         "If the name is already stored, respond with: 'Welcome back, [name]! How can I help you today?' "
#         "Do not process any other queries; forward them to the next agent."
#     ),
#     name="welcome",
# )

# # Define the StateGraph with ExtendedState
# def welcome_node(state: ExtendedState) -> ExtendedState:
#     messages = state["messages"]
#     user_name = state.get("user_name")
#     last_message = messages[-1] if messages else None
    
#     if not user_name:
#         if last_message and isinstance(last_message, HumanMessage):
#             # Assume the user provided their name
#             state["user_name"] = last_message.content.strip()
#             response = AIMessage(content=f"Thank you, {state['user_name']}! How can I help you today?", name="welcome")
#             state["messages"].append(response)
#         else:
#             # Ask for the name
#             response = AIMessage(content="Hello! May I have your name, please?", name="welcome")
#             state["messages"].append(response)
#     else:
#         # Name already exists, proceed
#         response = AIMessage(content=f"Welcome back, {user_name}! How can I help you today?", name="welcome")
#         state["messages"].append(response)
    
#     return state

# supervisor = (
#     StateGraph(ExtendedState)
#     .add_node("welcome", welcome_node)
#     .add_node("supervisor", supervisor_agent)
#     .add_node("RAG_agent", qa_agent)
#     .add_node("psychiatrist", psychiatrist_agent)
#     .add_edge(START, "welcome")
#     .add_edge("welcome", "supervisor")
#     .add_edge("supervisor", "RAG_agent")
#     .add_edge("RAG_agent", "psychiatrist")
#     .add_edge("psychiatrist", END)
#     .compile(checkpointer=checkpointer)
# )

# # Generate a unique thread_id for each session
# config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# from langchain_core.messages import convert_to_messages

# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return

#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)

# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         if len(ns) == 0:
#             return
#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

# # Keep conversation going until user enters "quit"
# while True:
#     # Check current state to determine if name is needed
#     current_state = supervisor.get_state(config)
#     user_name = current_state.values.get("user_name") if current_state else None
    
#     if not user_name:
#         # Ask for name first
#         query = input("Hello! May I have your name, please? (Type 'quit' to exit): ")
#         if query.lower() == "quit":
#             print("Goodbye!")
#             break
        
#         # Send name to welcome agent
#         try:
#             for chunk in supervisor.stream(
#                 {
#                     "messages": [
#                         HumanMessage(content=query)
#                     ],
#                     "user_name": None
#                 },
#                 config=config
#             ):
#                 pretty_print_messages(chunk, last_message=True)
#             continue  # Wait for next input ("How can I help you?")
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")
#             continue
    
#     # Name exists, ask how to help
#     query = input(f"Welcome back, {user_name}! How can I help you today? (Type 'quit' to exit): ")
#     if query.lower() == "quit":
#         print("Goodbye!")
#         break

#     try:
#         for chunk in supervisor.stream(
#             {
#                 "messages": [
#                     HumanMessage(content=query)
#                 ],
#                 "user_name": user_name
#             },
#             config=config
#         ):
#             pretty_print_messages(chunk, last_message=True)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         continue

#     # Retrieve final state to get psychiatrist's response
#     final_state = supervisor.get_state(config)
#     if final_state and final_state.values.get("messages"):
#         final_messages = convert_to_messages(final_state.values["messages"])
#         for msg in final_messages[-1:]:
#             if msg.type == "ai" and getattr(msg, "name", None) == "psychiatrist":
#                 print("\nFinal Response from Psychiatrist:")
#                 print("\n")
#                 pretty_print_message(msg)
#                 break









#################### final version ########################

# import warnings
# # Suppress deeplake version check warning at the start
# warnings.filterwarnings(
#     "ignore",
#     message="A newer version of deeplake.*",
#     category=UserWarning,
#     module=r"deeplake\.util\.check_latest_version"
# )

# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.messages import HumanMessage
# from langchain_core.retrievers import BaseRetriever
# from utils.nodes import create_qa_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.tools import create_handoff_tool
# from utils.tools import assign_to_RAG_agent
# from langgraph.prebuilt import create_react_agent
# from utils.nodes import llm

# checkpointer = InMemorySaver()
# qa_agent = create_qa_agent()

# # Psychiatrist Agent
# psychiatrist_agent = create_react_agent(
#     model=llm,
#     tools=[],
#     prompt=(
#         "You are an expert psychiatrist with extensive experience in mental health support. Your role is to receive responses from a RAG agent, which provides evidence-based information, and reframe them into empathetic, professional, and actionable advice for the user. Use a compassionate tone, validate the user's feelings, and provide clear, practical steps to address their concerns. Ensure your response is concise, supportive, and aligns with best practices in psychiatry. Do not include disclaimers such as 'I am not a mental health professional' or similar qualifiers, as you are positioned as an expert. Do not use any tools; only process the input messages and generate a final response."
#     ),
#     name="psychiatrist",
# )

# # Supervisor Agent
# supervisor_agent = create_react_agent(
#     model=llm,
#     tools=[assign_to_RAG_agent],
#     prompt=(
#         "You are a supervisor managing a RAG agent and a Psychiatrist agent.\n"
#         "For any user query, assign the task to the RAG agent using the 'transfer_to_RAG_agent' tool.\n"
#         "After the RAG agent responds, the response will be forwarded to the Psychiatrist agent for final processing.\n"
#         "Do not invoke any tools or take further actions after receiving the Psychiatrist agent's response.\n"
#         "Do not attempt to answer the query yourself under any circumstances."
#     ),
#     name="supervisor",
# )

# # Define the StateGraph
# supervisor = (
#     StateGraph(MessagesState)
#     .add_node("supervisor", supervisor_agent)
#     .add_node("RAG_agent", qa_agent)
#     .add_node("psychiatrist", psychiatrist_agent)
#     .add_edge(START, "supervisor")
#     .add_edge("supervisor", "RAG_agent")
#     .add_edge("RAG_agent", "psychiatrist")
#     .add_edge("psychiatrist", END)  # End after psychiatrist
#     .compile(checkpointer=checkpointer)
# )

# config = {"configurable": {"thread_id": "3"}}

# from langchain_core.messages import convert_to_messages

# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return

#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)

# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         if len(ns) == 0:
#             return
#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

# # Keep conversation going until user enters "quit"
# while True:
#     query = input("Hi, how can I help you? (Type 'quit' to exit): ")
#     if query.lower() == "quit":
#         print("Goodbye!")
#         break

#     try:
#         for chunk in supervisor.stream(
#             {
#                 "messages": [
#                     {
#                         "role": "user",
#                         "content": f"{query}",
#                     }
#                 ]
#             },
#             config=config
#         ):
#             pretty_print_messages(chunk, last_message=True)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         continue

#     # Retrieve final state to get psychiatrist's response
#     final_state = supervisor.get_state(config)
#     if final_state and final_state.values.get("messages"):
#         final_messages = convert_to_messages(final_state.values["messages"])
#         for msg in final_messages[-1:]:
#             if msg.type == "ai" and getattr(msg, "name", None) == "psychiatrist":
#                 print("\nFinal Response from Psychiatrist:")
#                 print("\n")
#                 pretty_print_message(msg)
#                 break












########################### v2 #######################################



# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.messages import HumanMessage
# from langchain_core.retrievers import BaseRetriever
# from utils.nodes import create_qa_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.tools import create_handoff_tool
# from utils.tools import assign_to_RAG_agent
# from langgraph.prebuilt import create_react_agent
# from utils.nodes import llm
# import warnings
# # Ignore all warnings
# warnings.filterwarnings("ignore")

# checkpointer = InMemorySaver()
# qa_agent = create_qa_agent()


# # Supervisor Agent
# supervisor_agent = create_react_agent(
#     model=llm,
#     tools=[assign_to_RAG_agent],
#     prompt=(
#         "You are a supervisor managing a RAG agent.\n"
#         "For any user query, you must assign the task to the RAG agent using the 'transfer_to_RAG_agent' tool.\n"
#         "Do not use any other tools or attempt to answer the query yourself under any circumstances."
#     ),
#     name="supervisor",
# )

# supervisor = (
#     StateGraph(MessagesState)
#     .add_node("supervisor", supervisor_agent)
#     .add_node("RAG_agent", qa_agent)
#     .add_edge(START, "supervisor")
#     .add_edge("RAG_agent", "supervisor")
#     .compile(checkpointer=checkpointer)
# )

# config = {"configurable": {"thread_id": "3"}}
# supervisor.get_state(config)

# from langchain_core.messages import convert_to_messages

# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return

#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)

# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         # skip parent graph updates in the printouts
#         if len(ns) == 0:
#             return

#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

# # Keep conversation going until user enters "quit"
# while True:
#     query = input("Hi, how can I help you? (Type 'quit' to exit): ")
#     if query.lower() == "quit":
#         print("Goodbye!")
#         break

#     for chunk in supervisor.stream(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"{query}",
#                 }
#             ]
#         },
#         config=config
#     ):
#         pretty_print_messages(chunk, last_message=True)

# final_message_history = chunk["supervisor"]["messages"]
















############## v1 ########################

# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain.chat_models import init_chat_model
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain.tools.retriever import create_retriever_tool
# from langchain_core.messages import HumanMessage
# from langchain_core.retrievers import BaseRetriever # Added import
# from utils.nodes import create_qa_agent
# from langgraph.checkpoint.memory import InMemorySaver
# from utils.tools import create_handoff_tool
# from utils.tools import assign_to_RAG_agent
# from langgraph.prebuilt import create_react_agent
# from utils.nodes import llm

# checkpointer = InMemorySaver()
# qa_agent = create_qa_agent()


# # Supervisor Agent
# supervisor_agent = create_react_agent(
#     model=llm,
#     tools=[assign_to_RAG_agent],  # Ensure no other tools like brave_search are included
#     prompt=(
#         "You are a supervisor managing a RAG agent.\n"
#         "For any user query, you must assign the task to the RAG agent using the 'transfer_to_RAG_agent' tool.\n"
#         "Do not use any other tools or attempt to answer the query yourself under any circumstances."
#     ),
#     name="supervisor",
# )

# supervisor = (
#     StateGraph(MessagesState)
#     .add_node("supervisor", supervisor_agent)
#     .add_node("RAG_agent", qa_agent)
#     .add_edge(START, "supervisor")
#     .add_edge("RAG_agent", "supervisor")
#     .compile(checkpointer=checkpointer)
# )

# config = {"configurable": {"thread_id": "3"}}
# supervisor.get_state(config)

# from langchain_core.messages import convert_to_messages


# def pretty_print_message(message, indent=False):
#     pretty_message = message.pretty_repr(html=True)
#     if not indent:
#         print(pretty_message)
#         return

#     indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
#     print(indented)


# def pretty_print_messages(update, last_message=False):
#     is_subgraph = False
#     if isinstance(update, tuple):
#         ns, update = update
#         # skip parent graph updates in the printouts
#         if len(ns) == 0:
#             return

#         graph_id = ns[-1].split(":")[0]
#         print(f"Update from subgraph {graph_id}:")
#         print("\n")
#         is_subgraph = True

#     for node_name, node_update in update.items():
#         update_label = f"Update from node {node_name}:"
#         if is_subgraph:
#             update_label = "\t" + update_label

#         print(update_label)
#         print("\n")

#         messages = convert_to_messages(node_update["messages"])
#         if last_message:
#             messages = messages[-1:]

#         for m in messages:
#             pretty_print_message(m, indent=is_subgraph)
#         print("\n")

# querry = input("hiii how can i help you: ")
# for chunk in supervisor.stream(
#   { 
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"{querry}",
#             }
#         ]
#     },
#   config=config
# ):
#     pretty_print_messages(chunk, last_message=True)

# final_message_history = chunk["supervisor"]["messages"]


