from langchain.tools.retriever import create_retriever_tool
import os
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent # Added import

# Ignore all warnings
warnings.filterwarnings("ignore")

# --- REAL RETRIEVER SETUP ---
# It's better practice to load this from an environment variable if possible
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0ODUzODYzMiwiZXhwIjoxNzgwMDc0NjEwfQ.eyJpZCI6InNhd2Fua3VtYXJhcHNnIiwib3JnX2lkIjoic2F3YW5rdW1hcmFwc2cifQ.HL3TP72dEkDYCGYkt1xI3LIOZEPVsmg0HkmfPnNX7VnjGkrgdFvQoDeyX8Wi9LEfi6Q4xjwZuCFy08DBr_aikA"

print("Initializing embeddings and Deep Lake retriever...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = DeepLake(read_only=True, dataset_path='hub://sawankumarapsg/vector_store_mental_health', embedding=embeddings)
# retriever_councellor = db.as_retriever(search_kwargs={"k": 2})

# retriever_tool_councellor = create_retriever_tool(
#     retriever_councellor,
#     "retriever_councellor",
#     "Search and return information about conversation between user and councellor",
# )
# print("Retriever initialized successfully.")


# --- HANDOFF TOOLS (Defined but not used by the current QA agent) ---
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool

assign_to_RAG_agent = create_handoff_tool(
    agent_name="RAG_agent",
    description="Assign task to a RAG agent for fetching data related mental health condition.",
)
assign_to_general_agent = create_handoff_tool(
    agent_name="general_agent",
    description="Assign task to a general purpose agent for any query NOT related to mental health, such as general knowledge, trivia, or casual conversation.",
)
assign_to_psychiatrist_agent = create_handoff_tool(
    agent_name="psychiatrist_agent",
    description="Assign the task of reviewing and finalizing a response to the expert psychiatrist agent. This should be the final step after a RAG or general agent has provided an initial answer."
)












































# from langchain.tools.retriever import create_retriever_tool
# import os
# from langchain_community.vectorstores import DeepLake
# from langchain_huggingface import HuggingFaceEmbeddings
# import warnings
# from typing import Annotated
# from langchain_core.tools import tool, InjectedToolCallId
# from langgraph.prebuilt import InjectedState
# from langgraph.graph import StateGraph, START, MessagesState
# from langgraph.types import Command

# # Ignore all warnings
# warnings.filterwarnings("ignore")

# # --- REAL RETRIEVER SETUP ---
# # It's better practice to load this from an environment variable if possible
# os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0ODUzODYzMiwiZXhwIjoxNzgwMDc0NjEwfQ.eyJpZCI6InNhd2Fua3VtYXJhcHNnIiwib3JnX2lkIjoic2F3YW5rdW1hcmFwc2cifQ.HL3TP72dEkDYCGYkt1xI3LIOZEPVsmg0HkmfPnNX7VnjGkrgdFvQoDeyX8Wi9LEfi6Q4xjwZuCFy08DBr_aikA"

# print("Initializing embeddings and Deep Lake retriever...")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# db = DeepLake(read_only=True, dataset_path='hub://sawankumarapsg/vector_store_mental_health', embedding=embeddings)
# retriever_councellor = db.as_retriever(search_kwargs={"k": 2})

# retriever_tool_councellor = create_retriever_tool(
#     retriever_councellor,
#     "retriever_councellor",
#     "Search and return information about conversation between user and councellor",
# )
# print("Retriever initialized successfully.")


# # --- HANDOFF TOOLS (Defined but not used by the current QA agent) ---
# def create_handoff_tool(*, agent_name: str, description: str | None = None):
#     name = f"transfer_to_{agent_name}"
#     description = description or f"Ask {agent_name} for help."

#     @tool(name, description=description)
#     def handoff_tool(
#         state: Annotated[MessagesState, InjectedState],
#         tool_call_id: Annotated[str, InjectedToolCallId],
#     ) -> Command:
#         tool_message = {
#             "role": "tool",
#             "content": f"Successfully transferred to {agent_name}",
#             "name": name,
#             "tool_call_id": tool_call_id,
#         }
#         return Command(
#             goto=agent_name,
#             update={**state, "messages": state["messages"] + [tool_message]},
#             graph=Command.PARENT,
#         )

#     return handoff_tool

# assign_to_RAG_agent = create_handoff_tool(
#     agent_name="RAG_agent",
#     description="Assign task to a RAG agent for fetching data related mental health condition.",
# )
# assign_to_research_agent = create_handoff_tool(
#     agent_name="research_agent",
#     description="Assign task to a researcher agent.",
# )





















# # from langchain.tools.retriever import create_retriever_tool
# # import os
# # from langchain_community.vectorstores import DeepLake
# # from langchain_huggingface import HuggingFaceEmbeddings
# # import warnings
# # # Ignore all warnings
# # warnings.filterwarnings("ignore")


# # os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTc0ODUzODYzMiwiZXhwIjoxNzgwMDc0NjEwfQ.eyJpZCI6InNhd2Fua3VtYXJhcHNnIiwib3JnX2lkIjoic2F3YW5rdW1hcmFwc2cifQ.HL3TP72dEkDYCGYkt1xI3LIOZEPVsmg0HkmfPnNX7VnjGkrgdFvQoDeyX8Wi9LEfi6Q4xjwZuCFy08DBr_aikA"
# # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# # db = DeepLake(read_only=True, dataset_path='hub://sawankumarapsg/vector_store_mental_health', embedding=embeddings)
# # retriever_councellor = db.as_retriever(search_kwargs={"k": 2})

# # from langchain.tools.retriever import create_retriever_tool

# # retriever_tool_councellor = create_retriever_tool(
# #     retriever_councellor,
# #     "retriever_councellor",  
# #     "Search and return information about conversation between user and councellor",
# # )


# # ### creating handoff

# # from typing import Annotated
# # from langchain_core.tools import tool, InjectedToolCallId
# # from langgraph.prebuilt import InjectedState
# # from langgraph.graph import StateGraph, START, MessagesState
# # from langgraph.types import Command


# # def create_handoff_tool(*, agent_name: str, description: str | None = None):
# #     name = f"transfer_to_{agent_name}"
# #     description = description or f"Ask {agent_name} for help."

# #     @tool(name, description=description)
# #     def handoff_tool(
# #         state: Annotated[MessagesState, InjectedState],
# #         tool_call_id: Annotated[str, InjectedToolCallId],
# #     ) -> Command:
# #         tool_message = {
# #             "role": "tool",
# #             "content": f"Successfully transferred to {agent_name}",
# #             "name": name,
# #             "tool_call_id": tool_call_id,
# #         }
# #         return Command(
# #             goto=agent_name,
# #             update={**state, "messages": state["messages"] + [tool_message]},
# #             graph=Command.PARENT,
# #         )

# #     return handoff_tool


# # # Handoffs
# # assign_to_RAG_agent = create_handoff_tool(
# #     agent_name="RAG_agent",
# #     description="Assign task to a RAG agent for fetching data related mental health condition.",
# # )
# # assign_to_research_agent = create_handoff_tool(
# #     agent_name="research_agent",
# #     description="Assign task to a researcher agent.",
# # )
