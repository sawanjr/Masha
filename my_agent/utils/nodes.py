

import os
import warnings
import json
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
# We only need AIMessage and HumanMessage for this file now
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent # Added import
from typing import List, Dict
# --- IMPORTS FROM YOUR PROJECT ---
from utils.tools import db

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
api_key = "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"
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please set your API key.")

# llm is defined globally here, but we will pass it to nodes explicitly
llm= ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
)

# --- PTDANTIC MODELS ---
class ContextAnalysis(BaseModel):
    theme: str = Field(description="Main theme or cause of distress in the context (e.g., romantic betrayal, familial issues, anxiety)")
    input_output_relation: str = Field(description="Relationship between the key (input) and the corresponding value (output) retrived from vectorstore.")



# --- AGENT NODES (Refactored for MessagesState and explicit LLM) ---

def retrieve_context_node(state: MessagesState) -> MessagesState:
    print("\n--- Node: retrieve_context_node ---")
    question = state["messages"][0].content
    results = db.similarity_search(question, k=2) # UNCOMMENTED

    formatted_context = []
    if results:
        for i, doc in enumerate(results):
            input_text = doc.page_content
            output_text = doc.metadata.get('output', 'N/A')
            doc_string = f"Retrieved Document {i+1}:\nInput: {input_text}\nOutput: {output_text}"
            formatted_context.append(doc_string)
        final_context = "\n\n".join(formatted_context)
    else:
        final_context = "No relevant documents were found."

    print(f"Formatted Context:\n{final_context[:500]}...")
    context_message = AIMessage(content=final_context)
    return {"messages": state["messages"] + [context_message]}


def analyze_context_node(state: MessagesState, llm) -> MessagesState:
    print("\n--- RAG Agent Node: analyze_context ---")
    context = state["messages"][-1].content
    prompt = f"Analyze the retrieved document to understand its main theme and the relationship between its 'input' and 'output'. \nHere is the document: \n\n{context}\n\n"
    response = llm.with_structured_output(ContextAnalysis).invoke([{"role": "user", "content": prompt}])

    analysis_message = AIMessage(
        content=json.dumps(response.dict()),
        name="context_analysis"
    )
    return {"messages": state["messages"] + [analysis_message]}


def generate_answer_node(state: MessagesState, llm) -> MessagesState:
    print("\n--- RAG Agent Node: generate_answer ---")
    question = state["messages"][0].content
    context = state["messages"][-2].content

    prompt = f"""You are an empathetic counseling assistant your task is frame a note and explain the situation to the professional psychiatrist using the input and Output key from retrived context  the note format should be.
    'counseling_note':
    you response should not be like you are responding to user but to a senior and expert psychiatrist whom you are telling the patient condition.
    do not write in email format , just the content
    Use the retrieved context to inform your note. If the context is not relevant,
    ignore it and answer based on the question alone. Keep your answer supportive and concise (2-4 sentences). \n\nUser Question: {question}\n\nRetrieved Context: {context}"""


    response = llm.invoke([{"role": "user", "content": prompt}])
    # We add the final AI response to the message list
    # The key of the message needs to be "counseling_note".
    return {"messages": state["messages"] + [AIMessage(content=json.dumps({"counseling_note": response.content}))]}


def rewrite_question_node(state: MessagesState, llm) -> MessagesState:
    print("\n--- RAG Agent Node: rewrite_question ---")
    question = state["messages"][0].content
    prompt = f"A user is seeking help. Their initial question is vague: '{question}'. Re-formulate this as a clearer question to find helpful information. Only output the improved question."
    response = llm.invoke([{"role": "user", "content": prompt}])
    print(f"Rewritten Question: {response.content}")
    return {"messages": [HumanMessage(content=response.content)]}


# --- THE FINAL, COMPATIBLE AGENT ---
def create_qa_agent(llm):
    """Creates the compatible RAG agent graph that uses MessagesState."""
    workflow = StateGraph(MessagesState)

    workflow.add_node("retrieve", retrieve_context_node)
    # Pass llm to the nodes that need it
    workflow.add_node("analyze_context", lambda state: analyze_context_node(state, llm))
    workflow.add_node("rewrite_question", lambda state: rewrite_question_node(state, llm))
    workflow.add_node("generate_answer", lambda state: generate_answer_node(state, llm))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze_context")

    #Always generate answer, by removing the conditional edge and relevance checks.
    workflow.add_edge("analyze_context", "generate_answer")

    #removing rewrite and conditional
    #workflow.add_conditional_edges(
    #    "analyze_context",
    #    lambda state: grade_documents_node(state, llm),
    #    {
    #        "generate_answer": "generate_answer",
    #        "rewrite_question": "rewrite_question",
    #    }
    #)
    workflow.add_edge("generate_answer", END)
    #workflow.add_edge("rewrite_question", "retrieve")

    return workflow.compile()


def create_general_agent(llm):
    """Creates a simple, general-purpose agent."""
    # This uses a prebuilt basic agent from LangGraph.
    # It has no special tools and will answer from its own knowledge.
    return create_react_agent(llm, tools=[])

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

def create_psychiatrist_agent(llm):
    """Creates a psychiatrist agent that processes RAG_agent output as a custom node."""
    # Define the prompt for the psychiatrist agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a psychiatrist agent designed to provide empathetic, supportive, and evidence-based mental health assistance.\n"
            "Your role is to act as a compassionate, non-judgmental listener and guide. "
            "Your primary goal is to help users explore their thoughts, feelings, and challenges while promoting emotional well-being and self-awareness.\n"
            "Summarize or reflect on the user’s statements to show you understand their perspective. Ask open-ended, non-leading questions (e.g., 'Can you tell me more about what’s been going on?') to encourage exploration.\n"
            "**mandatory step**:if you ever recieve any councelling note , mention in your output sayning that you recived Counseling Note else mentioned your seceratory is unaviable and didnt provide you councelling note , please note very carefully , councelling note doesn't has exact situation to the user problem , but only similar problem , never ever mention the condition from councelling note to user , like death of parents , lost a match , playing golf etc... always focous on persons query.  \n"
        )),
        ("human", (
            "Here's the previous messages: {previous_messages}\n\n"
            "User Query: {user_query}\n\n"
            "Counseling Note: {counseling_note}\n\n"
        ))
    ])

    def psychiatrist_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Custom node to process MessagesState and generate a response."""
        messages: List[BaseMessage] = state["messages"]
        user_query = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
        counseling_note = state.get("counseling_note", state["messages"][-1].content)

        #Format Previous Messages for prompt
        previous_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]])

        formatted_prompt = prompt.format_prompt(
            user_query=user_query,
            counseling_note=counseling_note,
            previous_messages = previous_messages
        )

        response = llm.invoke(formatted_prompt)

        # Append the new message to the existing message history
        return {
            "messages": messages + [AIMessage(content=response.content)]
        }

    return psychiatrist_node

































#################### v3 ####################

# import warnings
# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode
# from pydantic import BaseModel, Field
# from typing import Literal, Optional
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_groq import ChatGroq
# from utils.tools import retriever_tool_councellor

# # Suppress all warnings
# warnings.filterwarnings("ignore")

# # Initialize the language model
# llm = ChatGroq(
#     api_key="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",
#     model="llama-3.1-8b-instant",
#     temperature=0.0,
#     max_retries=2,
# )

# response_model = llm
# grader_model = llm

# # Extend MessagesState to include user_name and context analysis
# class ExtendedState(MessagesState):
#     user_name: Optional[str] = None
#     context_analysis: Optional[dict] = None

# # Pydantic model for context analysis
# class ContextAnalysis(BaseModel):
#     theme: str = Field(description="Main theme or cause of distress in the context (e.g., romantic betrayal, familial issues)")
#     input_output_relation: str = Field(description="Relationship between input and output (e.g., emotional impact of input's cause)")

# # Welcome node
# def welcome_node(state: ExtendedState) -> ExtendedState:
#     messages = state["messages"]
#     user_name = state.get("user_name")
#     last_message = messages[-1] if messages else None

#     if not user_name:
#         if last_message and isinstance(last_message, HumanMessage):
#             state["user_name"] = last_message.content.strip()
#             response = AIMessage(
#                 content=f"Thank you, {state['user_name']}! How can I help you today?",
#                 name="welcome"
#             )
#             state["messages"] = [response]
#         else:
#             response = AIMessage(
#                 content="Hello! May I have your name, please?",
#                 name="welcome"
#             )
#             state["messages"] = [response]
#     else:
#         response = AIMessage(
#             content=f"Welcome, {user_name}! How can I help you today?",
#                 name="welcome"
#         )
#         state["messages"] = [response]
    
#     return state

# # Prompt for analyzing context
# ANALYZE_CONTEXT_PROMPT = (
#     "Analyze the retrieved document to understand the relationship between its 'input' and 'output'. \n"
#     "Here is the document: \n\n{context}\n\n"
#     "Identify the main theme or cause of distress (e.g., romantic betrayal, familial issues) and describe how the input (user's issue) relates to the output (suggested solution or insight)."
# )

# # Node to analyze retrieved context
# def analyze_context(state: ExtendedState) -> ExtendedState:
#     context = state["messages"][-1].content
#     prompt = ANALYZE_CONTEXT_PROMPT.format(context=context)
#     response = grader_model.with_structured_output(ContextAnalysis).invoke([{"role": "user", "content": prompt}])
#     state["context_analysis"] = response.dict()
#     return state

# # Grade documents prompt
# GRADE_PROMPT = (
#     "You are a grader assessing the relevance of a retrieved document to a user question. \n"
#     "Here is the user question: {question}\n"
#     "Here is the document's analysis: \n"
#     "- Theme: {theme}\n"
#     "- Input-Output Relation: {input_output_relation}\n"
#     "Compare the question's semantic intent to the document's theme and input-output relation. \n"
#     "Assign a relevance score from 0 to 100. A score ≥ 80 indicates high relevance."
# )

# class GradeDocuments(BaseModel):
#     relevance_score: float = Field(description="Relevance score from 0 to 100")

# # Grade documents node
# def grade_documents(state: ExtendedState) -> Literal["generate_answer", "rewrite_question"]:
#     question = state["messages"][0].content
#     context_analysis = state["context_analysis"]
#     prompt = GRADE_PROMPT.format(
#         question=question,
#         theme=context_analysis["theme"],
#         input_output_relation=context_analysis["input_output_relation"]
#     )
#     response = grader_model.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])
#     if response.relevance_score >= 80:
#         return "generate_answer"
#     return "rewrite_question"

# # Generate answer prompt
# GENERATE_PROMPT = (
#     "You are an expert assistant for question-answering tasks. "
#     "Use the retrieved context only if it aligns with the question's theme. "
#     "If the context is irrelevant, answer based on the question alone, addressing the user by name. "
#     "Keep the answer concise (max 3 sentences). \n"
#     "User Name: {user_name}\n"
#     "Question: {question}\n"
#     "Context Theme: {theme}\n"
#     "Context: {context}"
# )

# # Generate answer node
# def generate_answer(state: ExtendedState) -> ExtendedState:
#     question = state["messages"][0].content
#     context = state["messages"][-2].content  # Context is before analysis
#     context_analysis = state["context_analysis"]
#     user_name = state.get("user_name", "User")
#     prompt = GENERATE_PROMPT.format(
#         user_name=user_name,
#         question=question,
#         theme=context_analysis["theme"],
#         context=context
#     )
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": state["messages"] + [response]}

# # Rewrite question prompt
# REWRITE_PROMPT = (
#     "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
#     "Here is the initial question:\n\n{question}\n\n"
#     "Only output the improved question:"
# )

# # Rewrite question node
# def rewrite_question(state: ExtendedState) -> ExtendedState:
#     messages = state["messages"]
#     question = messages[0].content
#     prompt = REWRITE_PROMPT.format(question=question)
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": [{"role": "user", "content": response.content}]}

# # Create QA agent
# def create_qa_agent():
#     workflow = StateGraph(ExtendedState)

#     # Add nodes
#     workflow.add_node("welcome", welcome_node)
#     workflow.add_node("retrieve", ToolNode([retriever_tool_councellor], name="retrieve"))
#     workflow.add_node("analyze_context", analyze_context)
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)

#     # Add edges
#     workflow.add_edge(START, "welcome")
#     workflow.add_edge("welcome", "retrieve")
#     workflow.add_edge("retrieve", "analyze_context")
#     workflow.add_conditional_edges("analyze_context", grade_documents)
#     workflow.add_edge("generate_answer", END)
#     workflow.add_edge("rewrite_question", "retrieve")

#     return workflow.compile()

# # Run QA agent
# def run_qa_agent(question: str, user_name: Optional[str] = None) -> dict:
#     qa_agent = create_qa_agent()
#     initial_state = {
#         "messages": [HumanMessage(content=question)],
#         "user_name": user_name
#     }
#     result = qa_agent.invoke(initial_state)
#     return result







######################### V2 #########################

# import warnings
# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from pydantic import BaseModel, Field
# from typing import Literal, Optional
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.retrievers import BaseRetriever
# from langchain_groq import ChatGroq
# from utils.tools import retriever_tool_anxity

# # Suppress all warnings
# warnings.filterwarnings("ignore")

# # Initialize the language model
# llm = ChatGroq(
#     api_key="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",
#     model="llama-3.1-8b-instant",
#     temperature=0.0,
#     max_retries=2,
# )

# response_model = llm
# grader_model = llm

# # Extend MessagesState to include user_name
# class ExtendedState(MessagesState):
#     user_name: Optional[str] = None

# # Welcome node to handle name collection
# def welcome_node(state: ExtendedState) -> ExtendedState:
#     """Handle name collection for new sessions."""
#     messages = state["messages"]
#     user_name = state.get("user_name")
#     last_message = messages[-1] if messages else None

#     if not user_name:
#         if last_message and isinstance(last_message, HumanMessage):
#             # Assume the user provided their name
#             state["user_name"] = last_message.content.strip()
#             response = AIMessage(
#                 content=f"Thank you, {state['user_name']}! How can I help you today?",
#                 name="welcome"
#             )
#             state["messages"] = [response]  # Reset messages to avoid processing name as a query
#         else:
#             # Ask for the name
#             response = AIMessage(
#                 content="Hello! May I have your name, please?",
#                 name="welcome"
#             )
#             state["messages"] = [response]
#     else:
#         # Name already exists, proceed
#         response = AIMessage(
#             content=f"Welcome, {user_name}! How can I help you today?",
#             name="welcome"
#         )
#         state["messages"] = [response]
    
#     return state

# # Grade documents prompt
# GRADE_PROMPT = (
#     "You are a grader assessing relevance of a retrieved document to a user question. \n"
#     "Here is the retrieved document: \n\n{context}\n\n"
#     "Here is the user question: {question}\n"
#     "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
#     "Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."
# )

# class GradeDocuments(BaseModel):
#     """Grade documents using a binary score for relevance check."""
#     binary_score: str = Field(
#         description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
#     )

# # Rewrite question prompt
# REWRITE_PROMPT = (
#     "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
#     "Here is the initial question:\n\n{question}\n\n"
#     "Only output the improved question:"
# )

# # Generate answer prompt
# GENERATE_PROMPT = (
#     "You are an expert assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question. "
#     "Address the user by name if available. "
#     "If you don't know the answer, just say that you don't know. "
#     "Use three sentences maximum and keep the answer concise.\n"
#     "User Name: {user_name}\n"
#     "Question: {question}\n"
#     "Context: {context}"
# )

# def generate_query_or_respond(state: ExtendedState) -> ExtendedState:
#     """Call the model to generate a response or decide to retrieve based on the current state."""
#     response = (
#         response_model
#         .bind_tools([retriever_tool_anxity])
#         .invoke(state["messages"])
#     )
#     return {"messages": state["messages"] + [response]}

# def grade_documents(state: ExtendedState) -> Literal["generate_answer", "rewrite_question"]:
#     """Determine whether the retrieved documents are relevant to the question."""
#     question = state["messages"][0].content
#     context = state["messages"][-1].content
#     prompt = GRADE_PROMPT.format(question=question, context=context)
#     response = (
#         grader_model
#         .with_structured_output(GradeDocuments)
#         .invoke([{"role": "user", "content": prompt}])
#     )
#     score = response.binary_score
#     if score == "yes":
#         return "generate_answer"
#     else:
#         return "rewrite_question"

# def rewrite_question(state: ExtendedState) -> ExtendedState:
#     """Rewrite the original user question."""
#     messages = state["messages"]
#     question = messages[0].content
#     prompt = REWRITE_PROMPT.format(question=question)
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": [{"role": "user", "content": response.content}]}

# def generate_answer(state: ExtendedState) -> ExtendedState:
#     """Generate an answer using the user’s name if available."""
#     question = state["messages"][0].content
#     context = state["messages"][-1].content
#     user_name = state.get("user_name", "User")
#     prompt = GENERATE_PROMPT.format(
#         user_name=user_name if user_name else "User",
#         question=question,
#         context=context
#     )
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": state["messages"] + [response]}

# def create_qa_agent():
#     """Create a question-answering agent with name collection."""
#     workflow = StateGraph(ExtendedState)

#     # Add nodes
#     workflow.add_node("welcome", welcome_node)
#     workflow.add_node("generate_query_or_respond", generate_query_or_respond)
#     workflow.add_node("retrieve", ToolNode([retriever_tool_anxity], name="retrieve"))
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)

#     # Add edges
#     workflow.add_edge(START, "welcome")
#     workflow.add_edge("welcome", "generate_query_or_respond")
#     workflow.add_conditional_edges(
#         "generate_query_or_respond",
#         tools_condition,
#         {"tools": "retrieve", END: END},
#     )
#     workflow.add_conditional_edges("retrieve", grade_documents)
#     workflow.add_edge("generate_answer", END)
#     workflow.add_edge("rewrite_question", "generate_query_or_respond")

#     # Compile the graph
#     return workflow.compile()

# # Function to run the agent (for handoff or standalone use)
# def run_qa_agent(question: str, user_name: Optional[str] = None) -> dict:
#     """Run the QA agent with a user question and optional user name."""
#     qa_agent = create_qa_agent()
#     initial_state = {
#         "messages": [HumanMessage(content=question)],
#         "user_name": user_name
#     }
#     result = qa_agent.invoke(initial_state)
#     return result








######################## V1 ########################




# from langgraph.graph import MessagesState, StateGraph, START, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from pydantic import BaseModel, Field
# from typing import Literal
# from langchain_core.messages import HumanMessage
# from langchain_core.retrievers import BaseRetriever # Added import
# from langchain_groq import ChatGroq
# from utils.tools import retriever_tool_anxity 
# import warnings
# # Ignore all warnings
# warnings.filterwarnings("ignore")

# llm = ChatGroq(
#     api_key  =  "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",
#     model="llama-3.1-8b-instant",
#     temperature=0.0,
#     max_retries=2,
#     # other params...
# )

# response_model = llm  # Assumes llm is defined elsewhere

# def generate_query_or_respond(state: MessagesState):
#     """Call the model to generate a response based on the current state. Given
#     the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
#     """
#     response = (
#         response_model
#         .bind_tools([retriever_tool_anxity]).invoke(state["messages"]) # Removed duplicate create_retriever_tool
#     )
#     return {"messages": [response]}

# GRADE_PROMPT = (
#     "You are a grader assessing relevance of a retrieved document to a user question. \n "
#     "Here is the retrieved document: \n\n {context} \n\n"
#     "Here is the user question: {question} \n"
#     "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
#     "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
# )

# class GradeDocuments(BaseModel):
#     """Grade documents using a binary score for relevance check."""
#     binary_score: str = Field(
#         description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
#     )

# grader_model = llm  # Assumes llm is defined elsewhere

# def grade_documents(
#     state: MessagesState,
# ) -> Literal["generate_answer", "rewrite_question"]:
#     """Determine whether the retrieved documents are relevant to the question."""
#     question = state["messages"][0].content
#     context = state["messages"][-1].content
#     prompt = GRADE_PROMPT.format(question=question, context=context)
#     response = (
#         grader_model
#         .with_structured_output(GradeDocuments).invoke(
#             [{"role": "user", "content": prompt}]
#         )
#     )
#     score = response.binary_score
#     if score == "yes":
#         return "generate_answer"
#     else:
#         return "rewrite_question"

# REWRITE_PROMPT = (
#     "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
#     "Here is the initial question:"
#     "\n ------- \n"
#     "{question}"
#     "\n ------- \n"
#     "only and only output the improved question:"
# )

# def rewrite_question(state: MessagesState):
#     """Rewrite the original user question."""
#     messages = state["messages"]
#     question = messages[0].content
#     prompt = REWRITE_PROMPT.format(question=question)
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": [{"role": "user", "content": response.content}]}

# GENERATE_PROMPT = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer the question. "
#     "If you don't know the answer, just say that you don't know. "
#     "Use three sentences maximum and keep the answer concise.\n"
#     "Question: {question} \n"
#     "Context: {context}"
# )

# def generate_answer(state: MessagesState):
#     """Generate an answer."""
#     question = state["messages"][0].content
#     context = state["messages"][-1].content
#     prompt = GENERATE_PROMPT.format(question=question, context=context)
#     response = response_model.invoke([{"role": "user", "content": prompt}])
#     return {"messages": [response]}


# def create_qa_agent():
#     """Create a question-answering agent that can be called by other agents."""
#     workflow = StateGraph(MessagesState)

#     # Add nodes
#     workflow.add_node("generate_query_or_respond", generate_query_or_respond)
#     workflow.add_node("retrieve", ToolNode([retriever_tool_anxity], name="retrieve"))
#     workflow.add_node("rewrite_question", rewrite_question)
#     workflow.add_node("generate_answer", generate_answer)

#     # Add edges
#     workflow.add_edge(START, "generate_query_or_respond")
#     workflow.add_conditional_edges(
#         "generate_query_or_respond",
#         tools_condition,
#         {"tools": "retrieve", END: END},
#     )
#     workflow.add_conditional_edges("retrieve", grade_documents)
#     workflow.add_edge("generate_answer", END)
#     workflow.add_edge("rewrite_question", "generate_query_or_respond")

#     # Compile the graph
#     return workflow.compile()

# # Function to run the agent (for handoff or standalone use)
# def run_qa_agent(question: str) -> dict:
#     """Run the QA agent with a user question and return the result."""
#     qa_agent = create_qa_agent()
#     initial_state = {"messages": [HumanMessage(content=question)]}
#     result = qa_agent.invoke(initial_state)
#     return result





