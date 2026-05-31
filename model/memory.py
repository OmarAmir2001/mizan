import uuid
from trustcall import create_extractor
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from datetime import datetime
from dotenv import load_dotenv

TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 
Use the provided tools to retain any necessary memories about the user.
Use parallel tool calling to handle updates and insertions simultaneously.
System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction between a user and Mizan, an Egyptian legal assistant.
Based on this interaction, update the behavioral instructions for how Mizan should respond to this user.
Consider: response language, length preference, citation style, formality level, and any explicit feedback.
Your current instructions are:
<current_instructions>
{current_instructions}
</current_instructions>"""

# Load environment variables
load_dotenv()
#model/memory.py
model = ChatGroq(model="llama-3.3-70b-versatile")


class UserProfile(BaseModel):
    """Profile of the user interacting with Mizan legal assistant."""
    name: Optional[str] = Field(None, description="The user's name")
    profession: Optional[str] = Field(None, description="The user's profession")
    preferred_language: Optional[str] = Field(None, description="The user's preferred_language")
    last_topic: Optional[str] = Field(None, description="The user's last_topic")

class MizanInstructions(BaseModel):
    """Behavioral instructions for how Mizan should respond to this user."""
    instructions: str = Field(
        description="a string that contains all the behavioral rules accumulated over time" 
    )
# Create the Trustcall extractor
profile_extractor = create_extractor(
    model,           # the LLM to use for extraction
    tools=[UserProfile],   # the data structure we want to extract into
    tool_choice='UserProfile' # the name of the tool to use for extraction
)
instructions_extractor = create_extractor(
    model,
    tools=[MizanInstructions],
    tool_choice='MizanInstructions'
)

def load_profile(state, config: RunnableConfig, *, store: BaseStore):
    # 1. get user_id from config
    user_id = config["configurable"]["user_id"]

    # 2. define the namespace
    namespace = ("UserProfile", user_id)
    
    # 3. search the store
    memories = store.search(namespace)
    
    # 4. if memories exist, get the profile, else None
    profile = memories[0].value if memories else None
    
    # 5. return it so other nodes can use it
    return {"user_profile": profile}

def save_profile(state, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    # --- Save User Profile ---
    namespace_profile = ("UserProfile", user_id)
    existing_items = store.search(namespace_profile)
    existing = {item.key: item.value for item in existing_items}

    messages_profile = [
        SystemMessage(content=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())),
        HumanMessage(content=f"Query: {state['query']}"),
        HumanMessage(content=f"Answer given: {state.get('answer', '')}")
    ]

    result = profile_extractor.invoke({
        "messages": messages_profile,
        "existing": existing
    })

    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace_profile,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"))

    # --- Save Instructions ---
    namespace_instructions = ("MizanInstructions", user_id)
    current_inst_items = store.search(namespace_instructions)
    existing_inst = {item.key: item.value for item in current_inst_items}
    current_instructions = current_inst_items[0].value.get("instructions", "No instructions yet.") if current_inst_items else "No instructions yet."

    messages_inst = [
        SystemMessage(content=CREATE_INSTRUCTIONS.format(current_instructions=current_instructions)),
        HumanMessage(content=f"Query: {state['query']}"),
        HumanMessage(content=f"Answer given: {state.get('answer', '')}")
    ]

    result_inst = instructions_extractor.invoke({
        "messages": messages_inst,
        "existing": existing_inst
    })

    for r, rmeta in zip(result_inst["responses"], result_inst["response_metadata"]):
        store.put(namespace_instructions,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"))