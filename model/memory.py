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

# Create the Trustcall extractor
profile_extractor = create_extractor(
    model,           # the LLM to use for extraction
    tools=[UserProfile],   # the data structure we want to extract into
    tool_choice='UserProfile' # the name of the tool to use for extraction
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
    # 1. get user_id from config
    user_id = config["configurable"]["user_id"]

    # 2. define the namespace — same as load_profile
    namespace = ("UserProfile", user_id)

    # 3. get existing profile to pass to Trustcall
    existing_items = store.search(namespace)
    existing = {item.key: item.value for item in existing_items}
    
    # 4. get the conversation to extract from
    # hint: what's in state that has the query and answer?
    messages = [
        SystemMessage(content=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())),
        HumanMessage(content=f"Query: {state['query']}"),
        HumanMessage(content=f"Answer given: {state.get('answer', '')}")
    ]
    
    # 5. invoke Trustcall — fill in the two inputs
    result = profile_extractor.invoke({
        "messages": messages,
        "existing": existing
    })
    
    # 6. save each result back to store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
              rmeta.get("json_doc_id", str(uuid.uuid4())),
              r.model_dump(mode="json"),
    )