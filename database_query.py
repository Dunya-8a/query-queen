import pandas as pd

from chembl_webresource_client.new_client import new_client

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import tool


@tool("search_target_id", return_direct=False)
def search_target_id_tool(target_name: str) -> str:
    """Searches ChEMBL for the target ID of a given target name. Returns a dictionary of the target ID and the target name.
    The input to this tool should be a string of the target name. If the dictionary has length greather than one, prints the possible target IDs and ask the user to select one of the targets."""
    target = new_client.target
    res = target.filter(target_synonym__icontains=target_name).only(['target_chembl_id', 'pref_name'])
    if len(res) > 1:
        print(f"\nFound {len(res)} possible targets:")
        for i in res:
            print(i)
        user_prompt = ("\nPlease select one of the targets:")
        return input(user_prompt)
    elif len(res) == 1:
        return res[0]
    else:
        return "No targets found."


@tool("search_activities", return_direct=True)
def search_activities_for_target_id(target_id):
    """Searches ChEMBL for the activities of a given target ID. Returns a list of the activities."""
    activity = new_client.activity
    target_activities = activity.filter(target_chembl_id=target_id)
    return pd.DataFrame.from_dict(target_activities)


@tool("search_top_n_activities", return_direct=True)
def search_top_n_activities_for_target_id(target_id, top_n=10):
    """After finding the target ChEMBL ID, searches ChEMBL for the top N activities of the given target ID.
    
    Returns a list of the activities.
    """
    activity = new_client.activity
    target_activities = activity.filter(target_chembl_id=target_id)
    df = pd.DataFrame.from_dict(target_activities)
    df['pchembl_value'] = df['pchembl_value'].astype(float)
    df = df.dropna(subset=['pchembl_value'])
    return df.nlargest(top_n, 'pchembl_value')


def do_database_query(query: str):
    llm = ChatOpenAI(temperature=0.0, openai_api_key=OPENAI_API_KEY, openai_organization='org-zcFmIksFIyEPBZIOZ9kDKkRN')

    tools = load_tools(
        ["human", ],
    )

    agent_chain = initialize_agent(
        tools + [search_target_id_tool] + [search_activities_for_target_id] + [search_top_n_activities_for_target_id],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent_chain.run(query)
