import logging
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


# Set the logging level
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def get_trajectory_from_neo4j(driver, traj: int):
    """
    Example of a docstring.

    Args:
        a (int): The first number.
        traj (int): The trajectory number. As ID. They are unique.

    Returns:
        int: The product of a and b.
    """
    # traj is like an id
    
    query_patient_timestep = """
    MATCH (p:Patient {traj:  $traj})-[has_timestep]->(ts:TimeStep)
    RETURN p, ts
    LIMIT 20;
    """
    
    query_actions = """
    MATCH (p:Patient {traj: $traj})-[]->(:TimeStep)-[action]->(:TimeStep)
    RETURN action
    LIMIT 20;
    """
    
    records, summary, keys = driver.execute_query(
        query_patient_timestep, traj=traj,
        
    )
    logging.debug(f'{len(records)=}')
    patient = records[0].data()['p']
    time_steps = [record.data()['ts'] for record in records]
    # for record in records:
    #     print(record.data())
    
    records, summary, keys = driver.execute_query(
        query_actions, traj=traj,
        
    )
    actions = [record["action"]._properties for record in records]

    return patient, time_steps, actions

class Neo4jConnection:
    def __init__(self):
        load_dotenv()
        self.neo4j_url = os.environ.get("NEO4J_URI")
        self.neo4j_username = os.environ.get("NEO4J_USERNAME")
        self.neo4j_password = os.environ.get("NEO4J_PASSWORD")
    
        # Initialize the Neo4j driver
        self.driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_password))

    def get_driver(self):
        return self.driver