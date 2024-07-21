from llm import LLM  # Import the LLM class
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RegexParser

class FNExampleGenerator(LLM):  # Inherit from LLM
    """Represents a zero-shot chain-of-thought implementing an intension for triples."""

    PROMPT_TEMPLATE = """
Consider the following knowledge graph triple: 
Subject: <{s}>
Predicate: <{p}>
Object: <{o}>

Given a hypothetical world where the following is true:
{graph}

This triple was assigned an incorrect truth value of false
with the following rationale:
{rationale}

Let's think step by step. Determine why the assignment was 
incorrect, and then provide a correct revised rationale for 
why the truth value of the triple is true.
###
Revision: {{revision}}
"""

    PROMPT_TEMPLATE = """
Consider the following knowledge graph triple: 
Subject: <{s}>
Predicate: <{p}>
Object: <{o}>

Given a hypothetical world where the following is true:
{graph}

This triple was assigned an incorrect truth value of false
with the following rationale:
{rationale}

Generate a new rationale explaining why the truth value 
of the triple is true.
###
Rationale: {{revision}}
"""

    PROMPT = PromptTemplate(input_variables=["s", "p", "o", "graph", "rationale"], template=PROMPT_TEMPLATE)

    OUTPUT_PARSER = RegexParser(
        regex=r"(?is).*Rationale:\**\s*(.*?)",
        output_keys=["revision"],
        default_output_key="revision"
    )
    
    def __init__(self, model="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes an intension-as-classifier.
        
        Parameters:
            model: The name of the model to be used for zero shot CoT classification (default "gpt-4-0125-preview").
            temperature: The temperature parameter for the model (default 0.1).
         """
        super().__init__(self.PROMPT, self.OUTPUT_PARSER, model, temperature)
