from llm import LLM  # Import the LLM class
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RegexParser

class Intension(LLM):  # Inherit from LLM
    """Represents a zero-shot chain-of-thought implementing an intension for triples."""

    PROMPT_TEMPLATE = """
Determine the truth value of following knowledge graph triple 
in a hypothetical world where the following is true:
{graph}

Let's think step by step. Provide a rationale for 
your decision, then based on that rationale,
provide an answer of 1 if true, otherwise 
provide an answer of 0.
###
Subject: <{s}>
Predicate: <{p}>
Object: <{o}>
Rationale: {{rationale}}
Answer: {{answer}}
"""

    PROMPT = PromptTemplate(input_variables=["s", "p", "o", "graph"], template=PROMPT_TEMPLATE)

    OUTPUT_PARSER = RegexParser(
        regex=r"(?is).*Rationale:\**\s*(.*?)Answer:\**\s*(0|1)",
        output_keys=["rationale", "answer"],
        default_output_key="rationale"
    )
    
    def __init__(self, model="gpt-4-0125-preview", temperature=0.1):
        """
        Initializes an intension-as-classifier.
        
        Parameters:
            model: The name of the model to be used for zero shot CoT classification (default "gpt-4-0125-preview").
            temperature: The temperature parameter for the model (default 0.1).
         """
        super().__init__(self.PROMPT, self.OUTPUT_PARSER, model, temperature)
