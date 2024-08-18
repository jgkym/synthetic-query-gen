import random
from tqdm import tqdm
import dspy
from typing import List
from pydantic import BaseModel, Field


class SyntheticGeneratorInput(BaseModel):
    document: str = Field()
    examples: List[str] = Field(default_factory=list)


class SyntheticGeneratorOutput(BaseModel):
    queries: List[str] = Field(
        default_factory=list, description="Must be written in Korean"
    )


class SyntheticGeneratorSignature(dspy.Signature):
    """Generate three synthetic queries based on the provided document, ensuring they resemble the example queries."""

    input: SyntheticGeneratorInput = dspy.InputField()
    output: SyntheticGeneratorOutput = dspy.OutputField()


class SyntheticGenerator(dspy.Module):
    def __init__(self, qp_map: dict):
        super().__init__()
        self.qp_map = qp_map
        self.synthetic_generator = dspy.TypedPredictor(SyntheticGeneratorSignature)

    def generate_queries(self, example: dspy.Example) -> SyntheticGeneratorOutput:
        """
        Generates synthetic queries for a given example.
        """
        sampled_examples = random.sample(self.qp_map[example.source], 5)
        doc_examples_pair = SyntheticGeneratorInput(
            document=example.content,
            queries=sampled_examples,
        )
        return self.synthetic_generator(input=doc_examples_pair)


def generate_synthetic_queries(
    generator: SyntheticGenerator, examples: List[dspy.Example]
) -> List[tuple]:
    """
    Generates synthetic queries for a list of examples.
    """
    qp_pairs = []
    for example in tqdm(examples, desc="Generating Synthetic Queries"):
        try:
            pred = generator.generate_queries(example)
            for query in pred.output.queries:
                qp_pairs.append((query, example.content))
        except ValueError:
            print(
                f"ValueError encountered for document: {example.content}. Skipping this item."
            )
            continue

    return qp_pairs
