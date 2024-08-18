from typing import List
import json
import dspy
from omegaconf import OmegaConf
from modules.synthetic_generator import SyntheticGenerator, generate_synthetic_queries
from modules.setup_llm import configure_llama_cpp_model
from modules.prepare_data import map_questions_to_sources, corpus_to_examples

def save_to_json(data: List[tuple], output_file: str):
    """
    Save data to a JSON file.

    Parameters:
    - data: List of tuples containing the data to save.
    - output_file: The file path where the JSON data will be saved.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {output_file}")

def load_and_prepare_data(config):
    """
    Load and prepare training and testing data.

    Parameters:
    - config: The configuration object containing file paths and chunk size.

    Returns:
    - examples: Combined list of training and testing examples.
    - qp_map: Merged question-to-source map for training and testing data.
    """
    # Create Examples and Queries-PDF map
    train_examples = corpus_to_examples(
        json_path=config.train_json_path, 
        chunk_size=config.chunk_size
    )
    test_examples = corpus_to_examples(
        json_path=config.test_json_path, 
        chunk_size=config.chunk_size
    )
    examples = train_examples + test_examples
    
    train_qp_map = map_questions_to_sources(csv_path=config.train_csv_path)
    test_qp_map = map_questions_to_sources(csv_path=config.test_csv_path)
    qp_map = {**train_qp_map, **test_qp_map}

    return examples, qp_map

def main(config):
    """
    Main function to configure the model, generate synthetic queries,
    and save them to a JSON file.

    Parameters:
    - config: The configuration object containing all necessary parameters.
    """
    # Configure the LlamaCpp model
    llamalm = configure_llama_cpp_model(
        repo_id=config.repo_id, 
        model_name=config.model_name,
        local_dir=config.local_dir, 
        temperature=config.temperature,
    )
    dspy.settings.configure(lm=llamalm)

    # Load and prepare data
    examples, qp_map = load_and_prepare_data(config)

    # Initialize the synthetic generator and generate synthetic queries
    generator = SyntheticGenerator(qp_map=qp_map)
    qp_pairs = generate_synthetic_queries(generator, examples)

    # Save the generated queries to a JSON file
    save_to_json(qp_pairs, output_file=config.output_file)

if __name__ == "__main__":
    # Load configuration from a YAML file or from the command line
    config = OmegaConf.merge(
        OmegaConf.load("src/config.yaml"),  # Load from a YAML file if available
        OmegaConf.from_cli()  # Override with command-line arguments
    )

    # Call the main function with the loaded config
    main(config)