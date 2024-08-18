from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from dsp.modules.llama import LlamaCpp


def download_model(repo_id: str, model_name: str, local_dir: str) -> str:
    """
    Download a model file from Hugging Face Hub and save it to a specified local directory.
    """
    return hf_hub_download(repo_id=repo_id, filename=model_name, local_dir=local_dir)


def initialize_llama_model(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 0, verbose: bool = False) -> Llama:
    """
    Initialize the Llama model with the specified configuration.
    """
    return Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=verbose)


def configure_llama_cpp_model(repo_id: str, model_name: str, local_dir: str, temperature: float=0.) -> LlamaCpp:
    """
    Download the Llama model and configure it for use with dspy.
    """
    model_path = download_model(repo_id, model_name, local_dir)
    llm = initialize_llama_model(model_path)
    return LlamaCpp(model="llama", llama_model=llm, model_type="chat", temperature=temperature)
