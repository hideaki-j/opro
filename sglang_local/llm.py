from typing import List
from .sglang_wrappers import SGLangWrapper

def batch_complete(prompts: List[str], verbose: bool = False, model: str = "sglang", **kwargs) -> List[str]:
    """
    A batched completion interface that uses the SGLang backend.
    """
    if verbose:
        print(f"[sglang] Sending batch of {len(prompts)} prompts to SGLang backend.")
    sglang_engine = SGLangWrapper(
        max_tokens=kwargs.get("max_tokens", 768),
        temperature=kwargs.get("temperature", 0.0)
    )
    return sglang_engine.get_completion_from_prompts(prompts, verbose=verbose)
