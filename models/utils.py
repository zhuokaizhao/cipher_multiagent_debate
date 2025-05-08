import json
from zoneinfo import ZoneInfo
from typing import Optional
import torch
import gc
from typing import Dict
import yaml


def type_list(type_: str):
    if type_ == "int":
        return lambda s: [int(x) for x in s.split(",")]
    elif type_ == "float":
        return lambda s: [float(x) for x in s.split(",")]
    elif type_ == "str":
        return lambda s: [x for x in s.split(",")]
    else:
        raise NotImplementedError()


def ensure_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def maybe_duplicate(x, n_elements):
    x = ensure_list(x)
    if len(x) < n_elements:
        res = (x * n_elements)[:n_elements]
    else:
        res = x
    return res


def datetime_now(time_format: Optional[str] = None) -> str:
    from datetime import datetime

    if time_format is None:
        time_format = "%Y-%b-%d--%H-%M-%S"
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime(time_format)


def get_model_path(model_name, hdfs: bool = False):
    if not hdfs:
        if model_name == "mistral":
            # model_str = "mistralai/Mistral-7B-Instruct-v0.2"
            model_str = "mistralai/Ministral-8B-Instruct-2410"
        elif model_name == "llama3":
            # model_str = "meta-llama/Meta-Llama-3-8B"
            model_str = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif model_name == "ds-llama3":
            model_str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        elif model_name == "phi3":
            model_str = "microsoft/Phi-3-mini-128k-instruct"
        elif model_name == "qwen3-0_6b":
            model_str = "Qwen/Qwen3-0.6B"
        elif model_name == "qwen3-1_7b":
            model_str = "Qwen/Qwen3-1.7B"
        elif model_name == "phi4":
            model_str = "microsoft/Phi-4-mini-instruct"
        elif model_name == "gpt2":
            model_str = "gpt2"
        elif model_name == "llama7b":
            model_str = "meta-llama/Llama-2-7b-chat-hf"
    else:
        return model_name

    return model_str


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_gpu_mem_all() -> None:
    ## get all gpu available
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        free_gb = get_gpu_mem(cuda=f"cuda:{i}")
        print(f"\tdevice: {i+1}/{n_gpus}, avail mem: {free_gb}GB")


def get_gpu_mem(cuda="cuda:0") -> str:
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    return f"{round(free_gb, 2)}/{round(total_gb, 2)}"


def clear_gpu_mem(verbose: bool = False):
    if verbose:
        print(f"mem available before clearing:")
        get_gpu_mem_all()

    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"mem available after clearing:")
        get_gpu_mem_all()


def read_json(data_path: str) -> Dict:
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def duplicate_temp(n_rounds, n_agents, temp):
    ## update temperature_1 in new args
    temperatures = [-1] * n_rounds * n_agents
    for d in range(n_agents):
        temperatures[d * n_rounds : (d + 1) * n_rounds] = [temp[d]] * n_rounds
    return temperatures
