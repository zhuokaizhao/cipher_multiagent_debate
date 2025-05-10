"""
Generate Cipher Results
"""

import os
import subprocess


num_gpus = 4
all_model_names = [
    "llama3",
    # "phi4",
    # "qwen3-0_6b",
    # "qwen3-1_7b",
    # "ds-llama3",
]
problem_range = "501-1000"
all_num_agents = [3]  # only used for multiagent debate and cipher
all_num_rounds = [2]  # only used for multiagent debate and cipher

all_data_names = ["gsm8k"]  # "MATH", "GSM8K", "MMLU"
all_data_paths = {
    "gsm8k": "data/gsm8k/test.jsonl",
}

all_methods = [
    # "multiagent_debate",
    "cipher",
]


HF_HOME = os.environ.get("HF_HOME", "/fsx/zhuokai/.cache/huggingface")


for model_name in all_model_names:
    for data_name in all_data_names:
        # set the initial prompt
        initial_prompt_paths = f"/fsx/zhuokai/cipher_multiagent_debate/prompts_v2/{data_name}/init_question_0shot.txt"
        # set the data path
        data_path = all_data_paths[data_name]
        for method in all_methods:
            if method == "cipher" or method == "multiagent_debate":
                for num_agents in all_num_agents:
                    for num_rounds in all_num_rounds:
                        # generate the slurm file and directory
                        cur_version_name = f"{method}_{data_name}_agents_{num_agents}_rounds_{num_rounds}_problems_{problem_range}"
                        script_path = f"/fsx/zhuokai/cipher_multiagent_debate/scripts/{method}/{data_name}/{model_name}_{cur_version_name}.slurm"
                        script_dir = os.path.dirname(script_path)
                        if not os.path.exists(script_dir):
                            os.makedirs(script_dir, exist_ok=True)

                        # set the debaters
                        if num_agents == 1:
                            debaters = f"{model_name}"
                        elif num_agents == 2:
                            debaters = f"{model_name},{model_name}"
                        elif num_agents == 3:
                            debaters = f"{model_name},{model_name},{model_name}"
                        else:
                            raise ValueError(
                                f"Number of agents {num_agents} not supported"
                            )

                        with open(script_path, "w") as f:
                            lines_to_write = [
                                "#!/bin/bash\n",
                                "#\n",
                                "#SBATCH --chdir=/fsx/zhuokai/cipher_multiagent_debate/\n",
                                f"#SBATCH --gres=gpu:{num_gpus}\n",
                                "#SBATCH --mem 16G\n",
                                "#SBATCH -c 16\n",
                                f"#SBATCH --job-name={method}_{model_name}_{cur_version_name}\n",
                                f"#SBATCH --output=/fsx/zhuokai/cipher_multiagent_debate/slurm/{method}/{data_name}/{model_name}_{cur_version_name}.stdout\n",
                                f"#SBATCH --error=/fsx/zhuokai/cipher_multiagent_debate/slurm/{method}/{data_name}/{model_name}_{cur_version_name}.stderr\n",
                                "\n",
                            ]

                            if method == "cipher":  # vector language debate
                                # set the debate prompt
                                debate_prompt_paths = f"/fsx/zhuokai/cipher_multiagent_debate/prompts_v2/{data_name}/debate_{num_agents}debaters_vector_language_v1.txt"
                                lines_to_write.append(
                                    f"python run_debate.py --num_points 1 --n_rounds {num_rounds} --batch_size 4 --dataset {data_name} --data_path {data_path} --custom_range {problem_range} --debaters {debaters} --max_new_tokens 2048 --initial_prompt_paths {initial_prompt_paths} --debate_prompt_path {debate_prompt_paths} --temperatures 0.6,0.6,0.6,0.6,0.6,0.6 --n_ray_actors 1 --n_gpus_per_actor 1 --vector_language\n"
                                )
                            elif method == "multiagent_debate":  # human language debate
                                # set the debate prompt
                                debate_prompt_paths = f"/fsx/zhuokai/cipher_multiagent_debate/prompts_v2/{data_name}/debate_{num_agents}debaters_v1.txt"
                                lines_to_write.append(
                                    f"python run_debate.py --num_points 1 --n_rounds {num_rounds} --batch_size 4 --dataset {data_name} --data_path {data_path} --custom_range {problem_range} --debaters {debaters} --max_new_tokens 2048 --initial_prompt_paths {initial_prompt_paths} --debate_prompt_path {debate_prompt_paths} --temperatures 0.6,0.6,0.6,0.6,0.6,0.6 --n_ray_actors 1 --n_gpus_per_actor 1\n"
                                )

                            for cur_line in lines_to_write:
                                f.write(cur_line)
                            f.close()

                        subprocess.run(
                            [
                                "sbatch",
                                f"{script_path}",
                            ]
                        )
                        print(
                            f"Submitted task for {method}_{model_name}_{cur_version_name}\n"
                        )
