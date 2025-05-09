"""
Generate Cipher Results
"""

import os
import subprocess


num_gpus = 4
all_model_names = [
    "llama3",
    "phi4",
    "qwen3-0_6b",
    "qwen3-1_7b",
    "ds-llama3",
]
all_problem_ranges = ["501-1000"]
all_num_agents = [3]
all_num_rounds = [2]

all_data_names = ["gsm8k"]  # "MATH", "GSM8K", "MMLU"
all_data_paths = {
    "gsm8k": "data/gsm8k/test.jsonl",
}

all_methods = ["cipher"]


HF_HOME = os.environ.get("HF_HOME", "/fsx/zhuokai/.cache/huggingface")


for method in all_methods:
    for model_name in all_model_names:
        for num_agents in all_num_agents:
            for num_rounds in all_num_rounds:
                for data_name in all_data_names:
                    # set the data path
                    data_path = all_data_paths[data_name]

                    # set the initial prompt
                    initial_prompt_paths = f"/fsx/zhuokai/cipher_multiagent_debate/prompts_v2/{data_name}/init_question_0shot.txt"

                    # set the debate prompt
                    debate_prompt_paths = f"/fsx/zhuokai/cipher_multiagent_debate/prompts_v2/{data_name}/debate_{num_agents}debaters_v1.txt"

                    # set the debaters
                    if num_agents == 1:
                        debaters = f"{model_name}"
                    elif num_agents == 2:
                        debaters = f"{model_name},{model_name}"
                    elif num_agents == 3:
                        debaters = f"{model_name},{model_name},{model_name}"

                    for problem_range in all_problem_ranges:
                        model_name_str = model_name.split("/")[-1]

                        # generate the slurm file and directory
                        cur_version_name = f"{data_name}_agents_{num_agents}_rounds_{num_rounds}_problems_{problem_range}"
                        script_path = f"/fsx/zhuokai/cipher_multiagent_debate/scripts/{method}/{data_name}/{model_name_str}_{cur_version_name}.slurm"
                        script_dir = os.path.dirname(script_path)
                        if not os.path.exists(script_dir):
                            os.makedirs(script_dir, exist_ok=True)

                        with open(script_path, "w") as f:
                            lines_to_write = [
                                "#!/bin/bash\n",
                                "#\n",
                                "#SBATCH --chdir=/fsx/zhuokai/cipher_multiagent_debate/\n",
                                f"#SBATCH --gres=gpu:{num_gpus}\n",
                                "#SBATCH --mem 16G\n",
                                "#SBATCH -c 16\n",
                                f"#SBATCH --job-name={method}_{model_name_str}_{cur_version_name}\n",
                                f"#SBATCH --output=/fsx/zhuokai/cipher_multiagent_debate/slurm/{method}/{data_name}/{model_name_str}_{cur_version_name}.stdout\n",
                                f"#SBATCH --error=/fsx/zhuokai/cipher_multiagent_debate/slurm/{method}/{data_name}/{model_name_str}_{cur_version_name}.stderr\n",
                                "\n",
                            ]
                            if method == "cipher":
                                lines_to_write.append(
                                    f"python run_debate.py --num_points 1 --n_rounds {num_rounds} --batch_size 4 --dataset {data_name} --data_path {data_path} --custom_range {problem_range} --debaters {debaters} --max_new_tokens 2048 --initial_prompt_path {initial_prompt_paths} --debate_prompt_path {debate_prompt_paths} --temperatures 0.6,0.6,0.6,0.6,0.6,0.6 --n_ray_actors 1 --n_gpus_per_actor 1\n"
                                )
                            else:
                                raise ValueError(f"Method {method} not supported")

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
                            f"Submitted task for {method}_{model_name_str}_{cur_version_name}\n"
                        )
