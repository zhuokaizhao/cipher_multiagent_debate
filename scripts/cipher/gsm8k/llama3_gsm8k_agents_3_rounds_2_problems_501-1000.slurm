#!/bin/bash
#
#SBATCH --chdir=/fsx/zhuokai/cipher_multiagent_debate/
#SBATCH --gres=gpu:1
#SBATCH --mem 16G
#SBATCH -c 16
#SBATCH --job-name=cipher_llama3_gsm8k_agents_3_rounds_2_problems_501-1000
#SBATCH --output=/fsx/zhuokai/cipher_multiagent_debate/slurm/cipher/gsm8k/llama3_gsm8k_agents_3_rounds_2_problems_501-1000.stdout
#SBATCH --error=/fsx/zhuokai/cipher_multiagent_debate/slurm/cipher/gsm8k/llama3_gsm8k_agents_3_rounds_2_problems_501-1000.stderr

python run_debate.py --num_points 1 --n_rounds 2 --batch_size 16 --dataset gsm8k --data_path data/gsm8k/test.jsonl --custom_range 501-1000 --debaters llama3,llama3,llama3 --max_new_tokens 2048 --initial_prompt_path /fsx/zhuokai/cipher_multiagent_debate/prompts_v2/gsm8k/init_question_0shot.txt --debate_prompt_path /fsx/zhuokai/cipher_multiagent_debate/prompts_v2/gsm8k/debate_3debaters_v1.txt --temperatures 0.2,0.2,0.2,0.8,0.8,0.8 --n_ray_actors 1 --n_gpus_per_actor 1
