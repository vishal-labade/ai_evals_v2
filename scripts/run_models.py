#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

models = {}

models["small"] = ["phi_mini", "phi_mini_t0", "qwen_3b", "qwen_3b_t0"]

models["medium"] = ["gemma_9b_t0","gemma_9b", "llama_8b_t0", "llama_8b"]

models["large"] = ["qwen_14b_t0","qwen_14b", "mistral_12b_t0", "mistral_12b"]

models["xlarge"] = ["gpt_20b_t0","gpt_20b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", required=True)
    args = ap.parse_args()

    
    for model_id in models[args.model]:
        print(f"Running Model :  {model_id}")
        cmd = [
            sys.executable,
            "scripts/run_model_ollama.py",
            "--config",
            "configs/dev_rog.yaml",
            "--modelid",
            model_id,
            ]
        print("Running:", " ".join(cmd))
        env =  dict(os.environ)
        env["OLLAMA_KEEP_ALIVE"] = "0"
        result = subprocess.run(cmd, check=True, shell=False, env=env)
        if result.returncode != 0:
            print(f"❌ Error Running: {model_id}")
            sys.exit(result.returncode)
        print(f"Completed Model :  {model_id}")



if __name__ == '__main__':
    main()

