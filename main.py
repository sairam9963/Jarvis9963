# — coding: utf-8 –
import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from easytool import funcQA, restbench, toolbench_retrieve, toolbench
from easytool.util import *

# Model selection based on user input
MODEL_CHOICES = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "falcon": "tiiuae/falcon-7b",
    "mistral": "mistralai/Mistral-7B-Instruct"
}

def load_model(model_name):
    """ Load the selected model """
    print(f"Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return tokenizer, model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama', choices=MODEL_CHOICES.keys(), help="Choose model: llama, falcon, mistral")
    parser.add_argument('--task', type=str, default='funcqa', help='funcqa, toolbench_retrieve, toolbench, restbench')
    parser.add_argument('--data_type', type=str, default='G3')
    
    args = parser.parse_args()

    # Load the selected model
    MODEL_NAME = MODEL_CHOICES[args.model]
    tokenizer, model = load_model(MODEL_NAME)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate_response(prompt):
        result = generator(prompt, max_length=200, do_sample=True, temperature=0.7)
        return result[0]['generated_text']

    # Task execution remains the same
    print("Executing task:", args.task)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='funcqa_mh', help='funcqa, toolbench_retrieve, toolbench, restbench')
    parser.add_argument('--data_type', type=str, default='G3', help='G2 or G3 or funcqa_mh or funcqa_oh')
    parser.add_argument('--tool_root_dir', type=str, default='.toolenv/tools/')
    parser.add_argument('--retrieval_num', type=int, default=5)

    args = parser.parse_args()

    # Load dataset based on task selection
    if args.task == 'funcqa':
        dataset = read_json('data_funcqa/tool_instruction/functions_data.json')
        Tool_dic = read_jsonline('data_funcqa/tool_instruction/tool_dic.jsonl')
        test_data = read_json(f"data_funcqa/test_data/{args.data_type}.json")
        progress_file = f"FuncQA_{args.data_type}_LLM.txt"
    
    elif 'toolbench' in args.task:
        base_path = args.tool_root_dir
        index = build_index(base_path)
        dataset = read_json('data_toolbench/tool_instruction/toolbench_tool_instruction.json')
        test_data = read_json(f'''data_toolbench/test_data/{args.data_type}_instruction.json''')
        progress_file = f'''{args.data_type}_LLM.txt'''
    
    elif args.task == 'restbench':
        Tool_dic = read_json('data_restbench/tool_instruction/tmdb_tool.json')
        dic_tool = {data['ID']: data for data in Tool_dic}
        test_data = read_json('data_restbench/test_data/tmdb.json')
        progress_file = f"restbench_LLM.txt"

    else:
        print("Wrong task name")
        exit()

    start_index = get_last_processed_index(progress_file)
    total_files = len(test_data)
    retrieval_num = args.retrieval_num
    ind = start_index

    print("-------Start Execution-------")

    # Execute tasks using LLaMA 2 / Falcon instead of OpenAI
    if args.data_type in ['funcqa_mh', 'funcqa_oh']:
        funcQA.task_execution_mh(args.data_type, start_index, total_files, retrieval_num, ind, MODEL_NAME, dataset, Tool_dic, test_data, progress_file, generate_response)

    elif args.task == 'toolbench_retrieve':
        toolbench_retrieve.task_execution(args.data_type, base_path, index, dataset, test_data, progress_file, start_index, total_files, retrieval_num, ind, MODEL_NAME, generate_response)

    elif args.task == 'toolbench':
        toolbench.task_execution(args.data_type, base_path, index, dataset, test_data, progress_file, start_index, total_files, retrieval_num, ind, MODEL_NAME, generate_response)

    elif args.task == 'restbench':
        restbench.task_execution(Tool_dic, dic_tool, test_data, progress_file, start_index, total_files, retrieval_num, ind, MODEL_NAME, generate_response)

    else:
        print("Wrong task name")
        exit()

