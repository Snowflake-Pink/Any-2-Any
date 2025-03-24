import os
import yaml
import json
import argparse
import pandas as pd

from utils.parser import parse_code_to_workflow, parse_markdown_to_workflow
from utils.comfy import execute_workflow


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    proxy_config = config['proxy']

os.environ['http_proxy'] = proxy_config['http_proxy']
os.environ['https_proxy'] = proxy_config['https_proxy']


def main(args):
    with open(args.json_path, 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)
    record = dict()
    missing_directories = 0
    for inference_engine_name in args.inference_engine_name:
        record[inference_engine_name] = dict()
        for task_id, task_info in list(metadata.items()):
            record[inference_engine_name][task_id] = {
                'task_info': task_info,
                'num_runs': 0,
                'num_compile_passes': 0,
                'num_execute_passes': 0
            }

    for inference_engine_name in args.inference_engine_name:
        print(f'[Evaluation] inference_engine {inference_engine_name}') 
        for task_id in list(metadata.keys()):
            prefix = f'{args.save_path}/{inference_engine_name}/task_{task_id}'
            print(f'[Evaluation] task {task_id}')
            
            if not os.path.exists(prefix):
                print(f"Directory does not exist: {prefix}")
                missing_directories += 1
                continue

            for run_id in os.listdir(prefix):
                record[inference_engine_name][task_id]['num_runs'] += 1
                checkpoint = os.path.join(prefix, run_id)
                print(f'[Evaluation] checkpoint {checkpoint}')

                # Skip: already evaluated
                output_path = os.path.join(checkpoint, 'output')
                if os.path.exists(output_path):
                    record[inference_engine_name][task_id]['num_compile_passes'] += 1
                    record[inference_engine_name][task_id]['num_execute_passes'] += 1
                    print('skipped: already evaluated')
                    continue

                # Case: standard representation
                if inference_engine_name in [
                    'declarative',
                    'dataflow',
                    'pseudo_natural'
                ]:
                    # Check: no file
                    code_path = os.path.join(checkpoint, 'code.py')
                    if not os.path.exists(code_path):
                        print('skipped: no file')
                        continue

                    # Check: empty code
                    with open(code_path, 'r') as file:
                        code = file.read()
                    if code.strip() == '':
                        print('skipped: empty code')
                        continue

                    # Check: invalid workflow
                    try:
                        workflow = parse_code_to_workflow(code)
                    except Exception as error:
                        print('skipped: invalid workflow')
                        continue

                # Record: pass 1
                record[inference_engine_name][task_id]['num_compile_passes'] += 1

                # Check: execution failure
                try:
                    status, outputs = execute_workflow(workflow)
                except Exception as error:
                    print('skipped: execution failure')
                    continue

                # Check: invalid status
                if status['status_str'] != 'success':
                    print(f'status: {status['status_str']}')
                    print('skipped: invalid status')
                    continue

                # Save: execution output
                output_path = os.path.join(checkpoint, 'output')
                os.makedirs(output_path, exist_ok=True)
                for file_name, output in outputs.items():
                    file_path = os.path.join(output_path, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(output)

                # Record: pass 2
                record[inference_engine_name][task_id]['num_execute_passes'] += 1

    # For one run, run level equals to task level
    summary = {
        'inference_engine Name': [],
        'Run Compilation Pass Rate': [],
        'Run Execution Pass Rate': [],
        'Task Compilation Pass Rate': [],
        'Task Execution Pass Rate': [],
        'Task Name': []
    }
    for inference_engine_name, inference_engine_record in record.items():
        num_runs, num_tasks = 0, len(inference_engine_record)
        run_passes_1, run_passes_2 = 0, 0
        task_passes_1, task_passes_2 = 0, 0
        for task_record in inference_engine_record.values():
            num_runs += task_record['num_runs']
            run_passes_1 += task_record['num_compile_passes']
            run_passes_2 += task_record['num_execute_passes']
            if task_record['num_compile_passes'] > 0:
                task_passes_1 += 1
            if task_record['num_execute_passes'] > 0:
                task_passes_2 += 1
        summary['inference_engine Name'].append(inference_engine_name)
        summary['Run Compilation Pass Rate'].append(run_passes_1 / num_runs if num_runs > 0 else 0)
        summary['Run Execution Pass Rate'].append(run_passes_2 / num_runs if num_runs > 0 else 0)
        summary['Task Compilation Pass Rate'].append(task_passes_1 / (num_tasks-missing_directories) if num_tasks > 0 else 0)
        summary['Task Execution Pass Rate'].append(task_passes_2 / (num_tasks-missing_directories) if num_tasks > 0 else 0)
        summary['Task Name'].append(args.save_path.split("/")[-1])
    summary = pd.DataFrame(summary)
    
    if not os.path.join(args.save_path, "eval_result.csv"):
        summary.to_csv(os.path.join(args.save_path, "eval_result.csv"), mode='w', index=False)
    else:
        summary.to_csv(os.path.join(args.save_path, "eval_result.csv"), mode='a', header=False, index=False)
    print(summary.to_string())

    # Print summary of missing directories
    if missing_directories > 0:
        print(f"Total missing directories: {missing_directories}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inference_engine_name',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--save_path',
        default='./checkpoint/',
        type=str
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default="./dataset/query/meta.json"
    )
    args = parser.parse_args()
    main(args)
