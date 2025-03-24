import os
import yaml
import argparse

from inference_engine.dataflow.pipeline import DataflowPipeline
from inference_engine.declarative.pipeline import DeclarativePipeline
from inference_engine.pseudo_natural.pipeline import PseudoNaturalPipeline
from inference_engine.onestep.pipeline import OneStepPipeline

with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    proxy_config = config['proxy']

os.environ['http_proxy'] = proxy_config['http_proxy']
os.environ['https_proxy'] = proxy_config['https_proxy']
print(f"HTTP Proxy: {os.environ.get('http_proxy')}")
print(f"HTTPS Proxy: {os.environ.get('https_proxy')}")

def main(args):
    with open(args.json_path, 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)
    print(args.inference_engine_name)
    for inference_engine_name in args.inference_engine_name:
        print(f'[Inference] inference_engine {inference_engine_name}')

        for task_id, task_info in metadata.items():

            query = task_info['content']
            print(f'[Inference] task {task_id}')

            for run_id in range(1, args.num_runs + 1):
               
                checkpoint = f'{args.save_path}/{inference_engine_name}/task_{task_id}/run_{run_id:03d}'
                os.makedirs(checkpoint, exist_ok=True)
                print(f'[Inference] run {run_id}/{args.num_runs}')

                # Skip: already inferred
                log_path = os.path.join(checkpoint, 'run.log')
                if not args.force_run and os.path.exists(log_path):
                    print('skipped: already inferred')
                    continue
                
                # Create pipeline
                elif inference_engine_name == 'dataflow':
                    key_nodes_path = task_info.get('keynode')
                    if not key_nodes_path:
                        key_nodes_path = os.path.join(os.path.dirname(args.json_path), "keynode", "task_01001.py")
                    if not os.path.exists(key_nodes_path):
                        print(f"[Skip task_{task_id}] No node code in {key_nodes_path}")
                        continue
                    with open(key_nodes_path,"r") as f:
                        key_nodes = f.read()
                    pipeline = DataflowPipeline(
                        save_path=checkpoint,
                        key_nodes=key_nodes,
                        use_claude=args.use_claude 
                    )
                elif inference_engine_name == 'declarative':
                    key_nodes_path = task_info.get('keynode')
                    print(key_nodes_path)
                    if not key_nodes_path:
                        key_nodes_path = os.path.join(os.path.dirname(args.json_path), "keynode", "task_01001.py")
                    key_nodes  = ""
                    if isinstance(key_nodes_path, list): # For comfybench-complex
                        for path in key_nodes_path:
                            if not os.path.exists(path):
                                print(f"[Skip task_{task_id}] No node code in {path}")
                                continue
                            with open(path, "r") as f:
                                key_nodes += f.read() + "\n\n"
                        key_nodes = key_nodes.replace("# create nodes by instantiation", "[PLACEHOLDER]", 1) # remain first comment
                        key_nodes = key_nodes.replace("# create nodes by instantiation", "")
                        key_nodes = key_nodes.replace("[PLACEHOLDER]", "# create nodes by instantiation", 1)
                        
                    elif isinstance(key_nodes_path, str):
                        if not os.path.exists(key_nodes_path):
                            # key_nodes = ""
                            absolute_path = os.path.abspath(key_nodes_path)
                            print(f"[Skip task_{task_id}] No node code in {absolute_path}")
                            continue
                        with open(key_nodes_path, "r") as f:
                            key_nodes = f.read()
                    else:
                        print(f"[Skip task_{task_id}] Invalid keynode type: {type(key_nodes_path)}")
                        continue
                    pipeline = DeclarativePipeline(
                        save_path=checkpoint,
                        key_nodes=key_nodes,
                        use_claude=args.use_claude 
                    )
                elif inference_engine_name == 'pseudo_natural':
                    key_nodes_path = task_info.get('keynode')
                    if not key_nodes_path:
                        key_nodes_path = os.path.join(os.path.dirname(args.json_path), "keynode", "task_01001.py")
                    if not os.path.exists(key_nodes_path):
                        print(f"[Skip task_{task_id}] No node code in {key_nodes_path}")
                        continue
                    with open(key_nodes_path,"r") as f:
                        key_nodes = f.read()
                    pipeline = PseudoNaturalPipeline(
                        save_path=checkpoint,
                        key_nodes=key_nodes,
                        use_claude=args.use_claude 
                    )
                elif inference_engine_name == 'onestep':
                    key_nodes_path = task_info.get('keynode_path')
                    if not key_nodes_path:
                        key_nodes_path = os.path.join(os.path.dirname(args.json_path), "keynode", "task_01001.py")
                    key_nodes  = ""
                    if isinstance(key_nodes_path, list): # For comfybench-complex
                        for path in key_nodes_path:
                            if not os.path.exists(path):
                                print(f"[Skip task_{task_id}] No node code in {path}")
                                continue
                            with open(path, "r") as f:
                                key_nodes += f.read() + "\n\n"
                        key_nodes = key_nodes.replace("# create nodes by instantiation", "[PLACEHOLDER]", 1) # remain first comment
                        key_nodes = key_nodes.replace("# create nodes by instantiation", "")
                        key_nodes = key_nodes.replace("[PLACEHOLDER]", "# create nodes by instantiation", 1)
                        
                    elif isinstance(key_nodes_path, str):
                        if not os.path.exists(key_nodes_path):
                            print(f"[Skip task_{task_id}] No node code in {key_nodes_path}")
                            continue
                        with open(key_nodes_path, "r") as f:
                            key_nodes = f.read()
                    else:
                        print(f"[Skip task_{task_id}] Invalid keynode type: {type(key_nodes_path)}")
                        continue
                    pipeline = OneStepPipeline(
                        save_path=checkpoint,
                        key_nodes=key_nodes,
                        use_claude=args.use_claude 
                    )
                # Run pipeline
                try:
                    workflow = pipeline(query)
                except Exception as error:
                    print(error)
                    workflow = None

                # Check: pipeline status
                if workflow is None:
                    print(f'done: pipeline failed')
                else:
                    print(f'done: pipeline succeeded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inference_engine_name',
        nargs='+',
        type=str
    )
    parser.add_argument(
        '--save_path',
        default='./checkpoint/benchmark',
        type=str
    )
    parser.add_argument(
        '--num_runs',
        default=1,
        type=int
    )
    parser.add_argument(
        '--num_steps',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_refs',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_fixes',
        default=3,
        type=int
    )
    parser.add_argument(
        '--force_run',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--use_claude',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--json_path',
        type=str,
        default="./dataset/query/meta.json"
    )
    args = parser.parse_args()
    main(args)
