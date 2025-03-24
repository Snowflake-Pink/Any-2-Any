import os
import time
import argparse

from utils.comfy import execute_workflow

from inference_engine.dataflow.pipeline import DataflowPipeline
from inference_engine.declarative.pipeline import DeclarativePipeline
from inference_engine.pseudo_natural.pipeline import PseudoNaturalPipeline

def main(args):
    # Setup checkpoint
    # if args.save_path is None:
    #     args.save_path = f"./checkpoint/{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    # os.makedirs(args.save_path, exist_ok=True)

    if args.inference_engine_name == 'dataflow':
        if args.key_node_file != None:
            with open(args.key_node_file, 'r') as f:
                args.key_node = f.read()
            if args.save_path == None:
                args.save_path = os.path.join(os.path.dirname(os.path.dirname(args.key_node_file)), "test_result", os.path.basename(args.key_node_file).replace(".py",""))
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        
        pipeline = DataflowPipeline(
            save_path=args.save_path,
            key_nodes=args.key_node
        )
        
    elif args.inference_engine_name == 'declarative':
        if args.key_node_file != None:
            with open(args.key_node_file, 'r') as f:
                args.key_node = f.read()
            if args.save_path == None:
                args.save_path = os.path.join(os.path.dirname(os.path.dirname(args.key_node_file)), "test_result", os.path.basename(args.key_node_file).replace(".py",""))
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        
        pipeline = DeclarativePipeline(
            save_path=args.save_path,
            key_nodes=args.key_node
        )
        
    elif args.inference_engine_name == 'pseudo_natural':
        if args.key_node_file != None:
            with open(args.key_node_file, 'r') as f:
                args.key_node = f.read()
            if args.save_path == None:
                args.save_path = os.path.join(os.path.dirname(os.path.dirname(args.key_node_file)), "test_result", os.path.basename(args.key_node_file).replace(".py",""))
            if not os.path.exists(args.save_path): os.makedirs(args.save_path)

        pipeline = PseudoNaturalPipeline(
            save_path=args.save_path,
            key_nodes=args.key_node
        )
        
    else:
        print(f'Unknown inference_engine name: {args.inference_engine_name}')
        return


    workflow = pipeline(args.query_text)
        
    if workflow is None:
        print('failed to generate workflow')
        return

    # Execute workflow
    status, outputs = execute_workflow(workflow)
    print(f'execution status: {status}')
    
    if status['status_str'] == 'success':
        os.makedirs(f'{args.save_path}/output', exist_ok=True)
        for filename, output in outputs.items():
            print(f'save file: {args.save_path}/output/{filename}')
            with open(f'{args.save_path}/output/{filename}', 'wb') as output_file:
                output_file.write(output)
    else:
        print(f"error with statu {status['status_str']}")

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_text', type=str, required=True)
    parser.add_argument('--inference_engine_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_steps', type=int, default=6)
    parser.add_argument('--num_refs', type=int, default=5)
    parser.add_argument('--num_fixes', type=int, default=1)
    parser.add_argument('--key_node', type=str, default='')
    parser.add_argument('--key_node_file', type=str, default=None)
    args = parser.parse_args()
    main(args)
