import os
import json
import sys
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(current_directory)
from utils.parser import parse_workflow_to_code, safe_var_name

def parse_workflow_to_node(workflow):
    code = ''
    type_list = []
    node_dict = {}

    code += '# create nodes by instantiation\n'
    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']
        node_name = safe_var_name(f'{node_type.lower()}_{node_id}') # only space changes to '_'
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    parameter['value'] = node_info['widgets_values'][parameter['index']]
                    if isinstance(parameter['value'], str):
                        parameter['value'] = parameter['value'].replace('\n', ' ')
                        parameter['value'] = f'"""{parameter["value"]}"""'
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    if isinstance(parameter_value, str):
                        parameter_value = parameter_value.replace('\n', ' ')
                        parameter_value = f'"""{parameter_value}"""'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        parameter_list = []
        for key, value in node_template['parameters'].items():
            parameter_list.append(f'{key}={value["value"]}')
        code += f'{node_name} = {safe_var_name(node_type)}({", ".join(parameter_list)})\n'

        node_input = []
        if 'inputs' in node_info:
            node_input = [(item['name'], item['type'], item['link']) for item in node_info['inputs']]
        node_output = []
        if 'outputs' in node_info:
            node_output = [(item['name'], item['type'], item['links']) for item in node_info['outputs']]
        node_dict[node_id] = {'name': node_name, 'input': node_input, 'output': node_output}

    return code

if __name__ == "__main__":
    wfdir = "./checkpoint/test/"
    for wf_name in os.listdir(wfdir):
        if not wf_name.endswith("json"): continue
        
        wf_path = os.path.join(wfdir,wf_name)
        
        o_path= wfdir.replace("gt_workflow","gt_node")
        o_path2= wfdir.replace("gt_workflow","gt_node_and_link")

        if not os.path.exists(o_path): os.makedirs(o_path)
        if not os.path.exists(o_path2): os.makedirs(o_path2)
        
        with open(wf_path,"r") as f:
            wf = json.load(f)
        try:
            node = parse_workflow_to_node(wf)
        except Exception as e:
            print(e)
            print(wf_name)
            continue
        try:
            node_and_link = parse_workflow_to_code(wf)
        except Exception as e:
            print(wf_name)
            print(e)
            continue
        wf_name = wf_name.replace(".json","")
        with open(os.path.join(o_path, f'{wf_name}.py'), 'w') as f:
            f.write(node)
        with open(os.path.join(o_path2, f'{wf_name}.py'), 'w') as f:
            f.write(node_and_link)