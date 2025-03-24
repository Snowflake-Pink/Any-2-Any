import os
import json
def parse_workflow_to_prompt(workflow):
    prompt = {}
    links = {}

    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        links[link_id] = {'source_id': source_id, 'source_output': source_output}

    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    parameter['value'] = node_info['widgets_values'][parameter['index']]
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        node_inputs = {}
        for key, value in node_template['parameters'].items():
            node_inputs[key] = value['value']
        if 'inputs' in node_info:
            for item in node_info['inputs']:
                if item['link'] is not None:
                    source_id = links[item['link']]['source_id']
                    source_output = links[item['link']]['source_output']
                    node_inputs[item['name']] = [str(source_id), source_output]

        prompt[str(node_id)] = {'inputs': node_inputs, 'class_type': node_type}

    return prompt

if __name__ == '__main__':
    with open('/path/to/workflow.json', 'r') as f:
        workflow = json.load(f)
    prompt = parse_workflow_to_prompt(workflow)
    with open('./debug/prompt.json', 'w') as f:
        json.dump(prompt, f, indent=4)