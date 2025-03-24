import os
import re
import ast
import json
import io
import tokenize

def extract_key_value_pair(text):
    key, value = text.split(':', 1)
    return key.strip(), value.strip()


def fetch_name_by_index(dictionary, index):
    for key, item in dictionary.items():
        if item['index'] == index:
            return key
    return None

def safe_var_name(name):
    return name.replace('[', 'LeftBracket').replace(']', 'RightBracket').replace('(', 'LeftParen').replace(')', 'RightParen').replace(' ','Space').replace('+','PLUS').replace('-','Hyphen')

def unsafify_var_name(name):
    return name.replace('LeftBracket', '[').replace('RightBracket', ']').replace('LeftParen', '(').replace('RightParen', ')').replace('Space',' ').replace('PLUS','+').replace('Hyphen','-')

def extract_nested_markdown_list(markdown):
    stack = [{}]
    depth = 0
    pattern = re.compile(r'( *)- ([^:\n]+)(?:: ([^\n]*))?\n?')
    for space, key, value in pattern.findall(markdown):
        indent = len(space)
        if indent > depth:
            assert not stack[-1]
        elif indent < depth:
            stack.pop()
        if value:
            stack[-1][key] = value
        else:
            stack[-1][key] = {}
            stack.append(stack[-1][key])
        depth = indent
    content = stack[0]
    return content


def parse_code_to_workflow(code):
    node_count = 0
    link_count = 0
    object_dict = {}
    tensor_dict = {}
    node_dict = {}
    link_dict = {}
    tree_root = ast.parse(code)

    for tree_node in tree_root.body:
        code_line = ast.unparse(tree_node).strip()
        function_name = None
        variable_list = []
        parameter_list = []
        if isinstance(tree_node, ast.Assign):
            assign_node = tree_node
            call_node = assign_node.value
            function_name = unsafify_var_name(call_node.func.id)
        else:
            continue
        call_node = assign_node.value
        function_name = call_node.func.id

        target_list = assign_node.targets
        for target in target_list:
            if isinstance(target, ast.Name):
                variable_list.append(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    assert isinstance(element, ast.Name), f'code line {code_line}: unexpected target type {type(element)}'
                    variable_list.append(element.id)
            else:
                raise ValueError(f'code line {code_line}: unexpected target type {type(target)}')

        keyword_list = call_node.keywords
        for keyword in keyword_list:
            if isinstance(keyword.value, ast.Name):
                parameter_list.append((keyword.arg, keyword.value.id))
            elif isinstance(keyword.value, ast.Constant):
                parameter_list.append((keyword.arg, keyword.value.value))
            else:
                raise ValueError(f'code line {code_line}: unexpected keyword type {type(keyword.value)}')

        template_name = f'{unsafify_var_name(function_name)}.json'

        if template_name in os.listdir('./dataset/docs/template/'):
            template_path = f'./dataset/docs/template/{template_name}'
            with open(template_path, 'r') as template_file:
                node = json.load(template_file)

            for parameter_name, parameter_value in parameter_list:
                parameter_value = unsafify_var_name(parameter_value.replace('_',' ')) if isinstance(parameter_value, str) else parameter_value
                assert parameter_name in node['parameters'], f'code line {code_line}: parameter {parameter_name} not found in node {function_name}'
                node['parameters'][parameter_name]['value'] = parameter_value.replace(' ','_') if isinstance(parameter_value, str) else parameter_value

            node_count += 1
            node_id = node_count
            node['id'] = node_id
            node_name = variable_list[0]
            object_dict[node_name] = node_id
            node_dict[node_id] = node

        # invoke
        else:
            assert function_name in object_dict, f'code line {code_line}: function {function_name} not found'
            node_id = object_dict[function_name]
            node = node_dict[node_id]

            for parameter_name, parameter_value in parameter_list:
                parameter_value = unsafify_var_name(parameter_value) if isinstance(parameter_value, str) else parameter_value
                assert parameter_name in node['inputs'] or parameter_name in node['parameters'], f'code line {code_line}: input {parameter_name} not found in node {function_name}.'

                if parameter_value is None:
                    continue
                
                assert parameter_value in tensor_dict, f'code line {code_line}: variable {parameter_value} is used before defined.'

                if parameter_name in node['parameters']:
                    node['inputs'][parameter_name] = node['parameters'][parameter_name]

                input_index = node['inputs'][parameter_name]['index']
                link_type = node['inputs'][parameter_name]['type']   
                last_id, output_index = tensor_dict[parameter_value]
                last_node = node_dict[last_id]
                last_name = fetch_name_by_index(last_node['outputs'], output_index)
                last_type = last_node['outputs'][last_name]['type']
                assert link_type == last_type, f'code line {code_line}: type mismatch between {last_type} and {link_type}'

                link_count += 1
                link_id = link_count
                link = [last_id, output_index, node_id, input_index, link_type]
                link_dict[link_id] = link
                node['inputs'][parameter_name]['link'] = link_id
                last_node['outputs'][last_name]['links'].append(link_id)

            for output_index, variable in enumerate(variable_list):
                tensor_dict[variable] = (node_id, output_index)

    workflow = {
        'nodes': [],
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
        'version': '0.4'
    }

    for node_id, node in node_dict.items():
        node_info = {
            'id': node_id,
            'type': node['type'],
            'inputs': [],
            'outputs': [],
            'widgets_values': [],
        }
        for input_name, input in node['inputs'].items():
            node_info['inputs'].append({
                'name': input_name,
                'type': input['type'],
                'link': input['link'],
                'slot_index': input['index']
            })
        node_info['inputs'].sort(key=lambda x: x['slot_index'])

        for output_name, output in node['outputs'].items():
            node_info['outputs'].append({
                'name': output_name,
                'type': output['type'],
                'links': output['links'],
                'slot_index': output['index']
            })
        node_info['outputs'].sort(key=lambda x: x['slot_index'])

        parameter_list = list(node['parameters'].values())
        parameter_list.sort(key=lambda x: x['index'])
        for parameter in parameter_list:
            node_info['widgets_values'].append(parameter['value'])
        # parameter_list = list(node['parameters'].items())
        # parameter_list.sort(key=lambda x: x[1]['index'])
        # for key, parameter in parameter_list:
        #     node_info['widgets_values'].update({key: parameter['value']})
            
        workflow['nodes'].append(node_info)

    for link_id, link in link_dict.items():
        workflow['links'].append([link_id] + link)

    return workflow



def parse_workflow_to_code(workflow):
    code = ''
    type_list = []
    node_dict = {}
    link_dict = {}

    code += '# create nodes by instantiation\n'
    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']
        node_name = safe_var_name(f'{node_type.lower()}_{node_id}')
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node_template = json.load(template_file)

        if 'widgets_values' in node_info:
            if isinstance(node_info['widgets_values'], list):
                for parameter in node_template['parameters'].values():
                    # print(template_path)
                    # print(parameter['index'])
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

    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        link_dict[link_id] = {'variable': None, 'source': source_id, 'target': target_id}

    code += '\n# link nodes by invocation\n'
    remain_node = list(node_dict.keys())
    while remain_node:
        for node_id in remain_node:
            flag = True
            node = node_dict[node_id]
            for _, _, link_id in node['input']:
                if link_id is None:
                    continue
                if link_dict[link_id]['variable'] is None:
                    flag = False
                    break

            if flag:
                remain_node.remove(node_id)

                parameter_list = []
                for input_name, _, input_link in node['input']:
                    if input_link is None:
                        input_value = 'None'
                    else:
                        input_value = link_dict[input_link]['variable']
                    parameter_list.append(f'{input_name}={input_value}')

                return_list = []
                for output_name, _, output_links in node['output']:
                    return_name = f'{output_name.replace(" ", "_").lower()}_{node_id}'
                    return_list.append(return_name)
                    if isinstance(output_links, list):
                        for link_id in output_links:
                            link_dict[link_id]['variable'] = return_name
                if not return_list:
                    return_list.append(f'result_{node_id}')

                code += f'{", ".join(return_list)} = {node["name"]}({", ".join(parameter_list)})\n'

    return code


def parse_markdown_to_workflow(markdown):
    type_list = []
    node_dict = {}
    link_dict = {}

    pattern = re.compile(r'- Nodes:\n(.*)- Links:\n(.*)', re.DOTALL)
    node_content, link_content = pattern.search(markdown).groups()
    node_content = extract_nested_markdown_list(node_content)
    link_content = extract_nested_markdown_list(link_content)

    for node_name, node_info in node_content.items():
        node_id = int(node_name[1:])
        node_type = node_info['node_type'][1:-1]
        type_list.append(node_type)

        template_path = f'./dataset/docs/template/{node_type}.json'
        assert os.path.exists(template_path), f'template {template_path} not found'
        with open(template_path, 'r', encoding='utf-8') as template_file:
            node = json.load(template_file)
        node_output = {key.lower(): value for key, value in node['outputs'].items()}
        node['outputs'] = node_output

        node['id'] = node_id
        for key, value in node_info.items():
            if key == 'node_type':
                continue
            else:
                assert key in node['parameters'], f'parameter {key} not found in node {node_type}'
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                else:
                    value = eval(value)
                node['parameters'][key]['value'] = value

        node_dict[node_id] = node

    for link_name, link_info in link_content.items():
        link_id = int(link_name[1:])
        source_info, target_info = link_info.split(' -> ')
        source_name, source_port = source_info.split('.')
        target_name, target_port = target_info.split('.')

        source_id = int(source_name[1:])
        source_output = node_dict[source_id]['outputs'][source_port]['index']
        target_id = int(target_name[1:])
        target_input = node_dict[target_id]['inputs'][target_port]['index']
        link_type = node_dict[source_id]['outputs'][source_port]['type']

        node_dict[source_id]['outputs'][source_port]['links'].append(link_id)
        node_dict[target_id]['inputs'][target_port]['link'] = link_id
        link_dict[link_id] = [source_id, source_output, target_id, target_input, link_type]

    workflow = {
        'nodes': [],
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
        'version': '0.4'
    }

    for node_id, node in node_dict.items():
        node_info = {
            'id': node_id,
            'type': node['type'],
            'inputs': [],
            'outputs': [],
            'widgets_values': [],
        }

        for input_name, input in node['inputs'].items():
            node_info['inputs'].append({
                'name': input_name,
                'type': input['type'],
                'link': input['link'],
                'slot_index': input['index']
            })
        node_info['inputs'].sort(key=lambda x: x['slot_index'])

        for output_name, output in node['outputs'].items():
            node_info['outputs'].append({
                'name': output_name,
                'type': output['type'],
                'links': output['links'],
                'slot_index': output['index']
            })
        node_info['outputs'].sort(key=lambda x: x['slot_index'])

        parameter_list = list(node['parameters'].values())
        parameter_list.sort(key=lambda x: x['index'])
        for parameter in parameter_list:
            node_info['widgets_values'].append(parameter['value'])

        workflow['nodes'].append(node_info)

    for link_id, link in link_dict.items():
        workflow['links'].append([link_id] + link)

    return workflow


def parse_workflow_to_markdown(workflow):
    markdown = ''
    type_list = []
    node_dict = {}
    link_dict = {}

    markdown += '- Nodes:\n'
    for node_info in workflow['nodes']:
        node_id = int(node_info['id'])
        node_type = node_info['type']
        node_name = f'N{node_id}'
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
                        parameter['value'] = f'"{parameter["value"]}"'
            elif isinstance(node_info['widgets_values'], dict):
                for parameter_name, parameter_value in node_info['widgets_values'].items():
                    assert parameter_name in node_template['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                    if isinstance(parameter_value, str):
                        parameter_value = parameter_value.replace('\n', ' ')
                        parameter_value = f'"{parameter_value}"'
                    node_template['parameters'][parameter_name]['value'] = parameter_value
            else:
                raise ValueError(f'widgets_values should be a list or dict in node {node_type}')

        markdown += f'    - {node_name}:\n        - node_type: "{node_type}"\n'
        for key, value in node_template['parameters'].items():
            markdown += f'        - {key}: {value["value"]}\n'

        node_input = []
        if 'inputs' in node_info:
            node_input = [(item['name'], item['type'], item['link']) for item in node_info['inputs']]
        node_output = []
        if 'outputs' in node_info:
            node_output = [(item['name'], item['type'], item['links']) for item in node_info['outputs']]
        node_dict[node_id] = {'name': node_name, 'input': node_input, 'output': node_output}

    markdown += '\n- Links:\n'
    for link_info in workflow['links']:
        link_id, source_id, source_output, target_id, target_input, link_type = link_info
        link_dict[link_id] = {'variable': None, 'source': source_id, 'target': target_id}

        if source_id in node_dict and target_id in node_dict:
            link_name = f'L{link_id}'
            source_name = f'N{source_id}'
            target_name = f'N{target_id}'
            source_output = node_dict[source_id]['output'][source_output][0].lower()
            target_input = node_dict[target_id]['input'][target_input][0].lower()
            markdown += f'    - {link_name}: {source_name}.{source_output} -> {target_name}.{target_input}\n'

    return markdown


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


def parse_code_to_wfcode(code):
    object_dict = {}  # Map variable names to node types or node instances
    tensor_dict = {}  # Map tensor names to node instances
    addnode=""
    connectnode=""
    invokenode=""
    tree_root = ast.parse(code)
    for tree_node in tree_root.body:
        code_line = ast.unparse(tree_node).strip()

        if isinstance(tree_node, ast.Assign):
            assign_node = tree_node
            call_node = assign_node.value
            if not isinstance(call_node, ast.Call):
                continue  # Skip non-call assignments

            func = call_node.func
            if isinstance(func, ast.Name):
                func_name = func.id
            else:
                raise ValueError(f"Unexpected function type in code line: {code_line}")

            # Collect outputs
            target_list = assign_node.targets
            variable_list = []
            for target in target_list:
                if isinstance(target, ast.Name):
                    variable_list.append(target.id)
                elif isinstance(target, ast.Tuple):
                    for element in target.elts:
                        variable_list.append(element.id)
                else:
                    raise ValueError(f"Unexpected target type in code line: {code_line}")

            # Collect function parameters
            parameter_dict = {}
            for keyword in call_node.keywords:
                if isinstance(keyword.value, ast.Name):
                    parameter_dict[keyword.arg] = keyword.value.id
                elif isinstance(keyword.value, ast.Constant):
                    parameter_dict[keyword.arg] = repr(keyword.value.value)
                elif isinstance(keyword.value, ast.UnaryOp):
                    if isinstance(keyword.value.op, ast.USub):
                        sign = -1
                    parameter_dict[keyword.arg] = repr(sign * keyword.value.operand.value)
                else:
                    raise ValueError(f"Unexpected keyword type in code line: {code_line}")

            if func_name not in object_dict:
                # Node instantiation
                object_dict[variable_list[0]] = func_name  # Map variable name to node type
                params_str = ", ".join(f'"{k}": {v}' for k, v in parameter_dict.items())
                addnode += f'workflow.add_node("{unsafify_var_name(variable_list[0])}", "{unsafify_var_name(func_name)}", {{{params_str}}})\n'
            else:
                # Node invocation (linking)
                node_instance = func_name
                # For each input parameter that is a variable, connect it
                for param_key, param_value in parameter_dict.items():
                    if param_value in tensor_dict:
                        connectnode += f'workflow.connect("{unsafify_var_name(param_value)}", "{unsafify_var_name(node_instance)}", "{param_key}")\n'
                    else:
                        # Set parameter value directly
                        pass #workflow_code += f'workflow.set_param("{node_instance}", "{param_key}", {param_value})\n'
                # Handle outputs
                outputs_str = ", ".join(f'"{unsafify_var_name(v)}"' for v in variable_list)
                invokenode += f'workflow.invoke_node([{outputs_str}], "{unsafify_var_name(node_instance)}")\n'
                # Map outputs to tensors
                for output_var in variable_list:
                    tensor_dict[output_var] = output_var  # Map output variable to itself

        else:
            # Handle other node types if needed
            continue
    workflow_code = "# Add Node\n" + addnode + "\n# Invoke Node\n" + invokenode + "\n# Link Node\n" + connectnode
    return workflow_code

def parse_wfcode_to_code(wfcode):
    nodes = {}
    invokes = {}
    connects = []
    lines = wfcode.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('workflow.add_node'):
            # Parse add_node
            # Example: workflow.add_node("vaedecode_8", "VAEDecode", {})
            m = re.match(r'workflow\.add_node\("([^"]+)", "([^"]+)", (.+)\)', line)
            if m:
                node_id = m.group(1)
                node_type = m.group(2)
                params_str = m.group(3)
                # Evaluate params_str to a dict
                params_dict = ast.literal_eval(params_str)
                nodes[node_id] = {'type': node_type, 'parameters': params_dict}
        elif line.startswith('workflow.invoke_node'):
            # Parse invoke_node
            # Example: workflow.invoke_node(["latent_5"], "emptylatentimage_5")
            m = re.match(r'workflow\.invoke_node\((.+), "([^"]+)"\)', line)
            if m:
                outputs_str = m.group(1)
                node_id = m.group(2)
                outputs_list = ast.literal_eval(outputs_str)
                invokes[node_id] = outputs_list
        elif line.startswith('workflow.connect'):
            # Parse connect
            # Example: workflow.connect("clip_4", "cliptextencode_7", "clip")
            m = re.match(r'workflow\.connect\("([^"]+)", "([^"]+)", "([^"]+)"\)', line)
            if m:
                source_node_id = m.group(1)
                target_node_id = m.group(2)
                input_name = m.group(3)
                connects.append((source_node_id, target_node_id, input_name))
    # Now, build inputs for each node
    inputs = {}
    for source_node_id, target_node_id, input_name in connects:
        if target_node_id not in inputs:
            inputs[target_node_id] = {}
        inputs[target_node_id][input_name] = source_node_id
    # Now, generate code
    code_lines = []
    code_lines.append('# create nodes by instantiation')
    for node_id in nodes:
        node_type = nodes[node_id]['type']
        params = nodes[node_id]['parameters']
        # Convert params dict to string
        params_str_list = []
        for key, value in params.items():
            # For strings, we use triple quotes
            if isinstance(value, str):
                param_value_str = '"""{}"""'.format(value)
            else:
                param_value_str = repr(value)
            params_str_list.append('{}={}'.format(key, param_value_str))
        params_str = ', '.join(params_str_list)
        code_line = '{} = {}({})'.format(safe_var_name(node_id), safe_var_name(node_type), params_str)
        code_lines.append(code_line)
    code_lines.append('')
    code_lines.append('# link nodes by invocation')
    # Now, for each node that is invoked, generate the call
    # We need to keep track of the variables assigned to outputs
    for node_id in invokes:
        outputs_list = invokes[node_id]
        outputs_list = [safe_var_name(ops) for ops in outputs_list]
        if len(outputs_list) == 1:
            outputs_str = outputs_list[0]
        else:
            outputs_str = ', '.join(outputs_list)
            outputs_str = '{}'.format(outputs_str)
        # Build input arguments
        if node_id in inputs:
            input_args_list = []
            for input_name, source_node_id in inputs[node_id].items():
                input_args_list.append('{}={}'.format(safe_var_name(input_name), safe_var_name(source_node_id)))
            input_args_str = ', '.join(input_args_list)
        else:
            input_args_str = ''
        if input_args_str:
            code_line = '{} = {}({})'.format(outputs_str, safe_var_name(node_id), input_args_str)
        else:
            code_line = '{} = {}()'.format(outputs_str, safe_var_name(node_id))
        code_lines.append(code_line)
    # Join code lines
    code = '\n'.join(code_lines)
    return code

def fetch_name_by_index(output_dict, index):
    for name, info in output_dict.items():
        if info['index'] == index:
            return name
    raise ValueError(f'Output with index {index} not found')

def parse_wfcode_to_workflow(code):
    node_count = 0
    link_count = 0
    object_dict = {}
    tensor_dict = {}
    node_dict = {}
    link_dict = {}

    tree_root = ast.parse(code)
    code_lines = code.split('\n')
    node_line_map = {node: code_lines[node.lineno - 1].strip() for node in ast.walk(tree_root) if hasattr(node, 'lineno')}
    
    for tree_node in tree_root.body:
        if not isinstance(tree_node, ast.Expr):
            continue

        expr_node = tree_node.value
        if not isinstance(expr_node, ast.Call):
            continue

        func_node = expr_node.func
        if not isinstance(func_node, ast.Attribute):
            continue
        
        code_line = node_line_map.get(tree_node, '')

        if isinstance(func_node.value, ast.Name) and func_node.value.id == 'workflow':
            method_name = func_node.attr
            if method_name == 'add_node':
                # Process adding node
                args = expr_node.args
                if len(args) != 3:
                    raise ValueError(f'Code line "{code_line}": add_node requires 3 arguments, got {len(args)}')

                # Extract node_name
                node_name_node = args[0]
                if isinstance(node_name_node, ast.Constant):
                    node_name = node_name_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected node_name as a constant string')

                # Extract node_type
                node_type_node = args[1]
                if isinstance(node_type_node, ast.Constant):
                    node_type = node_type_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected node_type as a constant string')

                # Extract parameters_dict
                parameters_node = args[2]
                if isinstance(parameters_node, ast.Dict):
                    parameters_dict = {}
                    for key_node, value_node in zip(parameters_node.keys, parameters_node.values):
                        if isinstance(key_node, ast.Constant):
                            key = key_node.value
                        else:
                            raise ValueError(f'Code line "{code_line}": Expected parameter key as a constant')
                        if isinstance(value_node, ast.Constant):
                            value = value_node.value
                        else:
                            raise ValueError(f'Code line "{code_line}": Expected parameter value as a constant')
                        parameters_dict[key] = value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected parameters_dict as a dictionary')

                # Now, we need to create a node
                template_name = f'{unsafify_var_name(node_type)}.json'

                if template_name in os.listdir('./dataset/docs/template/'):
                    template_path = f'./dataset/docs/template/{template_name}'
                    with open(template_path, 'r') as template_file:
                        node = json.load(template_file)

                    # Now set parameters
                    for parameter_name, parameter_value in parameters_dict.items():
                        parameter_value = unsafify_var_name(parameter_value) if isinstance(parameter_value, str) else parameter_value
                        assert parameter_name in node['parameters'], f'parameter {parameter_name} not found in node {node_type}'
                        node['parameters'][parameter_name]['value'] = parameter_value if isinstance(parameter_value, str) else parameter_value

                    # Assign an id to the node
                    node_count +=1
                    node_id = node_count
                    node['id'] = node_id
                    object_dict[node_name] = node_id
                    node_dict[node_id] = node

                else:
                    raise ValueError(f'Code line "{code_line}": Template for node type {node_type} not found')

            elif method_name == 'invoke_node':
                # Process invoking node
                args = expr_node.args
                if len(args) !=2:
                    raise ValueError(f'Code line "{code_line}": invoke_node requires 2 arguments, got {len(args)}')
                # Extract output_variable_list
                output_list_node = args[0]
                if isinstance(output_list_node, (ast.List, ast.Tuple)):
                    output_variable_list = []
                    for elt in output_list_node.elts:
                        if isinstance(elt, ast.Constant):
                            output_variable_list.append(elt.value)
                        else:
                            raise ValueError(f'Code line "{code_line}": Expected output variable as a constant string')
                else:
                    raise ValueError(f'Code line "{code_line}": Expected output_variable_list as a list or tuple')

                # Extract node_name
                node_name_node = args[1]
                if isinstance(node_name_node, ast.Constant):
                    node_name = node_name_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected node_name as a constant string')

                # Now, we need to find the node_id
                assert node_name in object_dict, f'Code line "{code_line}": Node {node_name} not found'
                node_id = object_dict[node_name]
                node = node_dict[node_id]

                # For each output variable, we need to record it in tensor_dict
                for output_index, variable in enumerate(output_variable_list):
                    tensor_dict[variable] = (node_id, output_index)

            elif method_name == 'connect':
                # Process connecting nodes
                args = expr_node.args
                if len(args) != 3:
                    raise ValueError(f'Code line "{code_line}": connect requires 3 arguments, got {len(args)}')
                # Extract from_variable
                from_var_node = args[0]
                if isinstance(from_var_node, ast.Constant):
                    from_variable = from_var_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected from_variable as a constant string')

                # Extract to_node_name
                to_node_name_node = args[1]
                if isinstance(to_node_name_node, ast.Constant):
                    to_node_name = to_node_name_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected to_node_name as a constant string')

                # Extract to_input_name
                to_input_name_node = args[2]
                if isinstance(to_input_name_node, ast.Constant):
                    to_input_name = to_input_name_node.value
                else:
                    raise ValueError(f'Code line "{code_line}": Expected to_input_name as a constant string')

                # Now, we need to find the node_id of to_node_name
                assert to_node_name in object_dict, f'Code line "{code_line}": Node {to_node_name} not found'
                to_node_id = object_dict[to_node_name]
                to_node = node_dict[to_node_id]

                # Check that the input exists in the node
                assert to_input_name in to_node['inputs'] or to_input_name in node['parameters'], f'Input {to_input_name} not found in node {to_node_name}.'

                # Get the input slot index and type
                if to_input_name in to_node['inputs']:
                    input_info = to_node['inputs'][to_input_name]
                else:
                    # If it's in parameters, move it to inputs
                    to_node['inputs'][to_input_name] = to_node['parameters'][to_input_name]
                    input_info = to_node['inputs'][to_input_name]

                input_index = input_info['index']
                link_type = input_info['type']

                # Now, find the source of the from_variable
                assert from_variable in tensor_dict, f'Code line "{code_line}": Variable {from_variable} is used before defined.'

                from_node_id, output_index = tensor_dict[from_variable]
                from_node = node_dict[from_node_id]
                # Get output name
                output_name = fetch_name_by_index(from_node['outputs'], output_index)
                output_info = from_node['outputs'][output_name]
                output_type = output_info['type']

                # Check type compatibility
                if link_type != "CONDITIONING":
                    assert link_type == output_type, f'Code line "{code_line}": Type mismatch between {link_type} and {output_type}'
                
                # Create a link
                link_count +=1
                link_id = link_count
                link = [from_node_id, output_index, to_node_id, input_index, link_type]
                link_dict[link_id] = link
                input_info['link'] = link_id
                output_info['links'].append(link_id)

    # Build the workflow
    workflow = {
        'nodes': [],
        'links': [],
        'groups': [],
        'config': {},
        'extra': {},
        'version': '0.4'
    }

    for node_id, node in node_dict.items():
        node_info = {
            'id': node_id,
            'type': node['type'],
            'inputs': [],
            'outputs': [],
            'widgets_values': [],
        }

        for input_name, input in node['inputs'].items():
            node_info['inputs'].append({
                'name': input_name,
                'type': input['type'],
                'link': input.get('link', None),
                'slot_index': input['index']
            })
        node_info['inputs'].sort(key=lambda x: x['slot_index'])

        for output_name, output in node['outputs'].items():
            node_info['outputs'].append({
                'name': output_name,
                'type': output['type'],
                'links': output['links'],
                'slot_index': output['index']
            })
        node_info['outputs'].sort(key=lambda x: x['slot_index'])

        parameter_list = list(node['parameters'].values())
        parameter_list.sort(key=lambda x: x['index'])
        for parameter in parameter_list:
            node_info['widgets_values'].append(parameter['value'])

        workflow['nodes'].append(node_info)

    for link_id, link in link_dict.items():
        workflow['links'].append([link_id] + link)

    return workflow

def parse_code_to_nature_code(code):

    # Extract comments and their line numbers
    def extract_comments(code):
        comments = {}
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for toknum, tokval, start, _, _ in tokens:
            if toknum == tokenize.COMMENT:
                lineno = start[0]
                comments.setdefault(lineno, []).append(tokval.rstrip())
        return comments

    comments = extract_comments(code)
    tree = ast.parse(code)

    # Helper function to get the source code representation of a value
    def get_value_repr(value):
        if isinstance(value, ast.Constant):
            return repr(value.value)
        elif isinstance(value, ast.Name):
            return value.id
        elif isinstance(value, ast.Call):
            func_name = get_func_name(value.func)
            args = [get_value_repr(arg) for arg in value.args]
            kwargs = [f"{kw.arg} is {get_value_repr(kw.value)}" for kw in value.keywords]
            params = args + kwargs
            return f"{func_name}({', '.join(params)})"
        else:
            return ast.unparse(value)

    # Helper function to get the function name from a Call node
    def get_func_name(func):
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return get_func_name(func.value) + '.' + func.attr
        else:
            return ''

    # Build a mapping from line numbers to output lines
    output_lines = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            lineno = node.lineno
            targets = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                elif isinstance(target, ast.Tuple):
                    targets.extend([elt.id for elt in target.elts if isinstance(elt, ast.Name)])

            targets_str = ', '.join(targets)
            value = node.value
            if isinstance(value, ast.Call):
                func_name = get_func_name(value.func)
                params = []
                params.extend([get_value_repr(arg) for arg in value.args])
                params.extend([f"{kw.arg} is {get_value_repr(kw.value)}" for kw in value.keywords])

                if params:
                    params_str = ', '.join(params)
                    line = f"{targets_str} is {func_name} with the parameters of ({params_str})"
                else:
                    line = f"{targets_str} is {func_name}()"
                output_lines[lineno] = line

    # Build the final output by interleaving comments and code
    code_lines = code.split('\n')
    final_output = []
    for lineno, _ in enumerate(code_lines, start=1):
        if lineno in comments:
            for comment in comments[lineno]:
                final_output.append(comment)
        if lineno in output_lines:
            final_output.append(output_lines[lineno])

    return '\n'.join(final_output)

def parse_nature_code_to_code(nature_code):
    import re

    code_lines = []
    lines = nature_code.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # Keep comments and empty lines as is
            code_lines.append(line)
            continue

        # Match lines like: var is Func with the parameters of (params)
        # or var is Func()
        assignment_pattern = r'(.+?)\s+is\s+(\w+)(.*)'
        m = re.match(assignment_pattern, line)
        if m:
            targets = m.group(1).strip()
            func_name = m.group(2).strip()
            rest = m.group(3).strip()

            params = ''
            # Check for 'with the parameters of' and extract parameters
            if rest.startswith('with the parameters of'):
                params_str = rest[len('with the parameters of'):].strip()
                # Replace ' is ' with '=' in parameters
                params = params_str.strip('()').replace(' is ', '=')
            elif rest.startswith('(') and rest.endswith(')'):
                params = rest.strip('()').replace(' is ', '=')

            # Construct the code line
            code_line = f"{targets} = {func_name}({params})"
            code_lines.append(code_line)
        else:
            # If the line doesn't match, keep it as is
            code_lines.append(line)

    return '\n'.join(code_lines)