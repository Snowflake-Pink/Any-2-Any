import json
import os

SAVEPATH = "path/to/Any-to-Any/dataset/docs/template"

def dump_json(template, save_path, force=False):
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_path = os.path.join(save_path, template['type'].replace("/","_") + ".json")
    if not os.path.exists(save_path) or force:
        print(f"[INFO] Generating Custom Node Template: {template['type']}")
        with open(save_path,"w") as f:
            json.dump(template, f, indent=4)
import json

def generate_custom_template(class_obj, name, force=False):
    # Get input types dynamically
    input_types = class_obj.INPUT_TYPES()
    
    # Initialize the dynamic template structure
    template = {
        "id": None,
        "type": name,
        "title": None,
        "parameters": {},
        "inputs": {},
        "outputs": {}
    }
    
    # Separate parameters (those with a 'default' value) and inputs
    parameter_index = 0
    input_index = 0

    for input_category in ["required", "optional"]:
        for key, (field_type, *attributes) in input_types.get(input_category, {}).items():
            # Check if the attribute has a 'default' key, indicating it should be a parameter
            if attributes and isinstance(attributes[0], dict) and 'default' in attributes[0] or isinstance(field_type, list):
                # Add to parameters with default value if available
                template["parameters"][key] = {
                    "index": parameter_index,
                    "value": None, # attributes[0].get("default", None)
                    "type" : field_type,
                    "link": None
                }
                parameter_index += 1
            else:
                # Add to inputs without a default value
                template["inputs"][key] = {
                    "index": input_index,
                    "type": field_type,
                    "link": None
                }
                input_index += 1

    # Populate outputs dynamically
    output_types = getattr(class_obj, 'RETURN_TYPES', ())
    output_names = getattr(class_obj, 'RETURN_NAMES', ())
    if len(output_names) == 0:
        for idx, output_type in enumerate(output_types):
            if not isinstance(output_type, (str, int, float, tuple)):
                output_type = str(output_type)  # Convert to string if not hashable
            template["outputs"][output_type] = {
                "index": idx,
                "type": output_type,
                "links": []
            }
    else:
        for idx, (name, output_type) in enumerate(zip(output_names, output_types)):
            template["outputs"][name] = {
                "index": idx,
                "type": output_type,
                "links": []
            }
            
    save_path = SAVEPATH #
    dump_json(template, save_path=save_path, force=force)
    json_to_markdown(template, save_path=save_path.replace("template","node"), force=force)

    return template

def json_to_markdown(json_data, save_path, force=False):
    # Load the JSON data
    data = json_data
    # Prepare Markdown content
    md_content = f"- `{data['type']}`: The {data['type']} node description.\n"
    md_content += "    - Parameters:\n"
    
    # Append parameters
    for param_key, param_value in data["parameters"].items():
        md_content += f"        - `{param_key}`: Type should be `{param_value['type']}`.\n"
    
    md_content += "    - Inputs:\n"
    
    # Append inputs types
    for input_key, input_value in data["inputs"].items():
        md_content += f"        - `{input_key}`: Type should be `{input_value['type']}`.\n"
    
    md_content += "    - Outputs:\n"
    
    # Append outputs types
    for output_key, output_value in data["outputs"].items():
        md_content += f"        - `{output_key}`: Type should be `{output_value['type']}`.\n"
    
    # Write Markdown content to file
    if not os.path.exists(save_path): os.makedirs(save_path)
    save_path = os.path.join(save_path, data['type'].replace("/","_") + ".md")
    if not os.path.exists(save_path) or force:
        print(f"[INFO] Generating Custom Node Knowledge: {data['type']}")
        with open(save_path, 'w') as f:
            f.write(md_content)
            