import os
import json
import sys
current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(current_directory)

from utils.parser import parse_workflow_to_code, parse_workflow_to_markdown


def process_workflow_files(raw_path, code_path, md_path):
    # Iterate through all JSON workflow files in the raw directory
    for filename in os.listdir(raw_path):
        if filename.endswith(".json"):
            raw_file = os.path.join(raw_path, filename)
            base_name = os.path.splitext(filename)[0]

            # Load the raw workflow
            with open(raw_file, 'r') as file:
                workflow = json.load(file)

            # Parse the workflow to code
            code = parse_workflow_to_code(workflow)
            code_file = os.path.join(code_path, f"{base_name}.py")
            with open(code_file, 'w') as file:
                file.write(code)

            # Parse the workflow to markdown
            markdown = parse_workflow_to_markdown(workflow)
            md_file = os.path.join(md_path, f"{base_name}.md")
            with open(md_file, 'w') as file:
                file.write(markdown)

if __name__ == "__main__":
    raw_directory = "/path/to/wokflowdir/"
    code_directory = "/path/to/wokflowdir/code"
    md_directory = "/path/to/wokflowdir/md"

    # Ensure the code and md directories exist
    os.makedirs(code_directory, exist_ok=True)
    os.makedirs(md_directory, exist_ok=True)

    # Process the workflows
    process_workflow_files(raw_directory, code_directory, md_directory)
