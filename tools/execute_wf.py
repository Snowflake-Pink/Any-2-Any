from utils.comfy import execute_workflow
import json
import os
checkpoint = "/path/to/workflow/"
with open(os.path.join(checkpoint, "workflow.json"),'r') as f:
    wolkflow = json.load(f)
print(wolkflow)
status, outputs =  execute_workflow(wolkflow)
print(status)

# Check: invalid status
if status['status_str'] != 'success':
    print('skipped: invalid status')
    exit(0)

# Save: execution output
output_path = os.path.join(checkpoint, 'output')
os.makedirs(output_path, exist_ok=True)
for file_name, output in outputs.items():
    file_path = os.path.join(output_path, file_name)
    with open(file_path, 'wb') as file:
        file.write(output)