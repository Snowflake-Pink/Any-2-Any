import os
import json

def merge_tasks(base_dir, extract_count):
    # Path to save the merged meta JSON file
    merged_meta = {}
    
    # Iterate over each task folder in base directory
    for task_folder in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_folder)
        
        if not os.path.isdir(task_path):
            continue
        
        # Check if this is a directory and contains meta.json
        meta_path = os.path.join(task_path, 'meta.json')
        keynode_path = os.path.join(task_path, 'keynode', 'task_01001.py')
        
        if os.path.isfile(meta_path) and os.path.isfile(keynode_path):
            # Load meta.json file
            with open(meta_path, 'r') as f:
                task_data = json.load(f)
            
            # Get tasks based on the extract_count
            extracted_tasks = list(task_data.items())[:extract_count]
            
            # Modify keys and add keynode path
            for key, task_details in extracted_tasks:
                new_key = f"{task_folder}_{key}"
                task_details["keynode_path"] = keynode_path
                merged_meta[new_key] = task_details
        else:
            print(keynode_path)
    
    # Save merged meta JSON
    output_file = os.path.join(base_dir, f'meta_{extract_count}.json')
    with open(output_file, 'w') as f:
        json.dump(merged_meta, f, indent=4, ensure_ascii=False)

    print(f'Merged file saved as: {output_file}')

if __name__ == "__main__":
    merge_tasks('./workspace/multi_task_set', 1)
