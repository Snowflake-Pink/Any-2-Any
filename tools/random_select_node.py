import random

def random_delete_lines(input_file, output_file, delete_ratio):
    if not 0 <= delete_ratio <= 1:
        raise ValueError()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    comment = lines[0]
    lines = lines[1:]

    num_lines_to_keep = int(len(lines) * (1 - delete_ratio))
    lines_to_keep = random.sample(lines, num_lines_to_keep)
    lines_to_keep = [comment]+lines_to_keep

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines_to_keep)

    print(f"{output_file} keep {num_lines_to_keep} line.")

# example usage
input_dir = './workspace/multi_task_set/image_inpaint/gt_node_10'
import os
files = os.listdir(input_dir)
output_dir = './workspace/multi_task_set/image_inpaint/random_keynode'
delete_ratio = 0.3
for file in files:
    random_delete_lines(os.path.join(input_dir,file), os.path.join(output_dir,file), delete_ratio)
