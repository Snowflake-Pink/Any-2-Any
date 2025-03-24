import io
import os
import yaml
import json
import argparse
import pandas as pd
import re
import cv2
import base64
from PIL import Image
from bs4 import BeautifulSoup
from utils.parser import parse_code_to_workflow, parse_markdown_to_workflow
from utils.comfy import execute_workflow
from utils.llm import invoke_vision



t2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-image generation task. You should be  **tolerant** to the quality of the generation result, and focus on the consistency with the instruction. (focus on whether the output reasonably satisfies the core intent of the instruction (even with blurriness, discrepancies or quality issues), rather than enforcing word-by-word accuracy.)


The task instruction is described as: {instruction}

The given image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a cat in an astronaut suit, which is consistent with the instruction. The wall is white, which is different from the "green wall" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-image generation task. You should be **tolerant** to the quality of the generation result, and focus on the consistency with the the instruction. (focus on whether the output reasonably satisfies the core intent of the instruction (even with blurriness, discrepancies or quality issues), rather than enforcing word-by-word accuracy.)

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The second image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result keeps the structure of the input reference, but the car is not removed, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


t2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-video generation task. You should be **tolerant** to the quality of the generation result, and focus on the consistency with the instruction. (focus on whether the output reasonably satisfies the core intent of the instruction (even with blurriness, discrepancies or quality issues), rather than enforcing word-by-word accuracy or the motion of the frame.)

The task instruction is described as: {instruction}

The given {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, please dicard the duration and frames per second. 

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a walking robot, which is consistent with the instruction. However, the scene is a street, which is different from the "forest" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-video generation task. You should be  **tolerant** to the quality of the generation result, and focus on the consistency with the instruction. (focus on whether the output reasonably satisfies the core intent of the instruction (even with blurriness, discrepancies or quality issues), rather than enforcing word-by-word accuracy or the motion of the frame.)

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, please dicard the duration and frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result contains a moving car, which is consistent with the instruction. However, it fails to follow the style of the input reference.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


v2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a video-to-video generation task. You should be  **tolerant** to the quality of the generation result, and focus on the consistency with the instruction. (focus on whether the output reasonably satisfies the core intent of the instruction (even with blurriness, discrepancies or quality issues), rather than enforcing word-by-word accuracy.)

The task instruction is described as: {instruction}

The first {reference_frame_count} images are the frames sampled from the input reference, with an actual resolution of {reference_resolution}. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, please dicard the duration and frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result improves the resolution of the input reference. However, it fails to convert the input inference into an oil painting style, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image = image.resize(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image


def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info


def load_video_mp4(video_path: str, size_limit: tuple[int, int] = (512, 512), frame_limit: int = 5) -> tuple[list, dict]:
    base64_frames = []
    meta_info = {}
    video = cv2.VideoCapture(video_path)
    meta_info['width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta_info['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_info['num_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_info['frame_rate'] = int(video.get(cv2.CAP_PROP_FPS))
    meta_info['duration'] = meta_info['num_frames'] / meta_info['frame_rate']

    count = 0
    sample_interval = max(6, meta_info['num_frames'] // frame_limit)
    while video.isOpened():
        status, frame = video.read()
        if not status:
            break
        if count % sample_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            base64_frame = encode_image(image, size_limit)
            base64_frames.append(base64_frame)
        count += 1
    video.release()
    return base64_frames, meta_info

from PIL import Image, ImageSequence
import base64
import io

def load_video(video_path: str, size_limit: tuple[int, int] = (512, 512), frame_limit: int = 5) -> tuple[list, dict]:
    base64_frames = []
    meta_info = {}

    image = Image.open(video_path)
    meta_info['width'] = image.width
    meta_info['height'] = image.height
    meta_info['num_frames'] = image.n_frames
    meta_info['frame_rate'] = None
    meta_info['duration'] = None

    count = 0
    sample_interval = max(1, meta_info['num_frames'] // frame_limit)
    for frame in ImageSequence.Iterator(image):
        if count % sample_interval == 0:
            base64_frame = encode_image(frame, size_limit)
            base64_frames.append(base64_frame)
        count += 1

    return base64_frames, meta_info


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


def parse_evaluation(evaluation: str) -> tuple[str, str]:
    soup = BeautifulSoup(evaluation, 'html.parser')
    analysis = safe_extract_from_soup(soup, 'analysis')
    judgment = safe_extract_from_soup(soup, 'judgment')
    return analysis, judgment


def evaluate_t2i(image_path, instruction) -> bool:
    result_base64_image, result_meta_info = load_image(image_path)

    prompt = t2i_prompt.format(
        instruction=instruction,
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_i2i(ref_image_path, res_image_path, instruction) -> bool:
    reference_base64_image, reference_meta_info = load_image(ref_image_path)
    result_base64_image, result_meta_info = load_image(res_image_path)

    prompt = i2i_prompt.format(
        instruction=instruction,
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_t2v(vid_path, instruction) -> bool:
    result_base64_frames, result_meta_info = load_video(vid_path)

    prompt = t2v_prompt.format(
        instruction=instruction,
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_i2v(ref_image_path, res_vid_path, instruction) -> bool:
    reference_base64_image, reference_meta_info = load_image(ref_image_path)
    result_base64_frames, result_meta_info = load_video(res_vid_path)

    prompt = i2v_prompt.format(
        instruction=instruction,
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_v2v(ref_vid_path, res_vid_path, instruction) -> bool:
    reference_base64_frames, reference_meta_info = load_video_mp4(ref_vid_path)
    result_base64_frames, result_meta_info = load_video(res_vid_path)

    prompt = v2v_prompt.format(
        instruction=instruction,
        reference_frame_count=len(reference_base64_frames),
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for reference_base64_frame in reference_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{reference_base64_frame}"
            }
        })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def process(output_path, instruction, input_path='', modality='') -> dict:
    task_dir = os.path.dirname(output_path)
    returns = {'resolved': False}
    if os.path.exists(f'{task_dir}/llm_evaluation.json'):
        with open(f'{task_dir}/llm_evaluation.json', 'r') as file:
            returns = json.load(file)
        return returns

    try:
        if not os.path.exists(output_path):
            return returns
        if not os.listdir(output_path):
            print('empty output path')
            return returns

        for output in os.listdir(output_path): # for multi-outputs
            output = os.path.join(output_path, output)
            if modality == 't2i':
                analysis, judgment = evaluate_t2i(output, instruction)
            elif modality == 'i2i':
                analysis, judgment = evaluate_i2i(input_path, output, instruction)
            elif modality == 't2v':
                analysis, judgment = evaluate_t2v(output, instruction)
            elif modality == 'i2v':
                analysis, judgment = evaluate_i2v(input_path, output, instruction)
            elif modality == 'v2v':
                analysis, judgment = evaluate_v2v(input_path, output, instruction)
            else:
                print('invalid modality')
                return returns
        
            returns['analysis'] = analysis
            print(analysis)
            if judgment.strip().lower() == 'true':
                returns['resolved'] = True
                break
        
        with open(f'{task_dir}/llm_evaluation.json', 'w') as file:
            json.dump(returns, file, indent=4)
        
    except Exception as error:
        print(f'failed to evaluate generation: {error}')
    finally:
        return returns


with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    proxy_config = config['proxy']

os.environ['http_proxy'] = proxy_config['http_proxy']
os.environ['https_proxy'] = proxy_config['https_proxy']

def get_modality(task_info):
    modality = ''
    for key in ['source', 'target']:
        if task_info[key][0] == 'text':
            modality += 't'
        elif task_info[key][0] == 'image':
            modality += 'i'
        elif task_info[key][0] == 'video':
            modality += 'v'
        modality += '2' if key == 'source' else ''
    print(f'modality: {modality}')
    return modality
        
def main(args):
    with open(args.json_path, 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)
    num_tasks = len(list(metadata.keys()))
    print(num_tasks)
    record = dict()
    missing_directories = 0
    for inference_engine_name in args.inference_engine_name:
        record[inference_engine_name] = dict()
        for task_id, task_info in list(metadata.items()):
            record[inference_engine_name][task_id] = {
                'task_info': task_info,
                'num_runs': 0,
                'num_passes_1': 0,
                'num_passes_2': 0,
                'resolved': 0
            }
    assert os.path.exists(args.comfyui_path), f"Directory does not exist: {args.comfyui_path}"
    for inference_engine_name in args.inference_engine_name:
        print(f'[Evaluation] inference_engine {inference_engine_name}') 
        for task_id in list(metadata.keys()):
            
            query = metadata[task_id]['content']
            modality = get_modality(metadata[task_id])
            prefix = f'{args.save_path}/{inference_engine_name}/task_{task_id}'
            output_prefix = f'{args.save_path}/{inference_engine_name}/task_{task_id}/output'
            if modality[0] == 'i' or modality[0] == 'v':
                inp_img = re.findall(r"`(.*)`",query)[0]
                ref_path = os.path.join(args.comfyui_path, f'input/{inp_img}')

            else:
                ref_path = ''   
            print(f'[Evaluation] task {task_id}')

            # Adding the check for missing directory
            if not os.path.exists(prefix):
                print(f"Directory does not exist: {prefix}")
                missing_directories += 1
                continue

            for run_id in os.listdir(prefix):
                record[inference_engine_name][task_id]['num_runs'] += 1
                checkpoint = os.path.join(prefix, run_id)
                print(f'[Evaluation] checkpoint {checkpoint}')

                # Skip: already evaluated
                output_path = os.path.join(checkpoint, 'output')
                
                if os.path.exists(output_path):
                    record[inference_engine_name][task_id]['num_passes_1'] += 1
                    record[inference_engine_name][task_id]['num_passes_2'] += 1
                    llm_judgement = process(output_path, query, ref_path, modality)
                    if llm_judgement['resolved']:
                        record[inference_engine_name][task_id]['resolved'] += 1
                    print('skipped: already evaluated')
                    continue
      
                # Check: no file
                code_path = os.path.join(checkpoint, 'code.py')
                if not os.path.exists(code_path):
                    print('skipped: no file')
                    continue

                # Check: empty code
                with open(code_path, 'r') as file:
                    code = file.read()
                if code.strip() == '':
                    print('skipped: empty code')
                    continue

                # Check: invalid workflow
                try:
                    workflow = parse_code_to_workflow(code)
                except Exception as error:
                    print('skipped: invalid workflow')
                    continue

                # Check: no file
                json_path = os.path.join(checkpoint, 'workflow.json')
                if not os.path.exists(json_path):
                    print('skipped: no file')
                    continue

                # Check: invalid format
                try:
                    with open(json_path, 'r') as file:
                        workflow = json.load(file)
                except Exception as error:
                    print('skipped: invalid workflow')
                    continue

                # Record: pass 1
                record[inference_engine_name][task_id]['num_passes_1'] += 1

                # Check: execution failure
                try:
                    status, outputs = execute_workflow(workflow)
                except Exception as error:
                    print(error)
                    print('skipped: execution failure')
                    continue

                # Check: invalid status
                if status['status_str'] != 'success':
                    print(f'status: {status['status_str']}')
                    print('skipped: invalid status')
                    continue

                # Save: execution output
                output_path = os.path.join(checkpoint, 'output')
                os.makedirs(output_path, exist_ok=True)
                for file_name, output in outputs.items():
                    file_path = os.path.join(output_path, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(output)

                # Record: pass 2
                record[inference_engine_name][task_id]['num_passes_2'] += 1
                
                llm_judgement = process(output_path, query, ref_path, modality)
                if llm_judgement['resolved']:
                    record[inference_engine_name][task_id]['resolved'] += 1

    summary = {
        'inference_engine Name': [],
        '(Run Level) Pass Rate 1': [],
        '(Run Level) Pass Rate 2': [],
        '(Task Level) Pass Rate 1': [],
        '(Task Level) Pass Rate 2': [],
        '(Task Level) Resolved': [],
        'Task Name': []
    }
    for inference_engine_name, inference_engine_record in record.items():
        num_runs, num_tasks = 0, len(inference_engine_record)
        run_passes_1, run_passes_2 = 0, 0
        task_passes_1, task_passes_2 = 0, 0
        task_resolved = 0
        for task_record in inference_engine_record.values():
            num_runs += task_record['num_runs']
            run_passes_1 += task_record['num_passes_1']
            run_passes_2 += task_record['num_passes_2']
            if task_record['num_passes_1'] > 0:
                task_passes_1 += 1
            if task_record['num_passes_2'] > 0:
                task_passes_2 += 1
            if task_record['resolved'] > 0:
                task_resolved += 1
        
        summary['inference_engine Name'].append(inference_engine_name)
        summary['(Run Level) Pass Rate 1'].append(run_passes_1 / num_runs if num_runs > 0 else 0)
        summary['(Run Level) Pass Rate 2'].append(run_passes_2 / num_runs if num_runs > 0 else 0)
        summary['(Task Level) Pass Rate 1'].append(task_passes_1 / (num_tasks-missing_directories) if num_tasks > 0 else 0)
        summary['(Task Level) Pass Rate 2'].append(task_passes_2 / (num_tasks-missing_directories) if num_tasks > 0 else 0)
        summary['(Task Level) Resolved'].append(task_resolved / (num_tasks-missing_directories) if num_tasks > 0 else 0)
        summary['Task Name'].append(args.save_path.split("/")[-1])
        print(num_runs)
        print(num_tasks)
    summary = pd.DataFrame(summary)
    print(summary.to_string())

    # Print summary of missing directories
    if missing_directories > 0:
        print(f"Total missing directories: {missing_directories}")


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
        '--json_path',
        type=str,
        default="./dataset/query/meta.json"
    )
    parser.add_argument(
        '--comfyui_path',
        type=str,
        default="../ComfyUI"
    )
    args = parser.parse_args()
    main(args)
