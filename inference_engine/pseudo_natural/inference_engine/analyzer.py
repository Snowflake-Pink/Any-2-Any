from bs4 import BeautifulSoup
from inference_engine.pseudo_natural.utils.function import safe_extract_from_soup
import yaml
import os
with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
USE_COMFYBENCH_WORKFLOW = config['use_comfybench_workflow']
workspace = "workflow_comfybench" if USE_COMFYBENCH_WORKFLOW else "workflow"
files = [file.split(".py")[0] for file in os.listdir(f"dataset/{workspace}/code")]

analyzer_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now, you are provided with a user query describing the required workflow. Your task is to analyze the query and provide an outline of the key nodes needed and their roles in the workflow.

The user query is as follows:

{query}

Based on the description, point out the key points behind the requirements (e.g. main object, specific style, target resolution, etc.) and the expected paradigm of the workflow (e.g., text-to-image, image-to-image, image-to-video, etc.) and the corresponding node types required to accomplish the workflow paradigm. 
You are not required to provide the code for the workflow. Please make sure your answers are clear and concise within a single paragraph.
Besides the single paragraph description, you should also select one of the most similar key word that match the query from {files}.

Your description should be enclosed with "<description>" tag. For example: <description> The user query describes a text-to-image workflow focused on generating a high-quality image. </description>.
Your selection of the key word should be enclosed with "<keyword>" tag. For example: <keyword> text_to_image <\keyword>.
'''


def get_analyzer_inference_engine_prompt(query: str):
    query_content = query
    prompt_text = analyzer_prompt.format(
        query=query_content,
        files=files
    )
    return prompt_text


def parse_analyzer_inference_engine_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    keyword = safe_extract_from_soup(soup, 'keyword')
    description = safe_extract_from_soup(soup, 'description')
    return keyword.strip('"'), description.strip()
