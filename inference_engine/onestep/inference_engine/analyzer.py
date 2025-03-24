from bs4 import BeautifulSoup
from inference_engine.onestep.utils.function import safe_extract_from_soup
import os
import yaml
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

##introduction

<introduction>
In ComfyUI workflows, the complete pipeline is built in three stages: `add`, `invoke`, and `link`. Each stage has specific responsibilities.

1. Add Node: In this stage, all nodes are instantiated and added to the workflow. This step defines the type of each node and specifies its parameters, such as input images, model checkpoints, or text prompts. The key requirement is that each node must have a unique ID, a node type, and a valid parameter dictionary. Example: `workflow.add_node("cliptextencode_2", "CLIPTextEncode", {{"text": "a beautiful scenery"}})`.

2. Invoke Node: In this stage, the added nodes are executed to perform their computations and produce outputs. The main purpose of invoking nodes is to generate the necessary outputs (e.g., image embeddings, latent variables) that will be used in subsequent stages of the workflow. Each node must be invoked only once. If the same function needs to be reused, a new instance of the node should be created and then invoked. The input variables provided must match the expected input ports of the node to ensure successful execution. Example: workflow.invoke_node(["positive_2"], "cliptextencode_2").

3. Link Node: In this final stage, the nodes are connected to form a complete workflow chain. The outputs of one node are linked to the inputs of another node, ensuring data flows correctly through the pipeline. It is essential to check that the input and output types are compatible; for example, image outputs should be connected to image inputs, and text encodings should be linked to text conditioning inputs. Example: `workflow.connect("latent_3", "vaedecode_8", "samples")`.

These three stages must be executed in order. First, `add_node` initializes all components. Next, `invoke_node` performs the computations. Finally, `link_node` connects the nodes into a coherent workflow. Following these rules ensures the generated workflow is functional and meets the user’s input requirements.

</introduciton>

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
