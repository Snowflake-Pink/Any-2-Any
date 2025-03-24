from bs4 import BeautifulSoup

from inference_engine.dataflow.utils.function import safe_extract_from_soup


combiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to instantiate their own workflows nodes.

Now you are required to create a ComfyUI workflow to finish the following task:

The user query is as follows:

{query}

The key points behind the requirements and the expected paradigm of the workflow are analyzed as follows:

{analysis}

## Reference

The code and description of the example workflow you are referring to are presented as follows:

{reference}

## Key Nodes
*Note that brackets of nodes are represented by words of their names in codelines (e.g. LeftBracketComfy3DRightBracketSpaceTripoSR represent the node [Comfy3D] TripoSR), so you need to be careful to match them with the corresponding node knowledge.
The user has provided the following key nodes that *must be included* in the workflow:

{key_nodes}

The corresponding knowledge for the key nodes is as follows:

{key_node_knowledge}

## Workspace

The code of the current nodes instantiation you are working on are presented as follows:

{code}

## Combination and Adaptation

Following the current progress, the step-by-step plan is outlined as follows:

1. Select Key Nodes: Select all nodes in the key nodes.
2. Analyze Knowledge and Queries: Review the information associated with the key nodes and evaluate the query.
3. Construct the Workflow Nodes: Base on the key nodes, choose the rest nodes that can assist in building the workflow from the workspace, followed by selecting nodes from the reference.
4. Adapt to New Inputs: Integrate new values (e.g., input text) into the selected nodes to fulfill the requirements but *do not* modify the model_name or ckpt_name. For example, change 'CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")' to 'CheckpointLoaderSimple(ckpt_name="""model.safetensors""")' is not allowed because this modify the ckpt_name.

In other words, you should merge the nodes instantiation in example workflow with the current nodes instantiation code according to your plan, so that their functions can be combined.

First, you should provide your Python code to initiate all the nodes needed for the updated workflow. Your code must meet the following rules:
Please note that the K-sampler requires negative input, so make sure to have an additional CLIPTextEncode specifically for the negative input, in addition to the one used for the positive input.

## Other Requirements

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated.
2. Each instantiated node should be invoked only once. You should instantiate another node if you need to reuse the same function. 
3. Avoid reusing the same variable name. For example: "value_1 = node_1(value_1)" is not allowed because the output "value_1" overrides the input "value_1".
4. Avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)".
5. Your code should only contain the complete node instantiation part of the workflow, but not include links.
6. Ensure that the provided key nodes are included and necessary extra nodes for the query are added.
7. Make sure all the key nodes are used.
8. Before your output your code, check the validity of all the parameters in node instantiation. You should fill the nodes in valid and sound texts/conditions/models/parameters that could work without any manual adjustment.
8. Please note that the SVD_img2vid_Conditioning for image to video generation will directly provide the positive condition and the negative condition for Ksampler, so make sure there is no extra textencode node provided for the ksampler.
9. Please note that the ImageOnlyCheckpointLoader for image to video generation will provide model and vae which is enough for further steps, so make sure there is no extra node like CheckpointLoaderSimple that provide extra model and vae.


Your code should be enclosed with "<code>" tag. For example: <code> output_1 = node_1() </code>.

After that, you should provide a brief description of the whole set of node instantiations and their intended roles in the workflow.
Your description should be enclosed with "<description>" tag. For example: <description> Added an upscaling module to enhance image resolution. </description>.

Now, provide your code and description with the required format.
'''

def get_generator_inference_engine_prompt(
    code:str,
    query:str,
    analysis:str,
    keyword:str,
    reference:str,
    key_nodes:str,
    key_node_knowledge:str
):

    prompt_text = combiner_prompt.format(
        code=f'<code>\n{code}\n</code>\n\n',
        query=query,
        analysis=analysis,
        reference=reference,
        key_nodes=f'<code>\n{key_nodes}\n</code>\n\n',
        key_node_knowledge=key_node_knowledge
    )
    return prompt_text

def parse_generator_inference_engine_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return code, description