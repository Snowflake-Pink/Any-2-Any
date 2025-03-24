from bs4 import BeautifulSoup
from utils.parser import parse_nature_code_to_code
from inference_engine.pseudo_natural.utils.function import safe_extract_from_soup
import re
import os

adapter_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now, you are provided a Workspace containing the code of nodes and required to create a **complete and correct** ComfyUI workflow by linking the nodes to finish the following task:

{query}

## Workspace

*Note that brackets of nodes are represented by words of their names in codelines (e.g. LeftBracketComfy3DRightBracket_TripoSR represent the node [Comfy3D] TripoSR), so you need to be careful to match them with the corresponding node knowledge.
The code and node knowledge of the current node instantiation code you are working on are presented as follows:

{node_code}

{node_knowledge}

## Link

Based on the current nodes and their corresponding knowledge, you should link the nodes together.
For example, if here are the codes in the workflow :
<code> 
# create nodes by instantiation
vaeencodeforinpaint_12 is VAEEncodeForInpaint with the parameters of (grow_mask_by is 16)
checkpointloadersimple_4 is CheckpointLoaderSimple with the parameters of (ckpt_name is 'dreamshaper_8.safetensors')
vaeloader_70 is VAELoader with the parameters of (vae_name is 'vae-ft-mse-840000-ema-pruned.safetensors')
checkpointloadersimple_25 is CheckpointLoaderSimple with the parameters of (ckpt_name is 'dreamshaper_8Inpainting.safetensors')
loadimage_78 is LoadImage with the parameters of (image is 'iceberg.jpg')
cliptextencode_7 is CLIPTextEncode with the parameters of (text is 'illustration, painting, text, watermark, copyright, signature, notes')
ksampler_21 is KSampler with the parameters of (seed is 1, control_after_generate is 'fixed', steps is 20, cfg is 7, sampler_name is 'dpmpp_2m', scheduler is 'karras', denoise is 1)
imagepadforoutpaint_11 is ImagePadForOutpaint with the parameters of (left is 256, top is 0, right is 256, bottom is 0, feathering is 0)
cliptextencode_6 is CLIPTextEncode with the parameters of (text is 'an image of iceberg')
saveimage_79 is SaveImage with the parameters of (filename_prefix is 'ComfyUI')
vaedecode_23 is VAEDecode()
</code> 

then the following is what you need to generate (start with the comment '# link nodes by invocation' and then link the nodes):
<code> 
# link nodes by invocation
model_4, clip_4, vae_4 is checkpointloadersimple_4()
model_25, clip_25, vae_25 is checkpointloadersimple_25()
conditioning_7 is cliptextencode_7 with the parameters of (clip is clip_4)
conditioning_6 is cliptextencode_6 with the parameters of (clip is clip_4)
vae_70 is vaeloader_70()
image_78, mask_78 is loadimage_78()
image_11, mask_11 is imagepadforoutpaint_11 with the parameters of (image is image_78)
latent_12 is vaeencodeforinpaint_12 with the parameters of (pixels is image_11, vae is vae_70, mask is mask_11)
latent_21 is ksampler_21 with the parameters of (model is model_25, positive is conditioning_6, negative is conditioning_7, latent_image is latent_12)
image_23 is vaedecode_23 with the parameters of (samples is latent_21, vae is vae_70)
result_79 is saveimage_79 with the parameters of (images is image_23)
</code> 

First, you should provide your Python code to formulate the updated workflow. Your code must meet the following rules:

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated.
2. Each instantiated node should be invoked only once. You should instantiate another node if you need to reuse the same function.
3. Avoid reusing the same variable name. For example: "value_1 is node_1 with the parameters of (value_1)" is not allowed because the output "value_1" overrides the input "value_1".
4. Avoid nested calls in a single code line. For example: "output_2 is node_2 with the parameters of (input_1, node_1())" should be separated into "output_1 is node_1() and output_2 is node_2 with the parameters of (input_1, output_1)". Another example is: "output_2 is node_2 with the parameters of (node_1()[0], node_1()[1])" should be separated into "output_1, output_2 is node_1() and output_2 is node_2 with the parameters of (output_1, output_2)". 
5. You should not modify the variable name id in the current workflow if there is no error.
6. You should not only generate the linked code but also *remain* and *do not modify* the code and the comment in workspace in your output.
7. Avoid using index to obtain variable names. For example: "output_2 is node_1()[1]" is not allowed and it should be "output_1, output_2 = node_1()" according to the node_1's output type.
8. You should make sure your link matches the required input or output types for both nodes. If the type match fails, you should modify the workflow, you may need to relink or add some nodes to make the workflow make sense and accomplish the task. For example, 'cliptextencode(clip=clip_vision)' is not allowed because type mismatch between CLIP_VISION and CLIP. Another example is, 'image=pose_keypoint' is also not allowed because type mismatch between POSE_KEYPOINT and IMAGE
9. When linking the node, you must extract all of its outputs, not just the ones currently needed. This ensures that the full range of outputs is available for potential use in later stages of the workflow.
10. Make sure that the nodes in link period are already defined.
11. Make sure that each node has the inputs it requires reference to the node knowledge.
12. Do not link more than one output to a same port.
13. When linking nodes, make sure that there will not be same variable name.
14. When linking the nodes, make sure the capitalization and spelling of node names are consistent with the ones in addnode part.

*Your code should be enclosed with "<code>" tag.* For example: <code> output_1 is node_1() </code>.
*Make sure you do not include comments about your action among your linked codes.
*Make sure your output code both initiates and links all nodes (including the saving node), and the linked ports have matched type.

After that, you should provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your code and description with the required format.
'''

def read_md(path):
    with open(path,'r') as f:
        return f.read()

def unsafify_var_name(name):
    return name.replace('LeftBracket', '[').replace('RightBracket', ']').replace('LeftParen', '(').replace('RightParen', ')').replace('Space',' ')

def get_node_knowledge(node_code, md_dir="./dataset/docs/node", strict=True):
    node_knowledge = ""
    node_names = re.findall(r'=\s*([a-zA-Z_][a-zA-Z_0-9\-]*)\s*\(', parse_nature_code_to_code(node_code))
    print("--")
    print(node_names)
    node_names = [unsafify_var_name(name).replace(" ","").replace("_","").replace("-","").lower() for name in node_names]
    print(node_names)
    # print(node_names)
    all_nodes = [(name.split(".")[0].replace(" ","").replace("-","").replace("_","").replace("(","").replace(")","").lower(), name) for name in os.listdir(md_dir)]
    # print(all_nodes)
    for name in node_names:
        if strict:
            for node in all_nodes:
                if name == node[0]:
                    node_knowledge+="Node " + read_md(os.path.join(md_dir, node[1])) + "\n"
                    break
        else:
            for node in all_nodes:
                if name in node[0]:
                    node_knowledge+="Node " + read_md(os.path.join(md_dir, node[1])) + "\n"
                    break
    return f"<node knowledge>\n{node_knowledge}\n</node knowledge>\n\n"

def get_linker_inference_engine_prompt(query, node_code):
    query_content = query
    node_code = f'<code>\n{node_code}\n</code>\n\n' 
    node_knowledge = get_node_knowledge(node_code)
    
    prompt_text = adapter_prompt.format(
        query=query_content,
        node_code=node_code,
        node_knowledge=node_knowledge,
    )
    return prompt_text


def parse_linker_inference_engine_response(response):
    soup = BeautifulSoup(response, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return code, description
