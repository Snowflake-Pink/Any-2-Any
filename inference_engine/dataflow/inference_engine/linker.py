from bs4 import BeautifulSoup

from inference_engine.dataflow.utils.function import safe_extract_from_soup
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
vaeencodeforinpaint_12 = VAEEncodeForInpaint(grow_mask_by=16) 
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""") 
vaeloader_70 = VAELoader(vae_name="""vae-ft-mse-840000-ema-pruned.safetensors""") 
checkpointloadersimple_25 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8Inpainting.safetensors""") 
loadimage_78 = LoadImage(image="""iceberg.jpg""") 
cliptextencode_7 = CLIPTextEncode(text="""illustration, painting, text, watermark, copyright, signature, notes""") 
ksampler_21 = KSampler(seed=1, control_after_generate="""fixed""", steps=20, cfg=7, sampler_name="""dpmpp_2m""", scheduler="""karras""", denoise=1) 
imagepadforoutpaint_11 = ImagePadForOutpaint(left=256, top=0, right=256, bottom=0, feathering=0) 
cliptextencode_6 = CLIPTextEncode(text="""an image of iceberg""") 
saveimage_79 = SaveImage(filename_prefix="""ComfyUI""") 
vaedecode_23 = VAEDecode()
</code> 

then the following is what you need to generate (start with the comment '# link nodes by invocation' and then link the nodes):
<code> 
# link nodes by invocation 
model_4, clip_4, vae_4 = checkpointloadersimple_4() 
model_25, clip_25, vae_25 = checkpointloadersimple_25() 
conditioning_7 = cliptextencode_7(clip=clip_4) 
conditioning_6 = cliptextencode_6(clip=clip_4) 
vae_70 = vaeloader_70() 
image_78, mask_78 = loadimage_78() 
image_11, mask_11 = imagepadforoutpaint_11(image=image_78) 
latent_12 = vaeencodeforinpaint_12(pixels=image_11, vae=vae_70, mask=mask_11) 
latent_21 = ksampler_21(model=model_25, positive=conditioning_6, negative=conditioning_7, latent_image=latent_12) 
image_23 = vaedecode_23(samples=latent_21, vae=vae_70) 
result_79 = saveimage_79(images=image_23)
</code> 

First, you should provide your Python code to formulate the updated workflow. Your code must meet the following rules:

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated.
2. Each instantiated node should be invoked only once. You should instantiate another node if you need to reuse the same function.
3. Avoid reusing the same variable name. For example: "value_1 = node_1(value_1)" is not allowed because the output "value_1" overrides the input "value_1".
4. Avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Another example is: "output_2 = node_2(node_1()[0], node_1()[1])" should be separated into "output_1, output_2 = node_1() and output_2 = node_2(output_1, output_2)". 
5. You should not modify the variable name id in the current workflow if there is no error.
6. You should not only generate the linked code but also *remain* and *do not modify* the code and the comment in workspace in your output.
7. Avoid using index to obtain variable names. For example: "output_2 = node_1()[1]" is not allowed and it should be "output_1, output_2 = node_1()" according to the node_1's output type.
8. You should make sure your link matches the required input or output types for both nodes. If the type match fails, you should modify the workflow, you may need to relink or add some nodes to make the workflow make sense and accomplish the task. For example, 'cliptextencode(clip=clip_vision)' is not allowed because type mismatch between CLIP_VISION and CLIP. Another example is, 'image=pose_keypoint' is also not allowed because type mismatch between POSE_KEYPOINT and IMAGE

*Your code should be enclosed with "<code>" tag.* For example: <code> output_1 = node_1() </code>.
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
    node_names = re.findall(r'=\s*([a-zA-Z_][a-zA-Z_0-9\-]*)\s*\(', node_code)
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
