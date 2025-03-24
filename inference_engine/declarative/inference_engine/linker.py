from bs4 import BeautifulSoup
from utils.parser import parse_wfcode_to_code, unsafify_var_name
from inference_engine.declarative.utils.function import safe_extract_from_soup
import re
import os

adapter_prompt = '''
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

Now, you are provided a Workspace containing the code of nodes and required to create a **complete and correct** ComfyUI workflow by linking the nodes to finish the following task:

{query}

## Workspace

The code and node knowledge of the current node instantiation code you are working on are presented as follows:

{node_code}

{node_knowledge}

## Link

Based on the current nodes and their corresponding knowledge, you should link the nodes together. You should focus on the task requirements, consider function of each node and the expected effects when linking.
You should be aware of the input and output types of the nodes, and ensure that they are compatible when linking the nodes. Make sure that the input and output ports of the nodes are correctly connected to maintain the logical flow of the workflow.
When processing `Invoke Node` part, you should invoke all the *outputs* of a node **IN ORDER** according to node knowledge if a node is used, not just the outputs currently needed. DO NOT INVOKE INPUTS OF A NODE.

For example, if here are the codes in the workflow :
<code> 
# Add Node
workflow.add_node("vaeencodeforinpaint_12", "VAEEncodeForInpaint", {{"grow_mask_by": 16}})
workflow.add_node("checkpointloadersimple_4", "CheckpointLoaderSimple", {{"ckpt_name": 'dreamshaper_8.safetensors'}})
workflow.add_node("vaeloader_70", "VAELoader", {{"vae_name": 'vae-ft-mse-840000-ema-pruned.safetensors'}})
workflow.add_node("checkpointloadersimple_25", "CheckpointLoaderSimple", {{"ckpt_name": 'dreamshaper_8Inpainting.safetensors'}})
workflow.add_node("loadimage_78", "LoadImage", {{"image": 'iceberg.jpg'}})
workflow.add_node("cliptextencode_7", "CLIPTextEncode", {{"text": 'illustration, painting, text, watermark, copyright, signature, notes'}})
workflow.add_node("ksampler_21", "KSampler", {{"seed": 1, "control_after_generate": 'fixed', "steps": 20, "cfg": 7, "sampler_name": 'dpmpp_2m', "scheduler": 'karras', "denoise": 1}})
workflow.add_node("imagepadforoutpaint_11", "ImagePadForOutpaint", {{"left": 256, "top": 0, "right": 256, "bottom": 0, "feathering": 0}})
workflow.add_node("cliptextencode_6", "CLIPTextEncode", {{"text": 'an image of iceberg'}})
workflow.add_node("saveimage_79", "SaveImage", {{"filename_prefix": 'ComfyUI'}})
workflow.add_node("vaedecode_23", "VAEDecode", {{}})

# Invoke Node

# Link Node

</code> 

then the following is what you need to generate (remember the comment '# Invoke Node' and '# Link Node') and add on to the original 'add node' code block:
<code> 
# Invoke Node
workflow.invoke_node(["model_4", "clip_4", "vae_4"], "checkpointloadersimple_4")
workflow.invoke_node(["model_25", "clip_25", "vae_25"], "checkpointloadersimple_25")
workflow.invoke_node(["conditioning_7"], "cliptextencode_7")
workflow.invoke_node(["conditioning_6"], "cliptextencode_6")
workflow.invoke_node(["vae_70"], "vaeloader_70")
workflow.invoke_node(["image_78", "mask_78"], "loadimage_78")
workflow.invoke_node(["image_11", "mask_11"], "imagepadforoutpaint_11")
workflow.invoke_node(["latent_12"], "vaeencodeforinpaint_12")
workflow.invoke_node(["latent_21"], "ksampler_21")
workflow.invoke_node(["image_23"], "vaedecode_23")
workflow.invoke_node(["result_79"], "saveimage_79")

# Link Node
workflow.connect("clip_4", "cliptextencode_7", "clip")
workflow.connect("clip_4", "cliptextencode_6", "clip")
workflow.connect("image_78", "imagepadforoutpaint_11", "image")
workflow.connect("image_11", "vaeencodeforinpaint_12", "pixels")
workflow.connect("vae_70", "vaeencodeforinpaint_12", "vae")
workflow.connect("mask_11", "vaeencodeforinpaint_12", "mask")
workflow.connect("model_25", "ksampler_21", "model")
workflow.connect("conditioning_6", "ksampler_21", "positive")
workflow.connect("conditioning_7", "ksampler_21", "negative")
workflow.connect("latent_12", "ksampler_21", "latent_image")
workflow.connect("latent_21", "vaedecode_23", "samples")
workflow.connect("vae_70", "vaedecode_23", "vae")
workflow.connect("image_23", "saveimage_79", "images")
</code> 

First, you should provide your Python code to formulate the updated workflow. Your code must meet the following rules:
1. Each code line should among add (`add_node`) a node, invoke (`invoke_node`) a node and connect. You should not invoke or connect a node before it is added.
2. Each node should be invoked only once after it is added and it can be connect more than once.
3. Do not reuse the same variable name for input and output. For example, `"workflow.invoke_node(["node_1"], "node_1")"` is not allowed because it overwrites the input variable.
4. Avoid making nested calls in a single code line. Split nested calls into separate lines for clarity.
5. Do not modify existing variable names (IDs) in the current workflow if there are no errors. Keep the original names as they appear in the workspace.
6. Your output should include the new additions (invocations and links) but must not modify the existing code lines or comments in the workspace.
7. Do not use indexing to extract output variables. Instead, assign the outputs directly based on the node’s output type. For example, `workflow.invoke_node(["node_1[0]", "node_1[1]"], "node_1")` is not allowed and should be `workflow.invoke_node(["output_1", "output_2"], "node_1")`.
8. In Link Node part, verify that the input and output types of linked nodes are compatible. If there is a mismatch, modify the workflow by relinking nodes or adding necessary nodes to resolve the issue. For example, using `workflow.connect("clip_vision", "cliptextencode_7", "clip")` is incorrect due to a type mismatch, while `workflow.connect("clip_text", "cliptextencode_7", "clip")` is correct.
9. Your output code should consists of three part, the original 'Add node' code block, and the two code block you generate ('Invoke Node' and 'Link Node')
10. Do not make up new ports without having any references. You need to follow the exactly reference code (when it's uni-task, select the most relavant reference;) without changing port name/invoke order unless necessary. For example, you should do workflow.invoke_node(["result_9"], "saveimage_9") if the most relavant reference mentioned it. Also, You must not connect the output produced by invoking a node directly back to its own inputs. Instead, use the invoked outputs for other nodes or later stages in the workflow. This restriction ensures that the workflow maintains a logical flow and prevents recursive feedback loops.
11. When invoking a node, you must extract all of its outputs in order, not just the ones currently needed. This ensures that the full range of outputs is available for potential use in later stages of the workflow.
12. Ensure that every unCLIPConditioning node has a valid conditioning input by connecting it to a non-empty CLIPTextEncode output with a descriptive text prompt.
13. Make sure that each node has the inputs it require but do not link to non-existent input reference to the node knowledge in Link part. For example, workflow.invoke_node(["model_32", "clip_32", "vae_32", "clip_vision_32", "name_string_32"], "unclipcheckpointloader_32") is wrong because an extra input "name_string_32", the correct code line should be workflow.invoke_node(["model_32", "clip_32", "vae_32", "clip_vision_32"], "unclipcheckpointloader_32")
14. Do not link more than one output to a same port.
15. When invoke nodes, make sure that there will not be same variable name (repeated node names are allowed, but if repeat, their id numbers should be different).
16. When invoke and link the nodes, make sure the capitalization and spelling of node names are consistent with the ones in addnode part.
17. If you need to remove background from an image, ensure that the image first undergoes background removal (Image Remove Background (rembg)) before any conversion to RGB format (Images to RGB), to preserve the Alpha channel and avoid errors from attempting to access a non-existent Alpha channel. For example, you should do this: workflow.connect("image_23", "image remove background (rembg)_16", "image"), workflow.connect("image_16", "imagetomask_17", "image"). workflow.connect("image_16", "images to rgb_47", "images")
18. Check carefully the type of the input and output of the nodes, make sure they are compatible. And check carefully the connect part according to the node knowledge.
19. DO NOT SET VIDEO FRAMES MORE THAN 10, IT WILL CAUSE TIMEOUT ERROR. VIDEO SAMPLING STEP SHOULD BE LESS THAN 50 AND GREATER THAN 20 TO IMPROVE QUALITY.
20. If you are required to generate a video (including upscale video frames), use `SaveAnimatedWEBP` node as the lastest node to save the video. Also, make sure you follow the retrieved workflow so that **all CogvideoTextEncode nodes have <Node cliploader's output Clip port> linked to their input Clip port**
21. ** THE LATENT SAMPLES FROM CogVideoImageEncode MUST CONNECT TO THE `image_cond_latents` PORT OF CogVideoSampler INSTEAD OF `samples` PORT**
22. If a node has no output, you can still invoke it using an empty []. For example, `workflow.invoke_node([], "saveanimatedwebp_7")`.
23. If your task is about loading a video and do something on it, i.e. modality is video-to-video, you can just use `VHS_LoadVideo` node to load this video and treat each frame's processing as an batched image-to-image task and finnaly use SaveAnimatedWEBP to save the processed images to video. You should not use CogVideoSampler node to process V2V tasks. (This means you can just replace the `load image` with `load video` and `save image` with `save video` nodes to adapt functions from image-to-image task to video-to-video task)

* Your output code should be enclosed by <code> tag, for example: 
<code>
workflow.add_node("cliptextencode_7", "CLIPTextEncode", {{"text": "an image of iceberg"}})
workflow.invoke_node(["conditioning_7"], "cliptextencode_7")
workflow.connect("conditioning_7", "ksampler_21", "positive")
</code>

*Make sure Do not include comments or explanations about your actions within the code block.
*Make sure your output code adds, invokes and links all nodes (including the saving node). The linked input and output ports must have matching types.
*You should not leave any port unlinked, for example: for an added node:workflow.add_node("cliptextencode_2", "CLIPTextEncode", {{"text": 'a woman'}}), you should invoke it like<workflow.invoke_node(["conditioning_2"], "cliptextencode_2")> and link it like<workflow.connect("conditioning_2", "ksampler_3", "positive")><workflow.connect("clip_29", "cliptextencode_2", "clip")> 
workflow.connect("clip_29", "cliptextencode_2", "clip")

After that, you should provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your code and description with the required format.
'''

def read_md(path):
    with open(path,'r') as f:
        return f.read()

def get_node_knowledge(node_code, md_dir="dataset/docs/node", strict=True):
    node_code = parse_wfcode_to_code(node_code)
    node_knowledge = ""
    node_names = re.findall(r'=\s*([a-zA-Z_][a-zA-Z_0-9\-]*)\s*\(', node_code)
    print("--")
    print(node_names)
    node_names = [unsafify_var_name(name).replace(" ","").replace("_","").replace("-","").replace("(","").replace(")","").lower() for name in node_names]
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
