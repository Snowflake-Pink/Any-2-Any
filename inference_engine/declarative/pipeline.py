import json
import logging
import hashlib
import os
from utils.parser import parse_wfcode_to_workflow, parse_wfcode_to_code, parse_code_to_wfcode
from utils.llm import retrieve_references, invoke_completion, invoke_completion_claude
from utils.comfy import execute_workflow

from inference_engine.declarative.inference_engine.analyzer import get_analyzer_inference_engine_prompt, parse_analyzer_inference_engine_response
from inference_engine.declarative.inference_engine.node_generator import get_generator_inference_engine_prompt, parse_generator_inference_engine_response
from inference_engine.declarative.inference_engine.linker import (
    get_linker_inference_engine_prompt,
    parse_linker_inference_engine_response,
    get_node_knowledge
)
from inference_engine.declarative.inference_engine.refiner import get_refiner_inference_engine_prompt, parse_refiner_inference_engine_response
import yaml
with open('./config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
USE_COMFYBENCH_WORKFLOW = config['use_comfybench_workflow']
workspace = "workflow_comfybench" if USE_COMFYBENCH_WORKFLOW else "workflow"

class DeclarativePipeline:
    def __init__(
        self,
        save_path: str,
        key_nodes: str,
        num_refs: int = 3,
        num_fixes: int = 3,
        use_claude = False
    ):
        self.save_path = save_path
        self.num_refs = num_refs
        self.num_fixes = num_fixes
        self.key_nodes = parse_code_to_wfcode(key_nodes)
        
        self.invoke_completion = invoke_completion_claude if use_claude else invoke_completion

        logger_name = hashlib.md5(save_path.encode()).hexdigest()
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='[{asctime}] {levelname}: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{'
        )
        file_handler = logging.FileHandler(
            filename=f'{self.save_path}/run.log',
            mode='w'
        )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def __call__(self, query_text: str):
        # Start pipeline
        self.logger.info('Pipeline started')

        # Fetch requirement
        self.logger.info('Fetched requirement:')
        self.logger.info(f'  {query_text}')
        with open(f'{self.save_path}/query.md', 'w') as query_file:
            query_file.write(query_text)

        # Initialize inference_engine
        self.logger.info('Initialize inference_engine')

        analyzer_message = get_analyzer_inference_engine_prompt(
            query=query_text
        )
        self.logger.info(f'Input prompt:{analyzer_message}')
        answer, usage = self.invoke_completion(analyzer_message)
        keyword, analysis = parse_analyzer_inference_engine_response(answer.content)
        self.logger.info(f'Generated answer:{answer.content}')
        self.logger.info(f'Selected keyword:{keyword}')
        self.logger.info(f'Token usage:{usage}')

        references = retrieve_references(
            requirement=analysis,
            count=self.num_refs
        )
        self.logger.info('Retrieved references:')
        for reference in references:
            self.logger.info(f'  {reference.metadata["name"]}: {reference.page_content}')

        # Retrieve key node knowledge
        key_node_knowledge = get_node_knowledge(self.key_nodes)
        self.logger.info(f'Retrieved key node knowledge:{key_node_knowledge}')

        with open(f'./dataset/{workspace}/meta.json') as meta_file:
            metadata = json.load(meta_file)
        
        # select ref by LLM
        if keyword in metadata.keys():
            path = metadata[keyword]
            with open(path['code'], 'r') as code_file:
                code = code_file.read()

            with open(path['description'], 'r') as desc_file:
                description = desc_file.read()
        else:
            code, description = "", ""
        
        code = parse_code_to_wfcode(code)
        
        # Generate workflow nodes
        self.logger.info('Generate workflow nodes')
        
        # Load the reference
        reference = ""
        for ref in references:
            if ref is None:
                raise RuntimeError('Invalid reference')
            
            # if ref.metadata["name"] == keyword: continue
            
            reference += f'- Example: {ref.metadata["name"]}\n\n'
            
            with open(ref.metadata['code'], 'r') as code_file:
                reference_code = code_file.read()
            reference_code = parse_code_to_wfcode(reference_code)
            reference += f'<code>\n{reference_code}\n</code>\n\n'
            
            with open(ref.metadata['description'], 'r') as desc_file:
                reference_description = desc_file.read()
            
            reference += f'<description>\n{reference_description}\n</description>\n\n'
            
        self.logger.info(f'Reference: {reference}')
        
        generator_message = get_generator_inference_engine_prompt(
            code=code,
            query=query_text,
            analysis=analysis,
            keyword=keyword,
            reference=reference,
            key_nodes=self.key_nodes,
            key_node_knowledge=key_node_knowledge
        )
        self.logger.info(f'Input prompt:{generator_message}')
        answer, usage = self.invoke_completion(generator_message)
        combiner_response = answer.content
        self.logger.info(f'Generated answer:{combiner_response}')
        self.logger.info(f'Token usage:{usage}')
        if "python" not in combiner_response:
            code, description = parse_generator_inference_engine_response(combiner_response)
        else:
            _ , description = parse_generator_inference_engine_response(combiner_response)
            code = combiner_response.split("```")[1].replace("python", "").strip()
        self.logger.info(f'Parsed code:{code}')
        self.logger.info(f'Parsed description:{description}')
        
        # Step 2: Link nodes
        self.logger.info('Link nodes to generate complete workflow')
        linker_message = get_linker_inference_engine_prompt(query=query_text, node_code=code)
        self.logger.info(f'Input prompt for linker:{linker_message}')
        answer, usage = self.invoke_completion(linker_message)
        linker_response = answer.content
        self.logger.info(f'Generated linker answer:{linker_response}')
        self.logger.info(f'Token usage:{usage}')
        
        try:
            if "python" not in linker_response:
                linker_code, description = parse_linker_inference_engine_response(linker_response)
            else:
                _ , description = parse_linker_inference_engine_response(linker_response)
                linker_code = linker_response.split("```")[1].replace("python", "").strip()
            if "add node" in linker_code.lower():
                code = linker_code # check nodes' existences
            else:
                code =  code + "\n" + linker_code
                
        except Exception as e:
            self.logger.error(f"Error parsing linker response: {str(e)}")
            self.logger.info('Entering refinement phase due to linker error')
            self._run_refiner(query_text, code, description, str(e))
            return None
        
        self.logger.info(f'Parsed code:\n  {code}')
        self.logger.info(f'Parsed description:\n  {description}')

        # Save workflow
        self.logger.info('Saving workflow')
        
        try:
            # code = parse_wfcode_to_code(code)
            workflow = parse_wfcode_to_workflow(code)
        except Exception as error:
            self.logger.error(f'Error parsing code to workflow: {str(error)}')
            self.logger.info('Entering refinement phase due to save error')
            code, workflow = self._run_refiner(query_text, code, description, str(error), reference)
        
        with open(f'{self.save_path}/code.py', 'w') as code_file:
            code_file.write(code)
        with open(f'{self.save_path}/workflow.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file, indent=4)
        
        # try run the workflow
        outputs = None
        try:
            self.logger.info('Try to execute the workflow')
            status, outputs = execute_workflow(workflow)
        except Exception as error:
            self.logger.error(f'Error executing workflow: {str(error)}')
            self.logger.info('Entering refinement phase due to save error')
            code, workflow = self._run_refiner(query_text, code, description, str(error), reference)
            try: 
                status, outputs = execute_workflow(workflow)
            except Exception as error:
                self.logger.error(f'Error executing workflow: {str(error)}')
                self.logger.info('Failed to refine workflow')
        
        

        self.logger.info(f'Parsed workflow:\n  {workflow}')
        with open(f'{self.save_path}/code.py', 'w') as code_file:
            code_file.write(code)
        with open(f'{self.save_path}/workflow.json', 'w') as workflow_file:
            json.dump(workflow, workflow_file, indent=4)
        
        if outputs != None and status['status_str'] == 'success':
            output_path = os.path.join(self.save_path, 'output')
            os.makedirs(output_path, exist_ok=True)
            for file_name, output in outputs.items():
                file_path = os.path.join(output_path, file_name)
                with open(file_path, 'wb') as file:
                    file.write(output)
        
        
        # Finish pipeline
        self.logger.info('Pipeline finished')
        return workflow

    def _run_refiner(self, query_text: str, code: str, description:str, error_message: str, reference: str):
        for fix_epoch in range(self.num_fixes + 1):
            # Generate refiner prompt
            self.logger.info(f'Starting refinement attempt {fix_epoch + 1}/{self.num_fixes}')
            # code = parse_code_to_wfcode(code)
            refiner_message = get_refiner_inference_engine_prompt(
                query=query_text,
                linked_code=code,
                descript=description,
                error_message=error_message,
                reference=reference
            )
            self.logger.info(f'Input prompt for refiner:\n  {refiner_message}')
            answer, usage = self.invoke_completion(refiner_message)
            refiner_response = answer.content
            self.logger.info(f'Generated refiner answer:\n  {refiner_response}')
            self.logger.info(f'Token usage: {usage}')

            try:
                explanation, refined_code, refined_description = parse_refiner_inference_engine_response(refiner_response)
                self.logger.info(f'Parsed explanation:\n  {explanation}')
                self.logger.info(f'Parsed code:\n  {refined_code}')
                self.logger.info(f'Parsed description:\n  {refined_description}')
                code = refined_code
                # Try saving the workflow again after refinement
                workflow = parse_wfcode_to_workflow(code)
                if workflow is not None:
                    self.logger.info('Refinement successful')
                    return code, workflow
            except Exception as parse_error:
                self.logger.error(f"Error during refinement: {str(parse_error)}")
                if fix_epoch == self.num_fixes:
                    self.logger.info('Failed to refine workflow after maximum attempts')
                    return None
            
