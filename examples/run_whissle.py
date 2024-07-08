import argparse
import ast
import csv
from pathlib import Path
import json
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, T5Tokenizer
import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.builder import get_engine_version
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
}

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM': ":{input_text}<eoh>\n:",
    'QWenForCausalLM': "system\nYou are a helpful assistant.\nuser\n{input_text}\nassistant\n",
}

def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    return model_arch, model_version

def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out

def load_tokenizer(tokenizer_dir: Optional[str] = None,
                   vocab_file: Optional[str] = None,
                   model_name: str = 'GPTForCausalLM',
                   model_version: Optional[str] = None,
                   tokenizer_type: Optional[str] = None):
    if vocab_file is None:
        use_fast = True
        if tokenizer_type is not None and tokenizer_type == "llama":
            use_fast = False
        # Should set both padding_side and truncation_side to be 'left'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  legacy=False,
                                                  padding_side='left',
                                                  truncation_side='left',
                                                  trust_remote_code=True,
                                                  tokenizer_type=tokenizer_type,
                                                  use_fast=use_fast)
    elif model_name == 'GemmaForCausalLM':
        from transformers import GemmaTokenizer

        # Initialize tokenizer from vocab file.
        tokenizer = GemmaTokenizer(vocab_file=vocab_file,
                                   padding_side='left',
                                   truncation_side='left',
                                   legacy=False)
    else:
        # For gpt-next, directly load from tokenizer.model
        tokenizer = T5Tokenizer(vocab_file=vocab_file,
                                padding_side='left',
                                truncation_side='left',
                                legacy=False)

    if model_name == 'QWenForCausalLM':
        with open(Path(tokenizer_dir) / "generation_config.json") as f:
            gen_config = json.load(f)
        chat_format = gen_config['chat_format']
        if chat_format == 'raw' or chat_format == 'chatml':
            pad_id = gen_config['pad_token_id']
            end_id = gen_config['eos_token_id']
        else:
            raise Exception(f"unknown chat format: {chat_format}")
    elif model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eop_token_id
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id

class ModelWrapper:
    def __init__(self, engine_dir, tokenizer_dir, max_output_len, use_py_session=True):
        self.engine_dir = engine_dir
        self.max_output_len = max_output_len
        self.use_py_session = use_py_session
        
        self.model_name, self.model_version = read_model_name(self.engine_dir)
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            vocab_file=None,
            model_name=self.model_name,
            model_version=self.model_version,
            tokenizer_type=None,
        )

        self.prompt_template = None
        if self.model_name in DEFAULT_PROMPT_TEMPLATES:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATES[self.model_name]

        if not PYTHON_BINDINGS and not self.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            self.use_py_session = True
        if self.use_py_session:
            runner_cls = ModelRunner
        else:
            runner_cls = ModelRunnerCpp

        self.runner = runner_cls.from_dir(
            engine_dir=self.engine_dir,
            lora_dir=None,
            rank=tensorrt_llm.mpi_rank(),
            debug_mode=False,
            lora_ckpt_source="hf"
        )

    def generate_response(self, input_text):
        batch_input_ids = parse_input(
            tokenizer=self.tokenizer,
            input_text=input_text,
            prompt_template=self.prompt_template,
            input_file=None,
            add_special_tokens=True,
            max_input_length=923,
            pad_id=self.pad_id,
            num_prepend_vtokens=[],
            model_name=self.model_name,
            model_version=self.model_version
        )

        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids,
                max_new_tokens=self.max_output_len,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_beams=1,
                length_penalty=1.0,
                early_stopping=1,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                stop_words_list=None,
                bad_words_list=None,
                output_cum_log_probs=False,
                output_log_probs=False,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=False,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=None
            )
            torch.cuda.synchronize()

        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        
        responses = []
        for batch_idx in range(len(batch_input_ids)):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][0]
            outputs = output_ids[batch_idx][0][output_begin:output_end].tolist()
            output_text = self.tokenizer.decode(outputs)
            responses.append(output_text)
        
        return responses

def parse_input(tokenizer, input_text=None, prompt_template=None, input_file=None, add_special_tokens=True, max_input_length=923, pad_id=None, num_prepend_vtokens=[], model_name=None, model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8', errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(input_text, add_special_tokens=add_special_tokens, truncation=True, max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(range(base_vocab_size, base_vocab_size + length)) + batch_input_ids[i]

    if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
    return batch_input_ids

# Example usage
if __name__ == '__main__':
    engine_dir = "/external2/models/engine/Llama-2-7b-chat-engine1_wq"
    tokenizer_dir = "/external2/models/hf/Llama-2-7b-chat-hf/"
    max_output_len = 100
    
    model = ModelWrapper(engine_dir, tokenizer_dir, max_output_len)
    
    inputs = ["How do I count to nine in French?", "What is the capital of France?"]
    for input_text in inputs:
        responses = model.generate_response([input_text])
        print(f"Input: {input_text}\nResponse: {responses[0]}\n")
