from celery import Celery

import os, glob
from pathlib import Path
import time
import ast
import re
import gc
import torch
import json
import time
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
import GPUtil
import random
import os
import subprocess
import argparse

def extract_used_contexts(full_text):
    try:
        # Find the start of the context section
        start_idx = full_text.find("Context used in answer:")
        if start_idx == -1:
            return []

        # Extract the substring from the start index to the end of the text
        context_section = full_text[start_idx:]

        # Find the start and end of the JSON-like structure
        start_json = context_section.find("[")
        end_json = context_section.find("]", start_json)
        if start_json == -1 or end_json == -1:
            return []

        # Extract and parse the JSON-like string
        json_string = context_section[start_json:end_json + 1]
        used_contexts = json.loads(json_string.replace("'", '"'))
        return used_contexts
    except json.JSONDecodeError:
        # Handle cases where JSON decoding fails
        return []


def inference(prompt, model, config, max_seq):
    cache = ExLlamaV2Cache(model)

    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temparature = 1.3
    settings.min_p = 0.02
    settings.token_repetition_penalty = 1.25

    ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    max_prompt_length = max_seq - 2048
    ids = ids[:, -max_prompt_length:]
    initial_len = ids.shape[-1]

    cache.current_seq_len = 0
    model.forward(ids[:, :-1], cache, input_mask=None, preprocess_only=True)
    prev_decoded_text = ""  # Variable to store the previously decoded text

    has_leading_space = False
    full_text = ""
    answer_started = False
    Answer_end_marker1 = "[EoA]"
    Answer_end_marker2 = "EoA"

    try:
        for i in range(2048):
            logits = model.forward(ids[:, -1:], cache, input_mask=None).float().cpu()
            token, _, _ = ExLlamaV2Sampler.sample(logits, settings, ids, random.random(), tokenizer)
            ids = torch.cat([ids, token], dim=1)

            if i == 0 and tokenizer.tokenizer.id_to_piece(int(token)).startswith('▁'):
                has_leading_space = True

            decoded_text = tokenizer.decode(ids[:, initial_len:])[0]

            if has_leading_space:
                decoded_text = ' ' + decoded_text

            new_text = decoded_text.replace(prev_decoded_text, '')

            prev_decoded_text = decoded_text

            full_text += new_text

            if "Answer:" in prev_decoded_text:
                answer_started = True

            if new_text.endswith(Answer_end_marker1) or new_text.endswith(Answer_end_marker2):
                answer_started = False

            if answer_started:
                yield new_text

            if token.item() == tokenizer.eos_token_id:
                break



            #print(new_text, end='', flush=True)

    finally:
        # Code here will execute after the loop ends, regardless of how it ends.
        full_text = full_text.replace('</s>', '')

        used_contexts = extract_used_contexts(full_text)

        with open("contexts.json", "w") as f:
            json.dump(used_contexts, f)
        
        #print("Full Text:", full_text)

        print("Used Contexts:", used_contexts)

        print("Done")



celery_app = Celery(
    "worker",
    broker="amqp://localhost",  # RabbitMQ broker
    backend="rpc://",  # RPC backend
)

@celery_app.task
def inference_task(prompt, max_seq):
    config = ExLlamaV2Config()

    model_directory = "turboderp_Mixtral-8x7B-instruct-exl2_4.0bpw/"

    max_seq = 32768

    config.model_dir = model_directory
    config.prepare()

    config.max_seq_len = max_seq

    model = ExLlamaV2(config)
    print("Cache Map:", model.cache_map)

    split = [17,24,0,0]

    model.load(split)

    return inference(prompt, model, config, max_seq)
