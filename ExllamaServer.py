from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.server import (
    ExLlamaV2WebSocketServer
)

config = ExLlamaV2Config()

model_directory = "turboderp_Mixtral-8x7B-instruct-exl2_4.0bpw/"

max_seq = 32768

config.model_dir = model_directory
config.prepare()

config.max_seq_len = max_seq

model = ExLlamaV2(config)

split = [0,17,0,24]

model.load(split)

cache = ExLlamaV2Cache(model)

tokenizer = ExLlamaV2Tokenizer(config)

ip = "0.0.0.0"

port = 7862

server = ExLlamaV2WebSocketServer(ip, port, model, tokenizer, cache)

server.serve()
