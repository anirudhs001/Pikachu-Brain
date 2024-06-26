import socket
from ferret import load_pretrained_model, tokenizer_image_token
from ferret import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
import torch
from typing import List, Tuple
import numpy as np
from PIL import Image
import io
from glob import glob
from tqdm import tqdm
import traceback
import math

DEFAULT_IM_W = 336
DEFAULT_IM_H = 336
DEVICE = "mps"

PIKACHU_PROMPT = f"A chat between a human and his pet PIKACHU that understands visuals.\
Act as if you are the pokemon. Very briefly respond to the situations as if pikachu would.\
After every sentence, add a small sentence or word describing pikachu's facial expression in curly braces '{' '}'.\
JUST RESPOND TO THE USER'S PROMPT, DO NOT ADD THE USER'S SIDE OF THE CONVERSATION.\
DO NOT HALLUCINATE WHAT THE USER DOES. ONLY GENERATE PIKACHU'S RESPONSES\
DESCRIBE WHAT ACTIONS THE POKEMON WOULD TAKE. Use English.\
ALWAYS END PIKACHU'S RESPONSES WITH <s>."


def insert_faces(text, images, processor, model, match_window=10):
    words = text.split(" ")
    # text_segments = []
    # for i in range(math.ceil(len(words) // match_window) + 1):
    #     text_segment = words[i * match_window : min((i + 1) * match_window, len(words))]
    #     text_segment = " ".join(text_segment)
    #     text_segments.append(text_segment)
    face_descs = []
    for i, word in enumerate(words):
        word = word.strip()
        if len(word) == 0:
            continue
        if word[0] == "{":
            face_desc = []
            for w in words[i:]:
                face_desc.append(w)
                w = w.strip()
                if len(w) == 0:
                    continue
                if w.strip()[-1] == "}":
                    break
            face_desc = " ".join(face_desc)
            face_descs.append(face_desc)

    if len(face_descs) == 0:
        face_descs = ["pikachu smiling"]
    inputs = processor(
        text=face_descs, images=images, return_tensors="pt", padding=True
    )

    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text  # this is the image-text similarity score
    probs = logits_per_text.softmax(dim=1)
    face_idx = []
    out_text = ""
    for row_num in range(probs.shape[0]):
        sample_idx = torch.multinomial(probs[row_num], 1)
        face_idx.append(sample_idx)
        out_text += f"<face:{int(sample_idx)+1}> " + face_descs[row_num]
    return out_text


@torch.inference_mode()
def infer_ferret(
    tokenizer,
    model,
    image_processor,
    context_len,
    messages: List[List[str]],
    image,
    image_h=DEFAULT_IM_H,
    image_w=DEFAULT_IM_W,
):
    if image:
        image = image_processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_center_crop=False,
            size=[image_h, image_w],
        )["pixel_values"]
        image = image.to(model.device, dtype=torch.float16)
    image_args = {"images": image}

    # construct prompt
    # messages are in the format [['USER', 'some text'], ['ASSITANT', 'some text'], ...]
    seps = [" ", "</s>"]
    system_prompt = f"A chat between a human and an AI that understands visuals. \
in images, [x, y] denotes points: top-left [0, 0], bottom-right [width-1, height-1]. \
increasing x moves right; y moves down. \
bounding box: [x1, y1, x2, y2]. image size: {image_w}x{image_h}. \
follow instructions. Only use English.\n"
    prompt = PIKACHU_PROMPT
    for i, (role, message) in enumerate(messages):
        if i == 0:
            message = f"<image>\n{message}"
            pass
        prompt += f"{role}: {message}{seps[i%2]}"
    prompt += f"ASSISTANT: "

    l_prompt = len(prompt)
    # params are defined in gradio_web_server.py
    temperature = 0.2
    top_p = 0.7
    max_new_tokens = 512
    stop_str = "</s>"
    stop_idx = tokenizer(stop_str).input_ids
    stop_idx = stop_idx[0] if len(stop_idx) == 1 else None

    # tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None
    )
    output_ids = list(input_ids)
    pred_ids = []

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    # run the model
    past_key_values = None
    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids]).to(DEVICE), use_cache=True, **image_args
            )
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=DEVICE
            )
            out = model(
                input_ids=torch.as_tensor([[token]], device=DEVICE),
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                region_masks=None,
            )
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)
        pred_ids.append(token)

        if stop_idx is not None and token == stop_idx:
            stopped = True
        elif token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i == max_new_tokens - 1 or stopped:
            cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
            pos = cur_out.rfind(stop_str)
            if pos != -1:
                cur_out = cur_out[:pos]
                stopped = True
            output = prompt + cur_out

            ret = {
                "text": output,
                "cur_out": cur_out,
                "error_code": 0,
            }
            return ret

        if stopped:
            break


def infer_flan_t5():
    pass


def infer_lmsys_fastchat_3b(model, tokenizer, message):
    pass


def infer_mistral_8x7b():
    pass


class Mistral:
    def __init__(self, model_path="models/mistral-7b-instruct-v0.2.Q5_K_M.gguf") -> None:
        self.model = Llama(
            model_path=model_path,
            n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=35,
        )
        self.messages = messages = [
            {"role": "user", "content": f"{PIKACHU_PROMPT}"},
            {"role": "system", "content": "Understood <s>"},
        ]

    def make_prompt(self):
        prompt = ""
        for i, message in enumerate(self.messages):
            role, content = message["role"], message["content"]
            if i == 0:
                prompt += "<s>"
            if role == "user":
                prompt += f"[INST] {content} [INST]\n"
            else:
                prompt += f"{content}\n"
        return prompt

    @torch.inference_mode()
    def infer(self, text_message: str):
        print("inside infer()")
        self.messages.append({"role": "user", "content": text_message})
        prompt = self.make_prompt()
        out = self.model(
            prompt,
            max_tokens=50,  # Generate up to 512 tokens
            stop=["<s>", "\n\n", "["],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=False,        # Whether to echo the prompt
        )
        cur_out = out["choices"][0]["text"]
        self.messages.append({"role": "system", "content": cur_out})
        return cur_out


def serve(server_port=5555, save_images=False):
    pbar = tqdm(range(3))
    # logging stuff

    # socket stuff
    pbar.set_description("Setting up TCP server")
    pbar.refresh()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", server_port))
    server_socket.listen(1)
    pbar.update(1)
    pbar.refresh()

    # load model
    pbar.set_description("loading model and tokenizer")
    pbar.refresh()
    ## ferret
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path="/Users/anirudhsingh/MISC/playground/ml-ferret/ferret/model/ferret-7b-v1-3",
    #     model_name="ferret-7b-v1-3",
    #     device_map=DEVICE,  # use m1 gpu for inference
    # )

    ## fastchat t5 3b
    # model_id = "lmsys/fastchat-t5-3b-v1.0"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

    ## mistral
    mistral = Mistral()

    pbar.update(1)
    pbar.refresh()
    # print(tokenizer)
    # print(model)
    # print(image_processor)
    # print(context_len)

    # clip for matching faces
    pbar.set_description("loading CLIP")
    pbar.refresh()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    num_images = len(glob("faces/*.png"))
    images = [Image.open(f"faces/Picture {i}.png") for i in range(1, num_images)]
    pbar.update(1)
    pbar.set_description("done")
    pbar.refresh()
    pbar.close()

    print(f"Server listening on port {server_port}")

    while True:
        print("Waiting for connection...")
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")
        try:
            while True:
                # Receive text message
                text_message = client_socket.recv(1024).decode("utf-8")
                print(f"Received text message: {text_message}")

                # Receive image
                image_size = int(client_socket.recv(20).decode("utf-8"))
                print("image size", image_size)
                image_data = b""
                remaining_size = image_size
                image = None
                while remaining_size > 0:
                    packet = client_socket.recv(min(1024, remaining_size))
                    if not packet:
                        break
                    image_data += packet
                    remaining_size -= len(packet)
                    image: Image = Image.open(io.BytesIO(image_data))

                if save_images:
                    image_filename = f"received_image_{client_address[1]}.png"
                    with open(image_filename, "wb") as image_file:
                        image_file.write(image_data)
                    print(f"Received image and saved as {image_filename}")

                # inference: pass image and text to model
                # for ferret: messages.append(["USER", text_message])
                # assist_out = infer_ferret(
                #     tokenizer, model, image_processor, context_len, messages, image
                # )
                assist_out = mistral.infer(text_message)
                print(assist_out)
                # insert face idxs
                assist_out = insert_faces(assist_out, images, processor, clip_model)
                client_socket.sendall(assist_out.encode("utf-8"))
        except Exception as e:
            print(e)
            traceback.print_exc()
            client_socket.close()


if __name__ == "__main__":
    serve()
