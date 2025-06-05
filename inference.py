import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import torch

from accelerate import Accelerator, PartialState
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, GenerationConfig, TextStreamer

##########################################################

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py
# CUDA_VISIBLE_DEVICES=1 accelerate launch inference.py

##########################################################

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print('====================================')
print('Count of using GPUs:', torch.cuda.device_count())
print('====================================')

@dataclass
class ScriptArguments:
    model_type: Optional[str] = field(default="openchat", metadata={"help": "the model type"})
    base_model_name: Optional[str] = field(default="../model/openchat-3.5-0106-private", metadata={"help": "the base model name"})
    peft_model_name: Optional[str] = field(default="", metadata={"help": "the peft model name"})

    # model_type: Optional[str] = field(default="llama3", metadata={"help": "the model type"})
    # base_model_name: Optional[str] = field(default="../model/Meta-Llama-3-8B-Instruct-private", metadata={"help": "the base model name"})
    # peft_model_name: Optional[str] = field(default="", metadata={"help": "the peft model name"})

##########################################################

def load_base_model(model_type, model_name, quantization_config):
    device_string = PartialState().process_index
    if model_type == "openchat":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            device_map={'': device_string},
            attn_implementation="flash_attention_2"
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
    elif model_type == "gemma" or model_type == "gemma2":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            device_map={'': device_string},
            attn_implementation="eager"
        )
    elif model_type == "llama3":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            device_map={'': device_string},
            attn_implementation="flash_attention_2"
        )
    return model

def load_peft_model(model_type, model_name):
    device_string = PartialState().process_index
    if model_type == "openchat":
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            is_trainable=True,
            device_map={'': device_string},
            attn_implementation="flash_attention_2"
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
    elif model_type == "gemma" or model_type == "gemma2":
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            is_trainable=True,
            device_map={'': device_string},
            attn_implementation="eager"
        )
    elif model_type == "llama3":
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            return_dict=True,
            is_trainable=True,
            device_map={'': device_string},
            attn_implementation="flash_attention_2"
        )
    return model

def load_model(args, quantization_config):
    if args.peft_model_name is None or args.peft_model_name == '' :
        model = load_base_model(args.model_type, args.base_model_name, quantization_config)
    else:
        model = load_peft_model(args.model_type, args.peft_model_name)
    model.config.use_cache = False
    model.config.use_xformers = True
    model.eval()
    return model

def load_tokenizer(args, model):
    if args.model_type == "openchat":
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<unk>'})
            model.resize_token_embeddings(len(tokenizer))
    elif args.model_type == "gemma" or args.model_type == "gemma2":
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        tokenizer.padding_side = "right"
    elif args.model_type == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    return tokenizer

##########################################################

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

model = load_model(args, None)
tokenizer = load_tokenizer(args, model)
streamer = TextStreamer(tokenizer, timeout=10, skip_prompt=True, skip_special_tokens=True)

##########################################################

def create_prompt(question):
    system_msg = "다음의 민원 상담에 대한 context를 기반으로 질문에 대한 적절한 답변을 한국어로 작성하세요. "
    if args.model_type == "openchat":
        chat = []
        chat.append({"role": "user", "content": f"{system_msg + question}"})
    elif args.model_type == "gemma" or args.model_type == "gemma2":
        chat = []
        chat.append({"role": "user", "content": f"{system_msg + question}"})
    elif args.model_type == "llama3":
        chat = []
        chat.append({"role": "system", "content": f"{system_msg}"})
        chat.append({"role": "user", "content": f"{question}"})

    # tokenizer.use_default_system_prompt = False
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # print('----------------')
    # print(prompt)
    # print('----------------')
    return prompt

def generation_config():
    generation_config = model.generation_config
    generation_config.temperature=0.1
    generation_config.max_new_tokens=1024
    generation_config.pad_token_id=tokenizer.eos_token_id

def create_result(response):
    if args.model_type == "openchat":
        find_start = "GPT4 Correct Assistant:"
        find_end = "<|end_of_turn|>"
    elif args.model_type == "gemma" or args.model_type == "gemma2":
        find_start = "<start_of_turn>model"
        find_end = "<end_of_turn>"
    elif args.model_type == "llama3":
        find_start = "<|start_header_id|>assistant<|end_header_id|>"
        find_end = "<|eot_id|>"
        
    decode = tokenizer.batch_decode(response)[0]
    start_index = decode.rfind(find_start)
    end_index = decode.rfind(find_end)
    if start_index != -1:
        if end_index != -1 and end_index > start_index:
            return decode[start_index + len(find_start) + 1 : end_index].strip()
        else:
            return decode[start_index + len(find_start) + 1 :].strip()
    else:
        return decode
    
def inference(question):
    try:
        torch.cuda.empty_cache()

        pre = "질문: "
        prompt = create_prompt(pre + question)
        inputs = tokenizer(prompt, return_tensors="pt")
        device_string = PartialState().process_index
        inputs = {k: v.to(device_string) for k, v in inputs.items()}
        response = model.generate(
            **inputs,
            generation_config=generation_config(),
            streamer=streamer,
        )
        answer = create_result(response)
        return answer
    except Exception as e:
        print(f"에러 발생: {e}")
        return "에러가 발생했습니다."

##########################################################

if __name__ == "__main__":
    print("#############################")
    print("추론 테스트 코드입니다. 종료 명령어는 exit 입니다.")
    print("질문 예시는 아래와 같습니다. 주제분류, 요약, 질의응답에 해당하는 질문과 상담 context 를 아래와 같이 입력해 주어야 합니다.")
    print("=============================")
    print(">>> 질문 : 민원인의 질의 내용을 100자 내외로 적어 줘. \ncontext: 출퇴근 맞춤 버스 도입요청\n\nQ : 000번 노선 이용자입니다\n고척 근린 버스정류장부터 해서 그 이후 정류장까지 신도림, 여의도 가는 사람들이 이 버스만 타나 봅니다.\n서로 타야 한다며 밀고 그러다 싸우고, 오늘도 승객들끼리 버스에서 밀지 말라고, 뼈 부러지겠다고 싸웠습니다. 종종 싸웁니다.\n사람이 너무 많아 심지어 승객을 태우지도 못하고 지나치는 상황도 발생이 됩니다.\n언제까지 이래야 하나요.\n기사님들 피로도도 장난 아니게 심하실 거예요.\n사이드미러가 안 보이는 상태에서, 심지어 그 상태에서 버스전용차로가 없는 일반 도로에서의 운전은 얼마나 불안하실까요.\n비단 000번 버스노선만의 문제일까요?\n언제 도입될까요? 출퇴근 맞춤 버스?\n게시공고 계획 등 진행 중이거나 계획하려는 내용을 시민들에게 알려주시면 안 될까요?\n비밀일까요? 아니면 이렇게 민원 쓰는 사람에게만 알려주시나요?\n빠른 개선 도입 등 뭐든 할 수 있는 일을 해주시길 부탁드립니다. 수고하세요.\n\nA : 안녕하십니까?\n귀하께서 응답소(시장에게 바란다)를 통해 신청하신 민원에 대한 검토 결과를 다음과 같이 알려드립니다.\n귀하의 민원내용은 \"고척동~여의도, 신도림 연계 출퇴근 맞춤버스 신설 요청\"에 관한 것으로 이해됩니다.\n귀하의 질의사항에 대해 검토한 의견은 다음과 같습니다.\n먼저, 우리 시 시내버스는 준공영제로 운영하고 있어 한정된 차량으로 서울 시내 모든 구간을 효율적으로 연계하기 위해 노력하고 있으며, 한정된 재원(차고지, 차량 등) 하에서 노선 신설을 위해서는 인근 타 노선을 감차하거나, 노선을 단축하여 차량을 확보하는 방안이 있으나 이는 기존 이용 승객들의 또 다른 불편을 유발하므로 신중한 검토가 필요한 점 양해 부탁드립니다.\n향후, 주변 지역의 여유차량이 확보되는 경우 건의 주신 구간의 혼잡도 및 이동수요 등을 종합적으로 고려하여 개선을 검토하겠습니다.\n추가적인 문의사항이 있으실 경우 서울시 도시교통실 교통기획관 버스정책과 ○○○에게 연락주시면 친절하게 설명하여 드리도록 하겠습니다. 감사합니다.")
    print("#############################")
    end = 1
    while end == 1 :
        question = input(">>> 질문 : ")
        if question == "" or question.strip() == "" :
            continue
        elif question == "exit" or question == "EXIT" or question == "quit" or question == "QUIT" :
            print("종료!!")
            break
        response = inference(question)
        print("\n>>> 답변 : " + response + "\n")

##########################################################