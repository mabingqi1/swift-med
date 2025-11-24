"""
在dataset上进行推理，多卡
"""
import copy
import json
import os
from typing import List

from loguru import logger

from swift.llm import PtEngine, InferRequest, RequestConfig
from . import InferArguments
from med_swift.med_template import MedIntern3_5VLTemplate


class MedSwiftInference:
    def __init__(
        self,
        infer_args: InferArguments = InferArguments(),
        device: str="cuda:0", 
        verbose: bool = True,
    ):
        self.infer_args = infer_args
        self.device = device
        self.verbose = verbose
        self.engine = PtEngine(
            **infer_args,
            device_map=device,
            # model_type="internvl3_5" # for internvl3_5
        )

    @staticmethod
    def genearte_messages(system: str, prompt: str):
        messages = [
            {'role': 'system', 'content': system},
            {"role": "user", "content" : prompt}
        ]
        return messages


    def infer_one_batch(self, batch: List):
        infer_requests = []
        for data in batch:
            videos = []
            images = []
            if self.infer_args.video_name is not None:
                videos.append(data[self.infer_args.video_name])
            if self.infer_args.image_name is not None:
                images.append(data[self.infer_args.image_name])
            infer_request = InferRequest(
                messages=self.genearte_messages(self.infer_args.system, self.infer_args.prompt), 
                videos=videos,
                images=images
            )
            infer_requests.append(infer_request)

        # infer
        request_config = RequestConfig(
            temperature=self.infer_args.temperature,
            max_tokens=self.infer_args.max_tokens
        )
        resp_list = self.engine.infer(infer_requests, request_config, use_tqdm=False)
        results = []
        for data, resp in zip(batch, resp_list):
            data = copy.deepcopy(data)
            pred = resp.choices[0].message.content
            if self.verbose:
                print("-" * 100)
                print(pred)
            data[self.infer_args.pred_name] = pred
            results.append(data)
        return results

    
    @classmethod
    def from_train_output(
        clss, 
        output_dir: str, 
        infer_args: InferArguments=InferArguments(),
        ckpt_use: str='best',
        device: str="cuda:0",
        verbose: bool=True,
    ):
        if ckpt_use == 'best':
            dir_checkpoint = get_best_checkpoint_dir(output_dir)
        elif ckpt_use == 'last':
            dir_checkpoint = get_last_checkpoint_dir(output_dir)
        else:
            dir_checkpoint = ckpt_use
            
        path_args = os.path.join(dir_checkpoint, "args.json")
        with open(path_args) as f:
            train_args = json.load(f)

        logger.info(f"load from {dir_checkpoint}")
        
        model_path = None
        adapter_path = None
        if train_args["train_type"] == 'full':
            model_path = dir_checkpoint
        else:
            model_path = train_args["model"]
            adapter_path = dir_checkpoint
        infer_args.model_id_or_path = model_path
        infer_args.adapters = adapter_path
        
        return clss(
            infer_args=infer_args,
            device=device,
            verbose=verbose,
        ) if train_args["train_type"] != 'full' else clss(
            infer_args=infer_args,
            device=device,
            verbose=verbose,
        )


def get_last_checkpoint_dir(dir_train: str):
    last_iter = max([
        int(x.replace("checkpoint-", "")) for x in os.listdir(dir_train) \
            if x.startswith("checkpoint-")
    ])
    dir_last = os.path.join(dir_train, f"checkpoint-{last_iter}")
    return dir_last


def get_best_checkpoint_dir(dir_train: str):
    dir_last = get_last_checkpoint_dir(dir_train)
    path_state_last = os.path.join(dir_last, "trainer_state.json")
    with open(path_state_last) as f:
        state_last = json.load(f)
    dir_best = state_last["best_model_checkpoint"]
    return dir_best
