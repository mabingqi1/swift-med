"""
在dataset上进行推理，多卡
"""
import copy
import json
import multiprocessing as mp
import os
import torch
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from . import InferArguments




class GPUWorker(mp.Process):
    def __init__(
        self, 
        queue_task: mp.JoinableQueue,
        queue_result: mp.Queue,
        device: str,
        output_dir: str, 
        infer_args: InferArguments=InferArguments(),
        ckpt_use: str = 'best',
        verbose: bool = True,
    ):
        super().__init__()
        self.device = device
        self.queue_task = queue_task
        self.queue_result = queue_result
        self.output_dir = output_dir
        self.infer_args = infer_args
        self.ckpt_use = ckpt_use
        self.verbose = verbose
    
    def run(self):
        try:
            from .med_infer_base import MedSwiftInference
            infer = MedSwiftInference.from_train_output(
                self.output_dir, self.infer_args, ckpt_use=self.ckpt_use, device=self.device, verbose=self.verbose
            )
        except KeyboardInterrupt:
            assert False
        except:
            import traceback
            logger.error(traceback.format_exc())
        while True:
            try:
                index, N, batch, command = self.queue_task.get()
                if command == "done":
                    self.queue_task.task_done()
                    break
                logger.info(f"infer start: {index}/{N} @ {self.device}")
                time_s = time.time()
                results = infer.infer_one_batch(batch)
                logger.info(f"infer end: {index}/{N} @ {self.device}: {round(time.time() - time_s, 3)}")
                self.queue_result.put((index, results))
                self.queue_task.task_done()
            except:
                import traceback
                logger.error(traceback.format_exc())
                self.queue_task.task_done()
        logger.info(f"{self.device} exit!")


class MedSwiftInferenceMultiGPU:
    def __init__(
        self,
        output_dir: str, 
        infer_args: InferArguments=InferArguments(),
        devices: List[str] = ["cuda:0"], 
        ckpt_use: bool = 'best',
        verbose: bool = True,
    ):
        self.output_dir = output_dir
        self.infer_args = infer_args
        self.devices = devices
        self.ckpt_use = ckpt_use
        self.verbose = verbose

        self.queue_task = mp.JoinableQueue(maxsize=10000000)
        self.queue_result = mp.Queue(maxsize=10000000)


    def infer_with_datas(self, datas: List[dict], batch_size: int=16):
        # init worker
        worker_list: List[GPUWorker] = []
        for device in self.devices:
            worker = GPUWorker(
                queue_task=self.queue_task,
                queue_result=self.queue_result,
                device=device,
                output_dir=self.output_dir,
                infer_args=self.infer_args,
                ckpt_use=self.ckpt_use,
                verbose=self.verbose,
            )
            worker.start()
            worker_list.append(worker)

        # put tasks
        batch_data_list = [datas[i: i+batch_size] for i in range(0, len(datas), batch_size)]
        for index, batch in enumerate(batch_data_list):
            self.queue_task.put((index, len(batch_data_list), batch, "task"))
        for _ in worker_list:
            self.queue_task.put((-1, len(batch_data_list), None, "done"))
        
        # join
        self.queue_task.join()
        logger.info("=== all completed ===")
        
        # collect results
        logger.info(f"queue_result: {self.queue_result.qsize()}")
        results = []
        while self.queue_result.qsize() > 0:
            result = self.queue_result.get()
            results.append(result)
            
        logger.info(f"batch_results: {len(results)}")
        results = sorted(results, key=lambda x: x[0])
        preds = []
        for result in results:
            preds.extend(result[1])
        logger.info(f"results: {len(preds)}")
        
        # kill worker
        for worker in worker_list:
            worker.terminate()
        return preds
    
    
    @classmethod
    def from_train_output(
        clss, 
        output_dir: str, 
        infer_args: InferArguments=InferArguments(),
        ckpt_use: str='best',
        verbose: bool=True,
        devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())]
    ):
        return clss(
            output_dir=output_dir,
            infer_args=infer_args,
            devices=devices,
            ckpt_use=ckpt_use,
            verbose=verbose,
        )


    def run(
        self, 
        path_jsonl: List[str]=None, 
        dir_save: str=None,
    ):
        if path_jsonl is None:
            path_args = os.path.join(self.output_dir, "args.json")
            with open(path_args) as f:
                train_args = json.load(f)
            path_jsonl = train_args["val_dataset"]
        
        if isinstance(path_jsonl, list) is not True:
            path_jsonl = [path_jsonl]
        for path in path_jsonl:
            with open(path) as f:
                datas = []
                for line in f:
                    datas.append(json.loads(line))
            results = self.infer_with_datas(datas, batch_size=self.infer_args.max_batch_size)

            if dir_save is None:
                dir_save = os.path.join(self.output_dir, "inference", os.path.basename(path))
            os.makedirs(dir_save, exist_ok=True)
            path_out = os.path.join(dir_save, f"infer_results-{os.path.basename(self.ckpt_use)}.jsonl")
            logger.info(f"save to {path_out}")
            with open(path_out, 'w') as f:
                f.write("\n".join([json.dumps(x, ensure_ascii=False) for x in results]))