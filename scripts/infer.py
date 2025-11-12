import click
from loguru import logger
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from med_swift.med_infer_multigpu import MedSwiftInferenceMultiGPU



@click.command()
@click.option("--output_dir", help="output_dir")
@click.option("--ckpt_use", default='best', help="use checkpoint")
@click.option("--path_jsonl", default=None, help="path_jsonl")
@click.option("--dir_save", default=None, help="dir save")
def medsft(
    output_dir: str, 
    path_jsonl: str,
    ckpt_use: str='best',
    dir_save: str=None,
):
    logger.info(f"inference with config: {output_dir}")
    infer = MedSwiftInferenceMultiGPU.from_train_output(
        output_dir, 
        ckpt_use=ckpt_use,
    )
    infer.run(
        path_jsonl=path_jsonl,
        dir_save=dir_save,
    )


if __name__ == "__main__":
    medsft()