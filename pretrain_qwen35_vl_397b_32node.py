import os

from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step
from megatron.bridge.recipes.qwen_vl import qwen35_vl_397b_a17b_sft_config
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.common_utils import get_rank_safe


if __name__ == "__main__":
    # If you have a local checkpoint, replace with local path.
    hf_path = "./Qwen/Qwen3.5-397B-A17B"
    cfg = qwen35_vl_397b_a17b_sft_config(hf_path)

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096


    # cfg.train.micro_batch_size = 1

    
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True


    # cfg.dataset.maker_name = "make_image_caption_pretrain_dataset"

    # cfg.model.context_model_parallel_size = 1 
    # cfg.model.use_cpu_initialization = True  # Required to avoid OOM during large model loading
    # cfg.model.context_model_parallel_size = 1  # Context model parallelism not needed at 16 nodes
    cfg.train.train_iters = 3000
    cfg.train.global_batch_size = 2048
    cfg.validation.eval_interval = 10000
    cfg.validation.eval_iters = 10
    cfg.dataset.pack_sequences_in_batch = False

    run_output_dir = os.path.join(os.getcwd(), "nemo_experiments", "qwen35_vl_397b_32node")
    cfg.checkpoint.save = os.path.join(run_output_dir, "checkpoints")
    cfg.checkpoint.load = cfg.checkpoint.save
    cfg.checkpoint.save_interval = 500
    cfg.checkpoint.async_save = True
    cfg.checkpoint.async_strategy = "mcore"
    cfg.logger.tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
    cfg.logger.log_interval = 1
    cfg.train.timing_log_level = 2


    ###### CONFIG TO TRY ######
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 16
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_method = None
    cfg.model.recompute_modules = ["moe_act", "layernorm"]
    cfg.model.recompute_num_layers = None
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.moe_shared_expert_overlap = True
    cfg.model.moe_router_fusion = True
    cfg.model.tp_comm_overlap = True
    cfg.train.manual_gc_interval = 10
    cfg.train.micro_batch_size = 1
    cfg.model.fp8 = "hybrid"
    # cfg.model.virtual_pipeline_model_parallel_size = 3  # OOM with EP=16
    ###### END OF CONFIG TO TRY ######



    rank = get_rank_safe()
    if rank == 0:
        cfg.to_yaml("qwen35_vl_397b_a17b_32node_config.yaml")

    pretrain(cfg, forward_step)
