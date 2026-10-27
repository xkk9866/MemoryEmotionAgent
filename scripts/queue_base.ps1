$env:HF_HOME = "D:\hf_cache"
$env:CUDA_VISIBLE_DEVICES = $args[0]
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force D:\cate_data\logs | Out-Null
$log = "D:\cate_data\logs\queue_gpu$($args[0]).log"
python -m cate.train_eam --variants base eam --seeds 42 43 44 --epochs 4 *>> $log
python -m cate.train_eam --variants eam_net eam_no_decay eam_no_fbsup eam_usage_only eam_no_prior --seeds 42 43 44 --epochs 4 *>> $log
python -m cate.train_eam --variants base --seeds 42 43 44 --epochs 4 --encoder bert-base-uncased --tag bert-base *>> $log
python -m cate.train_eam --variants base eam --seeds 42 43 44 --epochs 4 --encoder microsoft/deberta-v3-base --tag deberta-v3-base *>> $log
python -m cate.train_eam --variants base --seeds 42 43 44 --epochs 4 --encoder facebook/blenderbot_small-90M --tag blenderbot-small *>> $log
