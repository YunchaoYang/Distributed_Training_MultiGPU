
|compare |pytorch                                |          deepspeed                |
|--------------|:--------------------------------|:------------------------------------|
|init process  |torch.distributed.init_process_group(...) | deepspeed.init_distributed()|
|wrapper |model | model_engine |
|config |None  | configuration file|
|launcher |torchrun | deepspeed <hostfile or args> client_entry.py <client args> <config.json> |



