python main.py trainer=nafmamba optim=nafmamba model=nafmamba gpu_ids=\'2\' data=icvl  data.bs=32 noise.params.sigma_max=15 trainer.params.num_sanity_val_steps=0 
test=icvl15