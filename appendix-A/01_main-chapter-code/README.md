# Appendix A: Introduction to PyTorch

### Main Chapter Code

- [code-part1.ipynb](code-part1.ipynb) contains all the section A.1 to A.8 code as it appears in the chapter
- [code-part2.ipynb](code-part2.ipynb) contains all the section A.9 GPU code as it appears in the chapter 
- [DDP-script.py](DDP-script.py) contains the script to demonstrate multi-GPU usage (note that Jupyter Notebooks only support single GPUs, so this is a script, not a notebook). You can run it as `python DDP-script.py`. If your machine has more than 2 GPUs, run it as `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py`.
- [exercise-solutions.ipynb](exercise-solutions.ipynb) contains the exercise solutions for this chapter

### Optional Code

- [DDP-script-torchrun.py](DDP-script-torchrun.py) is an optional version of the `DDP-script.py` script that runs via the PyTorch `torchrun` command instead of spawning and managing multiple processes ourselves via `multiprocessing.spawn`. The `torchrun` command has the advantage of automatically handling distributed initialization, including multi-node coordination, which slightly simplifies the setup process. You can use this script via `torchrun --nproc_per_node=2 DDP-script-torchrun.py`
