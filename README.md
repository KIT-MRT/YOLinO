# YOLinO: Polyline Estimation

The repository contains code to train and evaluate our YOLinO network for polyline estimation from RGB images. The code was developed for a PhD thesis and is targeted towards evaluating rather productive use.

## Open Issues
There are some issues we want to tackle in the future:
- [ ] Cleanup code
- [ ] Extract experiment coding for clean code structure
- [ ] Migrate from Gitlab CI to Github Actions
- [ ] Add proper documentation
- [ ] Provide parametrization instructions
- [ ] Provide params.yaml for Argoverse, Tusimple, CULane, ...

## Citation
When using this code please cite our publications:
```
@inproceedings{meyer2021yolino,
  title={YOLinO: Generic Single Shot Polyline Setection in Real Time},
  author={Meyer, Annika and Skudlik, Philipp and Pauls, Jan-Hendrik and Stiller, Christoph},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={2916--2925},
  year={2021}
}
```


## Installation

### Virtual Environments
It is recommended to use a virtualenv or conda to wrap your python packages from the rest of your system. Especially tensorflow and pytorch might exist in another version on a server.

#### Conda
I recommend to use conda: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda
After installation you should be able to create your conda environment with
```
conda create --name yolino pip python==3.8
```
You might want to have autocomplete with `conda install argcomplete`. Then add `eval "$(register-python-argcomplete conda)"` to your bash file. For zshell use https://github.com/conda-incubator/conda-zsh-completion/blob/master/_conda.

#### Virtualenv
If you want to use virtualenvwrapper: https://virtualenvwrapper.readthedocs.io/en/latest/install.html
After installation you should be able to create your virtualenv with
```
mkvirtualenv yolino --python=/usr/bin/python3
```

Make sure your PYTHONPATH is empty. Maybe add `export PYTHONPATH=` to your virtualenv scripts e.g. `~/.virtualenvs/yolino/bin/postactivate`.


### Clone repo
```bash
mkdir yolino
cd yolino
git clone https://github.com/KIT-MRT/YOLinO.git

conda create --name yolino pip python==3.8
# Alternative: mkvirtualenv yolino --python=/usr/bin/python3

# You should be working in a virtual env now

cd yolino

# If you want a specifiy branch do
git checkout <branch-name>

# Make sure you did not source any mrt or ros stuff; should be empty for most cases!
echo $CMAKE_PREFIX_PATH
echo $PYTHONPATH

# Install requirements
pip install -e .
```

### CUDA

If you want to use a specific cuda version have a look at `https://pytorch.org/get-started/previous-versions/`. For CUDA 11.6 it is recommended to use
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### Get weights

Download the darknet weights at https://pjreddie.com/media/files/yolov3.weights.

### Dataset Paths

The code expects the dataset files to be accessible at the environment variable fitting the dataset e.g. `$DATASET_TUSIMPLE` and `$DATASET_CULANE`, respectively.

### Folder Setup

So far a second folder is necessary next to the acutal yolino package, where we store parametrization, checkpoints etc. We recommend to use separate folders for dealing with different datasets or configurations. For example, we use `tus_po_8p_dn19` as a good start for the tusimple dataset. The name encodes the **tus**imple dataset with **po**ints line representation, **8 p**redictors without any upsampling on **darknet-19**. The scripts expect this folder structure by default. Pass `--root` (path to the yolino folder), `--dvc` (folder containing your output) and `--config` (path to a params.yaml) if the structure is different.

  ```
  ├── tus_md_8p_dn19
  │   ├── params.yaml
  │   ├── default_params.yaml
  │   ├── ...
  ├── tus_po_8p_dn19
  │   ├── params.yaml
  │   ├── default_params.yaml
  │   ├── ...
  ├── yolino
    ├── setup.py
    ├── src
    │   ├── ...
    ├── ...
  ```

### Logging Server

We can log automatically to weights and biases, clearml, command line and logging file. With `--loggers` this can be
specified. It is recommended to use only weights and biases. File logging slows down the process as it logs everything.

- **Weights and Biases**: Setup your account on https://wandb.ai/site and run `wandb login` in your python environment
  on your machine.
- **Clearml**: Setup your account on https://app.clear.ml/ and run `clearml-init` in your python environment on your
  machine.

### Other Requirements

- **Use Python3 only!**
- On Servers: execute `echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc` for non interacting matplotlib
- List of requirements can be found in `setup.cfg` and will be automatically retrieved by `pip install -e yolino` (see
  above)
- If the tests fail, make sure you have git lfs setup and all files in test/test_data are fetched properly

### Code execution examples

- Training in folder e.g. called tus_po_8p_dn19 on e.g. GPU with ID 1 (check `nvidia-smi`)
    ```
    cd tus_po_8p_dn19
    CUDA_VISIBLE_DEVICES="1" python ../yolino/src/yolino/train.py --gpu
    ```
- Evaluation in folder e.g. called tus_po_8p_dn19 (https://gitlab.mrt.uni-karlsruhe.de/meyer/dvc_experiment_mgmt) on
  e.g. CPU (provide GPU IDs if you prefer GPU)
    ```
    cd tus_po_8p_dn19
    CUDA_VISIBLE_DEVICES="" python ../yolino/src/yolino/eval.py
    ```

## Usage

### Training

1. Your configuration is set with the given `params.yaml` from your configuration folder. Use `res/default_params.yaml` as inspiration. 
4. Use proper virtual environment with `workon <virtualenv-name>` or conda with `conda activate <conda-name>`.
2. Set your dataset paths properly to e.g. `$DATASET_TUSIMPLE`.
5. `cd tus_po_8p_dn19`
6. Execute training command on e.g. gpu with ID=1 `CUDA_VISIBLE_DEVICES="1" python ../yolino/src/yolino/train.py --gpu --loggers wb`
7. Open wandb site to watch your training. Link will be printed to cmd.

### Visualize Data Loading

1. Your configuration is already set with the given `params.yaml` from your configuration folder.
4. Use proper virtual environment with `workon <virtualenv-name>` or conda with `conda activate <conda-name>`.
2. Set your dataset paths properly to e.g. `$DATASET_TUSIMPLE`.
3. `cd tus_po_8p_dn19`
4. Execute visualization command for e.g. clips/0313-2/100/20.jpg
   `python ../yolino/src/yolino/show.py --explicit clips/0313-2/100/20.jpg`. By default all
   parameters from the params.yaml are taken. You might want to use no augmentation with `--augment ""`. If you wish to
   skim through the dataset leave the `--explicit` filename. Make sure to have access to the whole dataset or
   use `--ignore_missing`. With `--max_n` you can limit the number of files loaded.

### Hyperparameter Tuning

Use weights and biases: https://docs.wandb.ai/guides/sweeps

## Troubleshooting

- If the tests fail, make sure you have git lfs setup and all files in test/test_data are fetched properly
- If weights and biases complains about duplicate IDs after deleting some runs online, execute `wandb sync --clean` in
  your dvc folder and add (temporarily!) `id=wandb.util.generate_id()` to the initializing of your wandb connection. Run
  it once and delete that again.
- If installing the packages with pip does not work with conda, try using explicitly the conda pip/python executables in e.g. `<CONDA_HOME>/envs/<env_name>/bin/pip`

## Run on Horeka

https://www.scc.kit.edu/dienste/horeka.php
```
pip install virtualenv virtualenvwrapper
python3 -m venv ~/.virtualenvs/diss_yolino_venv
workon diss_yolino_venv
ml load devel/cuda/11.0
pip3 install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -e . # inside yolino repo
sbatch run_with_sbatch.sh
```

## CI and Docker (outdated)
The docker for CI jobs is generated every time the master is build and either the Dockerfile or the gitlab-ci.yaml script is changed. For a branch named 'docker' it is always run.
If you like to generate the docker file locally use the following commands to build and push
```
docker build -t gitlab.mrt.uni-karlsruhe.de:21443/mrt/private/meyer/publications/diss/yolino .
docker push gitlab.mrt.uni-karlsruhe.de:21443/mrt/private/meyer/publications/diss/yolino
```
