# GAMER - Graphs As Multi-modal intErmediate Representations


<div align="center">

[![paper](https://img.shields.io/badge/paper-black?logo=paperswithcode&style=for-the-badge)](https://youtube.com)
[![trained checkpoints](https://img.shields.io/badge/trained%20Checkpoints-black?logo=huggingface&style=for-the-badge)](https://huggingface.co)
[![experiments](https://img.shields.io/badge/experiments-black?logo=weightsandbiases&style=for-the-badge)](https://wandb.ai/the_GAMERs/GAMER/reports/Convergence-plots-for-GAMER-experiments--VmlldzoxNDM3NzQ3Ng)
</div>


**Here explain about project high level a couple of words/paragraphs.**


## File/Code structure

Our code is modular so many experiments could be easily integrated into the project:
- The `GAMER/data/VQA` directory contains the pre-processing pipeline for `VQA`, as well as `PyTorch Dataset` classes for `VQA`.
- The `GAMER/model/modules` directory contains all modules that make up the model (i.e. _"the pieces that work together"_).
- In `GAMER/model`, there are training and evaluation scripts that make use of all modules and pre-processed data.
- The `GAMER/model/encoders` directory contains examples of the encoders we used.

Since we experiment a lot with the graph structure, pre-processing it would result in long pre-processing times and very slow experimentation. Thus, the general code structure of this project follows as such:
1. Dataset is pre-processed to include embeddings and saved to disk.
2. Pre-processed dataset is wrapped in a `Dataset` class, building graphs from embeddings with a given function for each sample _"on-the-fly"_.
3. The model is a _"normal"_ `GNN` that is trained on this resulting dataset.

## Environment setup
We supply a conda environment for our project, aswell as a tester to run that would check if your environment has got everything to run the code for this project.
- To create the environment, run `conda env create -f GAMER/GAMER_env.yaml`. A conda environment with the name _"GAMER"_ should be created and can now be activated.
- To test an existing environment, run the first cell in `GAMER/env_tester.ipynb`. It should (not crash,) import everything and print versions.


## Inference on an existing checkpoint

## Training a model
As mentioned in the _'paper'_ pdf, we only train closed-answer, classification-based models.

To train a model, our pre-processing pipeline should be followed:
1. **Augment** the `VQA` dataset with embeddings for text and images.
2. **Extract** target classes from the `VQA` dataset, being the top-$k$ most appearing classes in the train set.
3. **Configure** the model you want to train.
4. **Train!**

We supply code for the entire pipeline, assuming the format of [this `VQA v2` dataset from 🤗](https://huggingface.co/datasets/pingzhili/vqa_v2) (which is standard, this should work on any respectable `VQA` distribution):
1. To **add embeddings** to the dataset, run `GAMER/data/VQA/augment_dataset.py` or `GAMER/data/VQA/augment_dataset_CLIP.py`. The first will do `BERT/BEiT` un-aligned embeddings, while the second will do `CLIP` aligned embeddings. A path to save the new augmented dataset should be provided in the last line of code, as the argument for `save_to_disk`.
2. To **extract the relevant classification targets**, run `GAMER/data/VQA/extract_targets.py`. A path to save the targets dictionary should be provided (a `.json` file) as the first argument to `open`. The default number of top candidates is $3000$, but can be changed in the code, as the argument for `answer_counter.most_common`.
3. To **configure the model's hyper-parameters**, fill in the relevant fields in `GAMER/model/config.py`, in the `get_model_config` function.
4. To **run training**, run `GAMER/model/train.py`. There are many hyper-parameters that can be tuned with command line flags, or in the `config.py` file. All can be viewed by running `GAMER/model/train.py --help`. \
One important hyper-parameter is `--graph-construction-method`, which controls the graph building function. Can be set to `mmg` or `cayley`. Graph construction is done outside of the model and the trained models are _"blind"_ to it, so one must be careful as to keep training aligned with inference, by using the same construction methods for both. \
 Chekpoints will be saved in the specified directory, and can be loaded as usual with `torch.load`, then `model.load_state_dict(ckpt['model'])`. \

 Graph construction is handled on-the-fly, as a part of the `Dataset` abstraction.

 ## Evaluating a trained checkpoint
 To evaluate a model, run the `GAMER/model/evaluate.py` script with the appropriate `config.py` file, specifying the model's parameters as-well as the relevant paths. **Make sure the model initialization and graph construction method matches that of the trained checkpoint**.

 The evaluation script will first run evaluation over the validation set of `VQA v2`, and report the mean [_`VQA` accuracy metric_](https://visualqa.org/evaluation.html). It will then run inference over the un-annotated test set, and produce a `.json` file containing the generated answers and question IDs, which is ready for submission to the [official `VQA` challenge](https://eval.ai/web/challenges/challenge-page/830/overview).