import argparse


def get_parser() -> argparse.ArgumentParser:
    """
    Return a parser that parses the needed config for `train.py`
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ------------- #
    # === Paths === #
    # ------------- #

    parser.add_argument(
        "--dataset-path",
        type=str,
        default='/home/yandex/MLWG2025/danielvolkov/datasets/VQA_w_embed',
        help="Path of VQA dataset, augmented with question and image embeddings.",
    )
    parser.add_argument(
        "--ans2idx-path",
        type=str,
        default='/home/yandex/MLWG2025/danielvolkov/Documents/GAMER/data/VQA/answer2idx.json',
        help="Path of answer to index json file for VQA dataset classification.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default='/home/yandex/MLWG2025/danielvolkov/checkpoints',
        help="Directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for which training will be resumed. None to start fresh training.",
    )

    # ----------------------- #
    # === Training Params === #
    # ----------------------- #

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--grad-acc-steps",
        type=int,
        default=2,
        help="Number of batches over which to perform grad accumulation. set to 1 to disable. Namely - update model when batch_idx_train %% accumulate_grad_steps == 0.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum norm of gradient vectors for grad clipping.",
    )
    parser.add_argument(
        "--val-interval-updates",
        type=int,
        default=100,
        help="Number of model updates after which to perform validation.",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=100,
        help="Maximum number of batches to validate over. None to do full validation.",
    )
    parser.add_argument(
        "--checkpoint-interval-updates",
        type=int,
        default=100,
        help="Number of model updates after which to save a checkpoint.",
    )
    parser.add_argument(
        "--log-every-n-updates",
        type=int,
        default=10,
        help="Number of model updates after which to log stats to wandb.",
    )
    parser.add_argument(
        "--save-best",
        type=bool,
        default=True,
        help="Wether or not to save the best validation score checkpoint.",
    )
    parser.add_argument(
        "--use-amp",
        type=bool,
        default=True,
        help="Wether or not to use AMP (Auto Mixed Precision).",
    )

    # --------------------------------------------------------------- #
    # === Data Params (Graph construction, technichalities, etc.) === #
    # --------------------------------------------------------------- #

    parser.add_argument(
        "--add-lap-pe",
        type=bool,
        default=True,
        help="Wether or not to add Laplacian positional encoding to graphs.",
    )
    parser.add_argument(
        "--lap-pe-dim",
        type=int,
        default=16,
        help="Dimension of Laplacian PE. Used only if add-lap-pe is True.",
    )
    parser.add_argument(
        "--graph-construction-method",
        type=str,
        default='mmg',
        help="Method used to construct graphs from image and text embeddings. Options are: mmg, cayley",
    )
    parser.add_argument(
        "--num-fusion-nodes",
        type=int,
        default=6,
        help="Used only for mmg construction. Number of fusion nodes.",
    )
    parser.add_argument(
        "--num-text-global-nodes",
        type=int,
        default=2,
        help="Used only for mmg construction. Number of text-global nodes.",
    )
    parser.add_argument(
        "--self-loops-in-image-graph",
        type=bool,
        default=False,
        help="Used only for mmg construction. Wether or not to include self-loops in the image grid.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker thread for DataLoader parallelization.",
    )
    parser.add_argument(
        "--pin-memory",
        type=bool,
        default=True,
        help="Pin memory for DataLoaders and moving of batches to GPU.", # An optimization to move straight to GPU memory instead of passing through swap memory
    )
    parser.add_argument(
        "--persistent-workers",
        type=bool,
        default=True,
        help="Wether or not to use persistent workers with DataLoaders.",
    )

    return parser


def get_model_config():
    return dict(
        hidden_dim = 768,	            # idk man just match BERT or smth seems legit
        num_layers = 5,	                # Graph is connected with max 5 hop distance
        heads = 8,         	            # Random ass number that seems cool
        dropout = 0.1,		            # idk man wtf is dropout
        mlps_hidden_layers = 2,         # THICC MLPS
        readout_method = 'mean'
    )