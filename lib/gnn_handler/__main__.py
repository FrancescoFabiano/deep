import argparse
import os

import torch

from src.preprocessing import GraphDataPipeline
from src.utils import (
    get_dataloaders,
    prepare_samples,
    print_values,
    seed_everything,
    select_model,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or build data for the distance estimator model."
    )
    # simple string / numeric args
    parser.add_argument(
        "--subset-train",
        action="extend",  # collect all values into one list
        nargs="+",  # each occurrence takes 1+ args
        type=str,
        default=[],
        help="Name(s) of problem subsets to use for training",
    )
    parser.add_argument(
        "--model-name",
        default="distance_estimator",
        type=str,
        help="Name for the distance estimator model",
    )
    parser.add_argument(
        "--normalization-constants-name",
        default="C",
        type=str,
        help="Name of normalization constants txt file, helpful for rescaling the output of regressor",
    )
    parser.add_argument(
        "--folder-raw-data",
        type=str,
        default="out/NN/Training",
        help="Where to find/build the data",
    )
    parser.add_argument(
        "--unreachable-state-value",
        type=int,
        default=1000000,
        help="Value to use for unreachable states",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of held-out test data"
    )
    parser.add_argument(
        "--max-percentage-per-class",
        type=float,
        default=0.5,
        help="Highest possible percentage for one class in the target variable.",
    )
    parser.add_argument(
        "--dir-save-data",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--dir-save-model",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Name of experiment with which data and models will be stored",
    )

    parser.add_argument(
        "--n-train-epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Training batch size"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random State")

    # boolean flags
    parser.add_argument(
        "--build-data",
        type=str2bool,
        default=True,
        help="Whether to (re)build the data",
    )
    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the models"
    )
    parser.add_argument(
        "--if-try-example",
        type=str2bool,
        default=False,
        help="Inference with pytorch and onnx model on example samples",
    )

    parser.add_argument(
        "--kind-of-ordering",
        type=str,
        choices=["hash", "map"],
        default="hash",
        help="Ordering strategy to consider",
    )
    parser.add_argument(
        "--kind-of-data",
        type=str,
        choices=["merged", "separated"],
        default="merged",
        help="Data split type to use",
    )

    parser.add_argument(
        "--use-goal",
        type=str2bool,
        default=False,
        help="Whether to include goal info (true/false)",
    )
    parser.add_argument(
        "--use-depth",
        type=str2bool,
        default=False,
        help="Whether to include depth info (true/false)",
    )

    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to print detailed eval errors",
    )

    args = parser.parse_args()

    return args


def main(args):
    seed = args.seed
    seed_everything(seed)

    list_subset_train = args.subset_train
    if_build_data = args.build_data
    if_train = args.train
    kind_of_ordering = args.kind_of_ordering
    kind_of_data = args.kind_of_data
    use_goal = args.use_goal
    use_depth = args.use_depth
    if_try_example = args.if_try_example

    experiment_name = args.experiment_name
    folder_raw_data = args.folder_raw_data

    path_save_model = args.dir_save_model
    path_save_data = args.dir_save_data

    model_name = args.model_name
    normalization_constants_name = args.normalization_constants_name

    unreachable_state_value = args.unreachable_state_value
    test_size = args.test_size
    max_percentage_per_class = args.max_percentage_per_class

    n_train_epochs = args.n_train_epochs
    batch_size = args.batch_size

    verbose = args.verbose

    path_data = path_save_data
    if experiment_name != "":
        path_data += "/" + experiment_name
    # /{kind_of_ordering}_{kind_of_data}"
    # path_save_data += "_goal" if use_goal else "_no_goal"
    # path_save_data += "_depth" if use_depth else "_no_depth"
    os.makedirs(path_data, exist_ok=True)
    data_path = path_data + "/samples.pt"

    print("\n************************************************")
    print(
        f"subset_train: {list_subset_train} | {kind_of_ordering} | {kind_of_data} | Use goal: {use_goal} | Use depth: {use_depth} | Model name: {model_name} | Train: {if_train} | Build Data: {if_build_data}",
    )

    if if_build_data:
        pipe = GraphDataPipeline(
            folder_data=folder_raw_data,
            list_subset_train=list_subset_train,
            kind_of_ordering=kind_of_ordering,
            kind_of_data=kind_of_data,
            unreachable_state_value=unreachable_state_value,
            max_percentage_per_class=max_percentage_per_class,
            test_size=test_size,
            use_goal=use_goal,
            use_depth=use_depth,
            random_state=seed,
        )

        pipe.save(
            out_dir=data_path,
            extra_params={},
        )

    data = torch.load(data_path, weights_only=False)
    train_samples = data["train_samples"]
    test_samples = data["test_samples"]

    train_samples_copy = train_samples.copy()
    test_samples_copy = test_samples.copy()

    train_samples_copy, test_samples_copy, params_f = prepare_samples(
        train_samples_copy, test_samples_copy, unreachable_state_value
    )

    if verbose:
        print("Train values:")
        print_values(train_samples_copy)
        print("Test values:")
        print_values(test_samples_copy)
        print("\n")
        print("Normalization parameters: ", params_f)

    train_loader, val_loader = get_dataloaders(
        train_samples_copy,
        test_samples_copy,
        batch_size=batch_size,
    )

    path_model = path_save_model
    if experiment_name != "":
        path_model += "/" + experiment_name

    # instantiate
    m = select_model(model_name, use_goal, use_depth)

    # train
    if if_train:
        m.train(
            train_loader,
            val_loader,
            n_epochs=n_train_epochs,
            checkpoint_dir=path_model,
            model_name=model_name,
        )

    # load
    m.load_model(f"{path_model}/{model_name}.pt")

    with open(
        f"{path_model}/{model_name}_{normalization_constants_name}.txt",
        "w",
        encoding="utf-8",
    ) as fh:
        for key, value in params_f.items():
            fh.write(f"{key} = {value}\n")

    kwargs = {"th": params_f["slope"] / 2}

    m.evaluate(val_loader, verbose=verbose, **kwargs)

    onnx_model_path = f"{path_model}/{model_name}.onnx"
    m.to_onnx(onnx_model_path, use_goal, use_depth)

    if if_try_example:
        example_state_to_predict = (
            f"./examples/{kind_of_ordering}_{kind_of_data}_state.dot"
        )
        example_goal = "./examples/goal_tree.dot"
        example_depth = 5

        out = m.predict_single(
            example_state_to_predict,
            depth=example_depth if use_depth else None,
            goal_dot=example_goal if use_goal else None,
        )

        print("PyTorch output: ", out)

        sss = [example_state_to_predict, example_state_to_predict]
        ggg = [example_goal, example_goal]
        ddd = [example_depth, example_depth]

        out_onnx = m.try_onnx(
            onnx_model_path,
            sss,
            depths=ddd if use_depth else None,
            goal_dot_files=ggg if use_goal else None,
        )

        print("Out onnx: ", out_onnx)

        true_val = 0 * params_f["slope"] + params_f["intercept"]
        print("True rescaled distance: ", true_val, " -> True Distance: 0")

    with open(
        f"{path_model}/{model_name}_info.txt",
        "w",
        encoding="utf-8",
    ) as fh:
        for name, value in vars(args).items():
            fh.write(f"{name} = {value}\n")

    return onnx_model_path


if __name__ == "__main__":
    args = parse_args()
    main(args)
