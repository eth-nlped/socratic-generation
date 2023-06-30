import os

from train import Run as TrainRun
from planning_train import Run as PlanningRun
from sample import Run as SampleRun
from convert_results import main as convert_to_one_row_per_mwp
from evaluate import main as evaluate

if __name__ == "__main__":
    # 1. Train
    if not os.environ.get("CRITIC", None):
        train_runner = TrainRun("train")
        if int(os.environ.get("EPOCHS", 30)) > 0:
            print("Running training")
            model_path = train_runner.train()
            TRAINED_MODEL_PATH = os.path.join(model_path, "best_valid")
        else:
            # Continue with pre-trained model only
            TRAINED_MODEL_PATH = os.path.join(train_runner.config['PRETRAINED_MODEL_PREFIX_PATH'],
                                              train_runner.config['MODEL_NAME'])
    else:
        critic_train_run = PlanningRun("train")
        model_path = critic_train_run.train()
        CRITIC_MODEL_PATH = os.path.join(model_path, "final")
        TRAINED_MODEL_PATH = os.environ["MODEL_IMPORT_PATH"]

    # 2. Inference
    print(f"Running inference with {TRAINED_MODEL_PATH}")
    sample_runner = SampleRun("test")
    # best_valid or final
    sample_runner.config["MODEL_IMPORT_PATH"] = TRAINED_MODEL_PATH

    if os.environ.get("CRITIC", None):
        sample_runner.config["MODEL_IMPORT_PATH"] = TRAINED_MODEL_PATH
        sample_runner.config["CRITIC_MODEL_PATH"] = CRITIC_MODEL_PATH
        print(f"Running inference with outputs from a critic / planner model with {CRITIC_MODEL_PATH}")

    EVALUATE_PATH = sample_runner.test()

    # 3. Conversion
    print(f"Running conversion of results with {EVALUATE_PATH}")
    file_to_evaluate = convert_to_one_row_per_mwp(EVALUATE_PATH)

    # 4. Evaluation
    print(f"Running evaluation report with {file_to_evaluate}")
    evaluate(file_to_evaluate)

