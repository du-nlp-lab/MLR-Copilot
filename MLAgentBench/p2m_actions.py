import os
import datasets
from .schema import ActionInfo, EnvException

from .prompt2model.prompt_parser import MockPromptSpec, TaskType
from .prompt2model.dataset_retriever import DescriptionDatasetRetriever
from .prompt2model.dataset_generator import PromptBasedDatasetGenerator, DatasetSplit
from .prompt2model.dataset_processor import TextualizeProcessor
from .prompt2model.model_retriever import DescriptionModelRetriever
from .prompt2model.model_trainer import GenerationModelTrainer

def generate_dataset(instruction, examples, save_dir, num_train, num_valid, num_test, work_dir = '.'):
    try:
        num_train = int(num_train)
        num_valid = int(num_valid)
        num_test = int(num_test)
    except ValueError:
        raise EnvException("Number of examples should be an integer")

    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, instruction=instruction, examples=examples)
    generator = PromptBasedDatasetGenerator()
    dataset_dict = generator.generate_dataset_dict(prompt_spec, {
        DatasetSplit.TRAIN: num_train,
        DatasetSplit.VAL: num_valid,
        DatasetSplit.TEST: num_test
    })

    save_path = os.path.join(work_dir, save_dir)
    dataset_dict.save_to_disk(save_path)

    return f"Dataset successfully generated and saved to {save_path}"

def retrieve_dataset(instruction, save_dir, work_dir = '.'):
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, instruction=instruction, examples="")
    retriever = DescriptionDatasetRetriever()
    dataset_dict = retriever.retrieve_dataset_dict(prompt_spec)

    save_path = os.path.join(work_dir, save_dir)
    dataset_dict.save_to_disk(save_path)

    return f"Dataset successfully generated and saved to {save_path}"

def retrieve_model(instruction, work_dir = '.'):
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, instruction=instruction, examples="")
    retriever = DescriptionModelRetriever(use_bm25=True, use_HyDE=True)
    top_models = retriever.retrieve(prompt_spec)

    return "Top Models:\n" + "".join(f"{i+1}. {model}\n" for i, model in enumerate(top_models))

def process_dataset(instruction, load_dirs, save_dirs, work_dir = '.'):
    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION, instruction=instruction, examples="")
    load_dirs = load_dirs.split(':')
    save_dirs = save_dirs.split(':')
    if len(load_dirs) != len(save_dirs):
        raise EnvException("Number of load directories should match number of save directories")
    load_paths = [os.path.join(work_dir, load_dir) for load_dir in load_dirs]
    save_paths = [os.path.join(work_dir, save_dir) for save_dir in save_dirs]

    # load the datasets
    dataset_dicts = [datasets.load_from_disk(load_path) for load_path in load_paths]

    # process the datasets
    processor = TextualizeProcessor(has_encoder=True)
    modified_dataset_dicts = processor.process_dataset_dict(prompt_spec, dataset_dicts)

    # save the processed datasets
    for dataset_dict, save_path in zip(modified_dataset_dicts, save_paths):
        dataset_dict.save_to_disk(save_path)

    return f"Data successfully processed and saved to {save_paths}"

def train_model(model_name, load_dirs, result_dir, epochs, batch_size, warmup_steps, weight_decay, learning_rate, work_dir = '.'):
    try:
        epochs = int(epochs)
        batch_size = int(batch_size)
        warmup_steps = int(warmup_steps)
        weight_decay = float(weight_decay)
        learning_rate = float(learning_rate)
    except ValueError:
        raise EnvException("Numerical parameters should be integers or floats as appropriate")

    result_dir = os.path.join(work_dir, result_dir)

    # load the datasets
    load_paths = [os.path.join(work_dir, load_dir) for load_dir in load_dirs]
    dataset_dicts = [datasets.load_from_disk(load_path) for load_path in load_paths]

    training_datasets = [dataset_dict["train"] for dataset_dict in dataset_dicts]
    validation_datasets = [dataset_dict["validation"] for dataset_dict in dataset_dicts]
        
    trainer = GenerationModelTrainer(
        model_name,
        has_encoder=True,
        executor_batch_size=batch_size,
        tokenizer_max_length=1024,
        sequence_max_length=1280,
    )

    hparams ={
        "output_dir": os.path.join(result_dir, "training_output"),
        "save_strategy": "epoch",
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "evaluation_strategy": "epoch",
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "learning_rate": learning_rate,
    },

    trained_model, trained_tokenizer = trainer.train_model(
        hyperparameter_choices=hparams,
        training_datasets=training_datasets,
        validation_datasets=validation_datasets,
    )

    trained_model.save_pretrained(os.path.join(result_dir, "trained_model"))
    trained_tokenizer.save_pretrained(os.path.join(result_dir, "trained_tokenizer"))

    return f"Model and Tokenizer successfully trained and saved respectively to {result_dir}/trained_model and {result_dir}/trained_tokenizer"

P2M_ACTIONS = [
    ActionInfo(
        name="Generate Dataset",
        description="Generate a dataset based on an instruction and examples. You can load the dataset later from `save_dir` using the load_from_disk function of the HuggingFace datasets library.",
        usage={
            "instruction": "an instruction on how to generate the output from the input",
            "examples": "examples of input-output pairs",
            "save_dir": "directory to save the generated dataset dict to. We recommend saving to data/generated/",
            "num_train": "number of examples to generate in the training set",
            "num_valid": "number of examples to generate in the validation set",
            "num_test": "number of examples to generate in the test set",
        },
        return_value="The observation will be a success message if the dataset was generated successfully. Otherwise, an error message will be returned.",
        is_primitive=False,
        function=generate_dataset
    ),
    ActionInfo(
        name="Retrieve Dataset",
        description="Retrieve a suitable dataset based on a detailed description of the requirements. You can load the dataset later from `save_dir` using the load_from_disk function of the HuggingFace datasets library.",
        usage={
            "instruction": "an instruction on how to generate the output from the input",
            "save_dir": "directory to save the generated dataset dict to. We recommend saving to data/retrieved/",
        },
        return_value="The observation will be a success message if the dataset was retrieved successfully. Otherwise, an error message will be returned.",
        is_primitive=False,
        function=retrieve_dataset
    ),
    ActionInfo(
        name="Retrieve Model",
        description="Retrieve a suitable model based on a detailed description of the requirements. You can obtain the model given the name using the transformers.AutoModelForSeq2SeqLM.from_pretrained function.",
        usage={
            "instruction": "an instruction on how to generate the output from the input",
        },
        return_value="The observation will be a list of suitable models. You can choose one of them based on the requirements.",
        is_primitive=False,
        function=retrieve_model
    ),
    ActionInfo(
        name="Process Dataset",
        description="Process dataset based on a detailed description of the requirements. You can load the processed data later from `save_dirs` using the load_from_disk function of the HuggingFace datasets library.",
        usage={
            "instruction": "an instruction on how to generate the output from the input",
            "load_dirs": "directories to load the dataset dicts from, separated by colons",
            "save_dirs": "directories to save the processed dataset dicts to, separated by colons. The order should match the order of the loaded datasets. We recommend saving to data/processed/",
        },
        return_value="The observation will be a success message if the data was processed successfully. Otherwise, an error message will be returned.",
        is_primitive=False,
        function=process_dataset
    ),
    ActionInfo(
        name="Train Model",
        description="Train a Seq2Seq model from HuggingFace transformers library using the processed datasets and given hyperparameters.",
        usage={
            "model_name": "name of the model to train",
            "load_dirs": "directories to load the dataset dicts from, separated by colons",
            "result_dir": "directory to save the trained model and tokenizer to. We recommend using results/{trial_id}/. The trained model will be available as `{result_dir}/trained_model/` and the tokenizer will be available as `{result_dir}/trained_tokenizer/`.",
            "epochs": "number of epochs to train the model for",
            "batch_size": "batch size for training the model",
            "warmup_steps": "number of warmup steps for the optimizer",
            "weight_decay": "weight decay for the optimizer",
            "learning_rate": "learning rate for the optimizer",
        },
        return_value="The observation will be a success message if the model was trained successfully. Otherwise, an error message will be returned.",
        is_primitive=False,
        function=train_model
    ),
    ActionInfo("Execute Model", )
]