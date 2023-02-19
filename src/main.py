import transformers

from transformers import Trainer, TrainingArguments

from src import data_docile, model


def main():
    # transformers.utils.logging.set_verbosity_error()
    proc, m = model.get_model("layoutlmv3-base", proc_kwargs={"apply_ocr": False})
    data = data_docile.DocileDataset("val", proc)

    train_args = TrainingArguments(
        output_dir="tmp_trainer",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
    )
    trainer = Trainer(model=m, args=train_args)

    trainer.predict(data)
    print(sum(data_docile.runtimes) / len(data_docile.runtimes))


if __name__ == "__main__":
    main()
