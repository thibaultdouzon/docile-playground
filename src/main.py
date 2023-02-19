from transformers import Trainer, TrainingArguments

from src import data_docile, model


def main():
    proc, m = model.get_model("layoutlmv3-base", proc_kwargs={"apply_ocr": False})
    data = data_docile.DocileDataset("val", proc)

    train_args = TrainingArguments(output_dir="tmp_trainer", no_cuda=True)
    trainer = Trainer(model=m, args=train_args)

    trainer.predict(data)


if __name__ == "__main__":
    main()
