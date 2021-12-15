from argparse import ArgumentParser

import torch
from datasets import (
    load_dataset
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def evaluate(model_name):
    torch.cuda.empty_cache()
    raw_datasets = load_dataset('csv',
                                data_files={'train': [args.train_input], 'validation': [args.validation_input]},
                                delimiter='\t',
                                column_names=['question', 'answer'],
                                skiprows=1)

    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets['validation']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocess the batches
    def preprocess_qanta_dataset(batch):
        input_strings = []
        output_strings = []

        for idx in range(len(batch['question'])):
            if not batch["answer"][idx]:
                batch["answer"][idx] = ""

            macaw_input = "$answer$ ; $question$ = " + batch["question"][idx]
            macaw_output = "$answer$ = " + batch["answer"][idx]

            input_strings.append(macaw_input)
            output_strings.append(macaw_output)

        inputs = tokenizer(input_strings, padding="max_length", truncation=True, max_length=256)
        outputs = tokenizer(output_strings, padding="max_length", truncation=True, max_length=30)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        return batch

    # Print out predictions for a given batch from eval dataset
    def predict(batch):
        for idx in range(len(batch['question'])):
            input_string = "$answer$; $question$ = " + batch["question"][idx]
            input_ids = tokenizer.encode(input_string, return_tensors="pt")
            output = model.generate(input_ids, max_length=200)
            pred = tokenizer.batch_decode(output, skip_special_tokens=True)
            print(f"question={batch['question'][idx]}")
            print(f"prediction={pred}")
            print(f"label={batch['answer'][idx]}")

        return batch



    columns_to_remove = ["question", "answer"]
    small_train_dataset = train_dataset.select(range(32)).map(
        preprocess_qanta_dataset,
        batched=True,
        batch_size=4,
        remove_columns=columns_to_remove
    ).shuffle(seed=42)

    small_eval_dataset = eval_dataset.select(range(32)).map(
        preprocess_qanta_dataset,
        batched=True,
        batch_size=4,
        remove_columns=columns_to_remove
    ).shuffle(seed=42)

    training_args = Seq2SeqTrainingArguments(
        "test_trainer",
        save_strategy="epoch",
        no_cuda=True,
        resume_from_checkpoint=True
    )

    if args.do_train:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
        )

        trainer.train()

        model.save_pretrained(args.save_model_path)
        tokenizer.save_pretrained(args.save_model_path)

    if args.do_evaluate:
        eval_dataset.select(range(32)).map(
            predict,
            batched=True,
            batch_size=4,
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="allenai/macaw-large", help="allenai macaw model to use")
    parser.add_argument('--save_model_path', type=str, default="./test/finetuned_macaw",
                        help="Place to save the finetuned model")
    parser.add_argument('--train_input', type=str, help="Training input TSV file")
    parser.add_argument('--validation_input', type=str, help="Validation input TSV file")
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_evaluate', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args.model_name)
