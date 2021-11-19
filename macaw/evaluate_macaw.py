import numpy as np
import torch
from argparse import ArgumentParser

from datasets import (
    load_dataset,
    load_metric
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    Trainer
)


def evaluate(model_name):
    torch.cuda.empty_cache()
    raw_datasets = load_dataset("qanta", 'mode=full,char_skip=25')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def process_data_to_model_inputs(batch):
        inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1000)
        outputs = tokenizer(batch["page"], padding="max_length", truncation=True, max_length=30)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        return batch

    columns_to_remove = ["text", "page", "answer", "first_sentence",
                         "full_question", "proto_id",
                         "raw_answer", "difficulty", "sentence_idx",
                         "subcategory", "dataset", "id", "gameplay",
                         "year", "qdb_id", "char_idx", "tournament",
                         "category", "tokenizations", "qanta_id"]

    # tokenized_datasets = raw_datasets
    small_train_dataset = raw_datasets["guesstrain"].map(
        process_data_to_model_inputs,
        remove_columns=columns_to_remove,
        batched=True).shuffle(seed=42).select(range(10))
    small_eval_dataset = raw_datasets["guessdev"].map(
        process_data_to_model_inputs,
        remove_columns=columns_to_remove,
        batched=True).shuffle(seed=42).select(range(10))

    training_args = Seq2SeqTrainingArguments("test_trainer",
                                             save_strategy="epoch",
                                             no_cuda=True,
                                             resume_from_checkpoint=True)
    if args.do_train:
        trainer = Seq2SeqTrainer(
            model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset,

        )
        trainer.train()

        model.save_pretrained('./test/finetuned_macaw')
        tokenizer.save_pretrained('./test/finetuned_macaw')

    if args.do_evaluate:
        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = logits[0].argmax(axis=2)
            score = metric.compute(predictions=predictions.flatten(), references=labels.flatten())
            print(f"accuracy={score}")
            return score

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.evaluate()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="allenai/macaw-large", help="allenai macaw model to use")
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_evaluate', action='store_true', default=False)
    args = parser.parse_args()
    evaluate(args.model_name)
