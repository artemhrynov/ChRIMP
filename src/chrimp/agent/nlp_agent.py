import os
import torch
from pathlib import Path
import pickle
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import wandb
from tqdm import tqdm
from time import time
from torch import cuda
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from rdkit.Chem import MolFromSmiles, MolToSmiles

from collections import defaultdict
import pandas as pd
import numpy as np
from colorama import Fore, Style
from abc import ABC, abstractmethod

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed

from chrimp.agent.supervised_evaluate import evaluation_metrics_w_groundtruth, answers_dict
from chrimp.agent.supervised_search import search

def human_to_input(
    rxn_smiles:str, # rSMILES in the form of 'SMILES_REAC>SMILES_COND>SMILES_PROD'
    format:str, # any of "retro", "forward", "reac"
    already_canonicalized: bool,
):
    reac, cond, prod = rxn_smiles.split('>')

    if not already_canonicalized:
        old_reac = reac
        old_prod = prod
        reac = MolToSmiles(MolFromSmiles(reac, sanitize=False) ,kekuleSmiles=True)
        prod = MolToSmiles(MolFromSmiles(prod, sanitize=False) ,kekuleSmiles=True)

    if format == "forward":
        return f"[reac]{reac}[prod]{prod}[mech]"
    elif format == "retro":
        return f"[prod]{prod}[reac]{reac}[mech]"
    elif format == "reac":
        return f"[reac]{reac}[mech]"

def _worker_function(
    args,
    reac_column,
    prod_column,
    code_mechsmiles_output,
    with_metadata,
    input_format,
    column_names,
):
    """Top-level worker function that can be pickled"""
    import signal
    
    class TaskTimeout(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TaskTimeout()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    try:
        idx, row = args
        reac_value = row[reac_column]
        prod_value = row[prod_column]
        output = row[f"mech_smi_{code_mechsmiles_output}"]
        
        human_input = f"{reac_value}>>{prod_value}"
        entry = {
            'row_index': idx,
            'human_input': human_input,
            'formatted_inputs': {},
            'formatted_output': output,
        }
        
        if with_metadata:
            entry['metadata'] = {c: row[c] for c in column_names}
        
        for format_type in input_format.split(';'):
            try:
                formatted_input = human_to_input(human_input, format_type, already_canonicalized=False)
                entry['formatted_inputs'][format_type] = formatted_input
            except Exception as e:
                print(f"Warning: Failed to format input for format '{format_type}' at row {idx}: {e}")
                entry['formatted_inputs'][format_type] = None
        
        return entry, None
    except TaskTimeout:
        print(f"Timeout with {args = }")
        return None, "Timeout 5s"
    except Exception as e:
        print(f"Exception {e} with {args}")
        return None, str(e)
    finally:
        signal.alarm(0)

class NLPAgent(ABC):
    def __init__(
        self,
        is_decoder_only:bool,

        inference_input_format:str, # any of "retro", "forward", "reac" (But only one)
    ):
        self.is_decoder_only = is_decoder_only
        self.tokenizer = None
        self.model = None
        self.inference_input_format = inference_input_format

    @abstractmethod
    def init_model(
        self,
        tokenizer_path: str,
        param_dict: dict,
    ):
        pass

    @abstractmethod
    def load_model(
        self,
        tokenizer_path: str,
        model_path: str,
    ):
        pass

    def human_to_input(
        self,
        rxn_smiles:str, # rSMILES in the form of 'SMILES_REAC>SMILES_COND>SMILES_PROD' where SMILES_COND can be ""
        format:str, # any of "retro", "forward", "reac"
        already_canonicalized: bool,
    ):
        reac, cond, prod = rxn_smiles.split('>')

        if not already_canonicalized:
            reac = MolToSmiles(MolFromSmiles(reac, sanitize=False) ,kekuleSmiles=True)
            prod = MolToSmiles(MolFromSmiles(prod, sanitize=False) ,kekuleSmiles=True)

        if format == "forward":
            return f"[reac]{reac}[prod]{prod}[mech]"
        elif format == "retro":
            return f"[prod]{prod}[reac]{reac}[mech]"
        elif format == "reac":
            return f"[reac]{reac}[mech]"

    def process_dataset_for_nlp(self, dataset, task, input_format, code_mechsmiles_output:str = "min", with_metadata: bool=False):
        """
        Process a HuggingFace dataset to create training inputs for an agent.

        3 letters codes:
        - ori: original
        - min: minimal
        - equ: equilibrated
        - spe: only species no stoichio
        
        Args:
            agent: NLPAgent
            dataset: HuggingFace dataset
            task: Task string in format "rrr_ppp_n"
            input_format: inputs desired "forward", "retro", "reac" or several divided by ';' character
            code_mechsmiles_output: 3 letters code of the MechSMILES desired as an output
        
        Returns:
            List of dictionaries containing human_input and formatted inputs for each format
        """
        # Parse the task string
        parts = task.split('_')
        if len(parts) != 3:
            raise ValueError(f"Task format should be 'xxx_yyy_z', got: {task}")
        
        rrr, ppp, n = parts
        

        # rrr: 3 letters code for reactant
        # ppp: 3 letters code for product
        #   n: lenght of questions, 1 = elementary steps, n = single steps
        
        # Column names based on parsed task
        reac_column = f"elem_reac_{rrr}"
        
        if n == '1':
            prod_column = f"elem_prod_{ppp}"
        elif n == 'n':
            prod_column = f"rxn_prod_{ppp}"
        else:
            raise ValueError(f"z value should be '1' or 'n', got: {n}")
        
        training_data = []
        
        # Process each row in the dataset
        for idx, row in enumerate(dataset):
            try:
                # Extract values from the specified columns
                reac_value = row[reac_column]
                prod_value = row[prod_column]
                output = row[f"mech_smi_{code_mechsmiles_output}"]
                
                # Construct human input
                human_input = f"{reac_value}>>{prod_value}"
                # Create entry for this row

                entry = {
                    'row_index': idx,
                    'human_input': human_input,
                    'formatted_inputs': {},
                    'formatted_output': output,
                }

                if with_metadata:
                    entry['metadata'] = {c:row[c] for c in dataset.column_names}# Every row that is not already encoded above
                
                for format_type in input_format.split(';'):
                    try:
                        formatted_input = self.human_to_input(human_input, format_type, already_canonicalized=False)
                        entry['formatted_inputs'][format_type] = formatted_input
                    except Exception as e:
                        print(f"Warning: Failed to format input for format '{format_type}' at row {idx}: {e}")
                        entry['formatted_inputs'][format_type] = None
                
                training_data.append(entry)
                
            except KeyError as e:
                print(f"Warning: Missing column {e} in row {idx}, skipping this row")
                continue
            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
                continue

        # Then rewrite this data into a HuggingFace dataset with columns 'input', 'output'
        inputs = []
        outputs = []
        metadatas = []
        
        for entry in training_data:
            formatted_output = entry['formatted_output']
            if with_metadata:
                metadata = entry['metadata']
            
            for format_type, formatted_input in entry['formatted_inputs'].items():
                if formatted_input is not None:  # Skip failed formatting attempts
                    inputs.append(formatted_input)
                    outputs.append(formatted_output)
                    if with_metadata:
                        metadatas.append(metadata)
        
        # Create HuggingFace dataset
        if with_metadata:
            return Dataset.from_dict({
                'input': inputs,
                'output': outputs,
                'metadata': metadatas
            })
        else:
            return Dataset.from_dict({
                'input': inputs,
                'output': outputs
            })

    def process_dataset_for_nlp_parallel(self, dataset, task, input_format, code_mechsmiles_output:str = "min", with_metadata: bool=False):
        """
        Process a HuggingFace dataset to create training inputs for an agent.

        3 letters codes:
        - ori: original
        - min: minimal
        - equ: equilibrated
        - spe: only species no stoichio
        
        Args:
            agent: NLPAgent
            dataset: HuggingFace dataset
            task: Task string in format "rrr_ppp_n"
            input_format: inputs desired "forward", "retro", "reac" or several divided by ';' character
            code_mechsmiles_output: 3 letters code of the MechSMILES desired as an output
        
        Returns:
            List of dictionaries containing human_input and formatted inputs for each format
        """
        # Parse the task string
        parts = task.split('_')
        if len(parts) != 3:
            raise ValueError(f"Task format should be 'xxx_yyy_z', got: {task}")
        
        rrr, ppp, n = parts
        

        # rrr: 3 letters code for reactant
        # ppp: 3 letters code for product
        #   n: lenght of questions, 1 = elementary steps, n = single steps
        
        # Column names based on parsed task
        reac_column = f"elem_reac_{rrr}"
        
        if n == '1':
            prod_column = f"elem_prod_{ppp}"
        elif n == 'n':
            prod_column = f"rxn_prod_{ppp}"
        else:
            raise ValueError(f"z value should be '1' or 'n', got: {n}")
        

        # Prep rows once (faster than iterrows)
        idx_rows = list((idx, row) for (idx, row) in enumerate(dataset))

        worker = partial(
            _worker_function,
            reac_column=reac_column,
            prod_column=prod_column,
            code_mechsmiles_output=code_mechsmiles_output,
            with_metadata=with_metadata,
            input_format=input_format,
            column_names=dataset.column_names
        )

        with ProcessPoolExecutor(max_workers=20) as ex:
            results = list(ex.map(worker, idx_rows))

        training_data = [r[0] for r in results if r[1] is None]

        # Then rewrite this data into a HuggingFace dataset with columns 'input', 'output'
        inputs = []
        outputs = []
        metadatas = []
        
        for entry in training_data:
            formatted_output = entry['formatted_output']
            if with_metadata:
                metadata = entry['metadata']
            
            for format_type, formatted_input in entry['formatted_inputs'].items():
                if formatted_input is not None:  # Skip failed formatting attempts
                    inputs.append(formatted_input)
                    outputs.append(formatted_output)
                    if with_metadata:
                        metadatas.append(metadata)
        
        # Create HuggingFace dataset
        if with_metadata:
            return Dataset.from_dict({
                'input': inputs,
                'output': outputs,
                'metadata': metadatas
            })
        else:
            return Dataset.from_dict({
                'input': inputs,
                'output': outputs
            })

    def supervised_train(
        self,
        # Trainer parameters
        task: str,
        # Formats used during training
        train_input_format:str, # any of "forward", "retro", "reac", or multiple split by ';'
        max_learning_rate: float,
        num_train_epochs: int,
        dataset_path: str, # Can be multiple split by ';'
        dataset_from_hub: bool,
        max_token_length: int, # Maximum token length for input sequences
        number_saves: int,
        # optional
        batch_size:int = 64,
        dataset_name: str | None = None, # If None, we'll take the string after last '/'
    ):
        
        if self.tokenizer is None or self.model is None:
            raise ValueError(f"Tokenizer ({self.tokenizer}) or model ({self.model}) cannot be None, please initialize or load them before training")

        self.model.train()

        if dataset_name is None:
            dataset_name = ';'.join([d.split('/')[-1] for d in dataset_path.split(';')])
            print(f"Dataset name by default: {dataset_name}")

        train_input_format_list = train_input_format.split(';')
        train_input_format_list.sort()
        train_input_format = ';'.join(train_input_format_list)
        if self.inference_input_format not in train_input_format_list:
            print(f"{Fore.LIGHTYELLOW_EX}{self.inference_input_format = } is not seen during training ({train_input_format = }). This might lead to large performance loss.{Fore.RESET}")

        run = wandb.init(
            entity="liac",
            project="debug_mechsmiles_agents_training",
        )

        # Some wandb logging parameters
        wandb.config.update({
            "dataset_name": dataset_name,
            "task": task,
            "train_input_format": train_input_format,
            "num_train_epochs": num_train_epochs,
            "max_token_length": max_token_length,
            "vocab_size": self.tokenizer.vocab_size,
            "model_size": self.model.num_parameters(),
        })

        try:
            wandb.config.update({
                "model_config": self.config.to_dict(),
            })
        except AttributeError:
            wandb.config.update({
                "model_config": "from_pretrained",
            })

        tokenized_train_dataset, tokenized_val_dataset = self.import_process_tokenize_dataset(
            dataset_path = dataset_path,
            dataset_from_hub = dataset_from_hub,
            splits = ['train', 'val'],
            task = task,
            input_format = train_input_format,
            max_token_length = max_token_length,
        )

        self.sanity_check_datasets(
            tokenized_train_dataset,
            tokenized_val_dataset,
            max_token_length = max_token_length,
        )

        print(f"Tokenizer's pad token is: {self.tokenizer.pad_token}")

        eval_and_save_epochs = num_train_epochs // number_saves
        steps_per_epoch = len(tokenized_train_dataset) // batch_size
        eval_and_save_steps = steps_per_epoch * eval_and_save_epochs

        run_int = run.name.split('-')[-1]
        train_args = TrainingArguments(
            seed=22,
            warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            lr_scheduler_type="cosine",
            logging_steps=50,
            save_steps=eval_and_save_steps,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=eval_and_save_steps,
            learning_rate=max_learning_rate,
            report_to="wandb",
            num_train_epochs=num_train_epochs,
            output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_hf_train" if run_int is None else f"checkpoints_hf_train_debug_{run_int}"),
        )

        if self.is_decoder_only:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False, # Masked language modeling is not used for GPT-2
                return_tensors="pt",
            )
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                return_tensors="pt"
            )

        trainer = Trainer(
            model = self.model,
            args = train_args,
            train_dataset = tokenized_train_dataset,
            eval_dataset = tokenized_val_dataset,
            data_collator = data_collator,
        )

        trainer.train()

    def supervised_eval(
        self,
        task: str,
        dataset_path: str,
        dataset_from_hub: bool,
        max_token_length: int, # Maximum token length for input sequences
        num_datapoints: int = -1,
        num_beams: int = 20,
        batch_size: int = 16,
        parquet_file_path: str | None = None,
        generate_only: bool = False,
        score_only: bool = False,
        n_batch_score: int = 1,
        intermediate_pickle_path: str | None = None,

    ):
        if generate_only and score_only:
            raise ValueError("Both generate_only and score_only are True, this behaviour is ambiguous")

        if (generate_only or score_only) and intermediate_pickle_path is None:
            raise ValueError("One must give an intermediate_pickle_path to do only generation or only scoring")

        if not score_only:
            if self.tokenizer is None or self.model is None:
                raise ValueError(f"Tokenizer ({self.tokenizer}) or model ({self.model}) cannot be None, please initialize or load them before evaluating")

            time0 = time()

            (tokenized_test_dataset, test_dataset), = self.import_process_tokenize_dataset(
                dataset_from_hub = dataset_from_hub,
                dataset_path = dataset_path,
                splits = ['test'],
                task = task,
                num_datapoints = num_datapoints,
                input_format = self.inference_input_format,
                max_token_length = max_token_length,
                return_untok_dataset = True
            )
            time1 = time()
            print(f"{Fore.GREEN}Time to import, process, tokenize: {time1-time0:>6.2f}s{Fore.RESET}")

            self.sanity_check_datasets(
                tokenized_test_dataset,
                max_token_length = max_token_length,
            )
            time2 = time()
            print(f"{Fore.GREEN}Time to sanitiy check: {time2-time1:>6.2f}s{Fore.RESET}")

            inputs_list = []
            predictions = []
            references = []

            self.device = 'cuda' if cuda.is_available() else 'cpu'

            def collate_fn(batch):
                # Create tensors on CPU first, then move once

                if self.is_decoder_only:
                    # Truncate after [mech] token to avoid data leakage during evaluation
                    input_ids = [torch.tensor(ex["input_ids"][:ex["mech_idx"]]) for ex in batch]
                    attention_masks = [torch.tensor(ex["attention_mask"][:ex["mech_idx"]]) for ex in batch]
                else:
                    input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
                    attention_masks = [torch.tensor(ex.get("attention_mask", [1] * len(ex["input_ids"]))) for ex in batch]
                
                # Pad on CPU
                padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side=self.tokenizer.padding_side)
                padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0, padding_side=self.tokenizer.padding_side)
                
                # Single transfer to GPU (DataLoader will handle this if pin_memory=True)
                inputs = {
                    "input_ids": padded_input_ids,
                    "attention_mask": padded_attention_masks
                }
                
                if self.is_decoder_only:
                    decoded_labels = [ex["output_text"].replace(' ', '') for ex in batch]
                else:
                    decoded_labels = [
                        self.tokenizer.decode(ex["labels"], skip_special_tokens=True).replace(' ', '')
                        for ex in batch
                    ]
                
                return inputs, decoded_labels

            loader = DataLoader(
                tokenized_test_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=8,
                prefetch_factor=2
            )

            self.model.to(self.device)
            self.model.eval()

            for inputs, gold_outputs in tqdm(loader, desc="Evaluating (generation)"):
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_token_length,
                    num_beams=num_beams,
                    #num_beam_groups=num_beam_groups,
                    num_return_sequences=num_beams,
                    # diversity_penalty=1.0 if num_beam_groups != 1 else 0,
                    # do_sample=True,
                )

                if self.is_decoder_only:
                    for i in range(input_ids.shape[0]):
                        outputs = generated_ids[i*num_beams:(i+1)*num_beams]
                        texts = [
                            '[mech]'.join(self.tokenizer.decode(out).split("[mech]")[1:]).replace(" ", "") for out in outputs
                        ]

                        texts = [t.split('[eos]')[0] + '[eos]' if 'eos' in t else t for t in texts]
                        predictions.append(texts)
                        references.append(gold_outputs[i] + self.tokenizer.eos_token)
                        inputs_list.append(self.tokenizer.decode(input_ids[i]).replace(" ", "").split('[pad]')[-1]) # Remove the left padding

                else:
                    decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    decoded_preds = self.tokenizer.batch_decode([g_ids[1:] for g_ids in generated_ids]) # 1: to skip the default [pad] in t5

                    for i in range(len(decoded_inputs)):
                        references.append(gold_outputs[i])
                        inputs_list.append(decoded_inputs[i].replace(' ', ''))
                        predictions.append([decoded_preds[i*num_beams+j].replace(' ', '').split('[pad]')[0] for j in range(num_beams)])

        if generate_only:
            # Save in an intermediate pickle
            with open(intermediate_pickle_path, 'wb') as f:
                pickle.dump((references, inputs_list, predictions, test_dataset), f)
            print(f"saved pickle at {intermediate_pickle_path}")

        if score_only:
            # Retrieve these pickled generations
            with open(intermediate_pickle_path, 'rb') as f:
                references, inputs_list, predictions, test_dataset = pickle.load(f)

        if not generate_only:
            def _eval_one_beam(task):
                # Unpack one flattened task
                k, input_text, gold_output, predictions, verbose, stoichio_given = task

                results_beam = []
                for j, text in enumerate(predictions):
                    try:
                        res = evaluation_metrics_w_groundtruth(
                            total_generation=input_text + text,
                            ground_truth_msmi=gold_output,
                            format=self.inference_input_format,
                            authorize_duplicate_species=(not stoichio_given),
                            verbose=verbose
                        )

                        result_row = {
                            "input": input_text,
                            "gold_output": gold_output,
                            "generated_text": text,
                            "beam_index": j,
                            "msmi_reac_legal": res["msmi_reac_legal"],
                            "msmi_reac_correct": res["msmi_reac_correct"],
                            "every_index_unique": res["every_index_unique"],
                            "at_least_two_mapped_atoms_used": res["at_least_two_mapped_atoms_used"],
                            "all_indices_used_were_mapped": res["all_indices_used_were_mapped"],
                            "product_stable": res["product_stable"],
                            "product_reached": res["product_reached"],
                            "ground_truth_prod_sim": res["ground_truth_prod_sim"],
                            "ground_truth_equiv": res["ground_truth_equiv"],
                            "ground_truth_exact": res["ground_truth_exact"],
                        }
                        results_beam.append(result_row)
                    except NotImplementedError as e:
                        print(f"{e}")
                        result_row = {
                            "input": input_text,
                            "gold_output": gold_output,
                            "generated_text": text,
                            "beam_index": j,
                            "msmi_reac_legal": 0,
                            "msmi_reac_correct": 0,
                            "every_index_unique": 0,
                            "at_least_two_mapped_atoms_used": 0,
                            "all_indices_used_were_mapped": 0,
                            "product_stable": 0,
                            "product_reached": 0,
                            "ground_truth_prod_sim": 0,
                            "ground_truth_equiv": 0,
                            "ground_truth_exact": 0,
                        }
                        results_beam.append(result_row)
                    except Exception as e:
                        print(f"{Fore.RED}Uncaught exception: {e}{Fore.RESET}")
                        result_row = {
                            "input": input_text,
                            "gold_output": gold_output,
                            "generated_text": text,
                            "beam_index": j,
                            "msmi_reac_legal": 0,
                            "msmi_reac_correct": 0,
                            "every_index_unique": 0,
                            "at_least_two_mapped_atoms_used": 0,
                            "all_indices_used_were_mapped": 0,
                            "product_stable": 0,
                            "product_reached": 0,
                            "ground_truth_prod_sim": 0,
                            "ground_truth_equiv": 0,
                            "ground_truth_exact": 0,
                        }
                        results_beam.append(result_row)

                # Return with k to reorder
                return k, results_beam

            filename = Path(parquet_file_path).stem
            folder_lock = Path("/scratch/neukomm/amlite/locks")  # Folder for lock files
            folder_results = Path("/scratch/neukomm/amlite/results")  # Folder for batch results
            os.makedirs(folder_lock, exist_ok=True)
            os.makedirs(folder_results, exist_ok=True)

            # Create all tasks
            tasks = []
            k = 0
            stoichio_given = task.split('_')[0]!='spe'
            print(f"{stoichio_given = }")
            for i, (input_text, gold_output) in enumerate(zip(inputs_list, references)):
                tasks.append((k, input_text, gold_output, predictions[i], (parquet_file_path is None), stoichio_given))
                k += 1

            total = len(tasks)
            size_batch = (total-1) // n_batch_score + 1

            max_workers = ((cpus - 2) if (cpus := os.cpu_count()) is not None else 1) # We let a few cpus resting on purpose (better perf. on my laptop)

            results_buffer = []   # (k, row)
            results = []

            
            for n_b in range(n_batch_score):
                lock_file = folder_lock / f"{filename}_batch_{n_b}.lock"
                result_file = folder_results / f"{filename}_batch_{n_b}.pkl"

                if os.path.exists(lock_file) or os.path.exists(result_file):
                    print(f"Lock file or result file for batch {n_b} exists, skipping...")
                    continue

                try:
                    # Open with 'x' mode - exclusive creation, fails if file exists
                    with open(lock_file, 'x') as f:
                        pass  # Just create the file
                except FileExistsError:
                    print(f"Batch {n_b} was just claimed by another worker, skipping...")
                    continue

                print(f"Processing batch {n_b}...")

                start_idx = size_batch * n_b
                end_idx = min(size_batch * (n_b + 1), total)
                batch_tasks = tasks[start_idx:end_idx]

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    #futures = [ex.submit(_eval_one, t) for t in tasks]
                    futures = [ex.submit(_eval_one_beam, t) for t in batch_tasks]
                    for fut in tqdm(as_completed(futures), total=len(batch_tasks), desc="Evaluating (scoring)"):
                        try:
                            results_buffer.append(fut.result())
                        except Exception as e:
                            print(f"Uncatched error {e}")
                            exit()

                results_buffer.sort(key=lambda x: x[0])  # by k

                # Extract only the row dicts
                results = [row for _, beam_row in results_buffer for row in beam_row]

                with open(result_file, 'wb') as f:
                    pickle.dump(results_buffer, f)
                    results_buffer = []

            if parquet_file_path is None:
                for i, (input_text, gold_output) in tqdm(enumerate(zip(inputs_list, references))):
                    print(f"{Fore.LIGHTBLACK_EX}Input: {input_text}{Fore.RESET}")
                    print(f"{Fore.LIGHTBLACK_EX} Gold: {Fore.LIGHTGREEN_EX}{gold_output}{Fore.RESET}")
                    for i, text in enumerate(predictions[i]):
                        print(f"{Fore.LIGHTBLACK_EX}{i+1:>5.0f}: {Fore.LIGHTYELLOW_EX}{text}{Fore.RESET}")
                    
                    all_gens = "".join([f"{n_beam+1:>4.0f} " for n_beam in range(num_beams)])
                    print(f"                                {all_gens}")
                    print("----- Score checks -----")
                    for metric in ["msmi_reac_legal", "msmi_reac_correct", "every_index_unique", "at_least_two_mapped_atoms_used", "all_indices_used_were_mapped", "product_stable", "product_reached"]:
                        all_scores = "".join([f"[{answers_dict[results[-num_beams + i][metric]]}]" for i in range(num_beams)])
                        print(f"{metric.replace('_', ' ').capitalize():<32} {all_scores}")

                    print("----- Match GT ------")
                    for metric in ["ground_truth_prod_sim", "ground_truth_equiv", "ground_truth_exact"]:
                        all_scores = "".join([f"[{answers_dict[results[-num_beams + i][metric]]}]" for i in range(num_beams)])
                        print(f"{Style.BRIGHT if metric == 'ground_truth_equiv' else Style.RESET_ALL}{metric.replace('_', ' ').capitalize():<32} {all_scores}{Style.RESET_ALL}")

            # Save to parquet if requested
            if parquet_file_path:
                all_results = []
                for n_b in range(n_batch_score):
                    result_file = folder_results / f"{filename}_batch_{n_b}.pkl"
                    with open(result_file, 'rb') as f:
                        batch_results = pickle.load(f)
                        all_results.extend(batch_results)

                df = pd.DataFrame(all_results)
                df.to_parquet(parquet_file_path, index=False)
                test_dataset.to_parquet(parquet_file_path.rsplit('.', 1)[0]+'_questions.parquet')
                print(f"\n✅ Saved evaluation results to {parquet_file_path}")

            return None

    def tokenize_and_filter_encoder_decoder(self, example):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None, please initialize it before running tokenize function")

        encodings = self.tokenizer(
            example['input'],
            text_target=example['output'] + self.tokenizer.eos_token,
            max_length=self.max_token_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        mech_token_idx = self.tokenizer.encode('[mech]')[0]
        eos_token_idx = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        is_valid = (
            encodings["input_ids"][-1] == mech_token_idx
            and encodings["labels"][-1] == eos_token_idx
        )
        return {
            "input_ids": encodings["input_ids"] if is_valid else [],
            "attention_mask": encodings["attention_mask"] if is_valid else [],
            "labels": encodings["labels"] if is_valid else [],
            "valid": is_valid,
        }

    def tokenize_and_filter_decoder(self, example):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None, please initialize it before running tokenize function")
        
        input_text = example['input'] + example['output'] + self.tokenizer.eos_token

        encoded = self.tokenizer(
            input_text,
            padding=False, # Let data collator handle batching
            return_attention_mask=True,
            add_special_tokens=False,
        )

        # Filter out sequences that are too long
        is_valid = len(encoded["input_ids"]) <= self.max_token_length

        # Create truncated versions after [mech] token for evaluation (to avoid data leakage)
        mech_token_id = self.tokenizer.encode('[mech]', add_special_tokens=False)[0]
        mech_idx = encoded["input_ids"].index(mech_token_id) + 1  # Include [mech] token

        return {
            "input_ids": encoded["input_ids"] if is_valid else [],
            "attention_mask": encoded["attention_mask"] if is_valid else [],
            "mech_idx": mech_idx,
            "output_text": example["output"] if is_valid else "",
            "valid": is_valid,
        }

    def import_process_tokenize_dataset(
        self,
        dataset_path,
        dataset_from_hub,
        splits,
        task,
        input_format,
        max_token_length,
        num_datapoints: int = -1,
        return_untok_dataset: bool = False,
    ):

        dict_dataset_raw = dict()
        dict_dataset = dict()
        dict_tokenized_dataset = dict()


        for split in splits:
            time00 = time()
            if dataset_from_hub:
                if ';' in dataset_path:
                    dict_dataset_raw[split] = concatenate_datasets([
                        load_dataset(ds, split=split, token=os.getenv("HF_TOKEN")) for ds in dataset_path.split(';')
                    ])
                else:
                    dict_dataset_raw[split] = load_dataset(dataset_path, split=split, token=os.getenv("HF_TOKEN"))
            else:
                dataset: DatasetDict = load_from_disk(dataset_path)
                dict_dataset_raw[split] = dataset[split]

            time01 = time()

            if num_datapoints > 0 and len(dict_dataset_raw[split]) > num_datapoints:
                np.random.seed(22)
                rxn_to_indices = defaultdict(list)
                for i, rxn_idx in enumerate(dict_dataset_raw[split]['rxn_idx']):
                    rxn_to_indices[rxn_idx].append(i)
                
                # Shuffle the groups
                unique_groups = list(rxn_to_indices.keys())
                np.random.shuffle(unique_groups)
                
                # Collect indices
                indices = []
                for group in unique_groups:
                    group_indices = rxn_to_indices[group]
                    indices.extend(group_indices)
                    
                    # Stop if we've reached the target
                    if len(indices) >= num_datapoints:
                        break

                dict_dataset_raw[split] = dict_dataset_raw[split].select(indices)

            time02 = time()

            dict_dataset[split] = self.process_dataset_for_nlp_parallel(
                dict_dataset_raw[split],
                task,
                input_format,
                with_metadata = return_untok_dataset,
            )
            time03 = time()

            self.max_token_length = max_token_length
            dict_tokenized_dataset[split] = dict_dataset[split].map(
                self.tokenize_and_filter_decoder if self.is_decoder_only else self.tokenize_and_filter_encoder_decoder,
                remove_columns=dict_dataset[split].column_names,
                num_proc=16,
            ).filter(
                lambda x: x["valid"],  # Filter out invalid sequences
            ).remove_columns(["valid"])

            time04 = time()

            print(f"{Fore.YELLOW}Import time            : {time01-time00:>6.2f}s{Fore.RESET}")
            print(f"{Fore.YELLOW}Shuffle time           : {time02-time01:>6.2f}s{Fore.RESET}")
            print(f"{Fore.YELLOW}Process time (parallel): {time03-time02:>6.2f}s{Fore.RESET}")
            print(f"{Fore.YELLOW}Filter time            : {time04-time03:>6.2f}s{Fore.RESET}")

        if return_untok_dataset:
            return ((dict_tokenized_dataset[split], dict_dataset[split]) for split in splits)
        else:
            return (dict_tokenized_dataset[split] for split in splits)

    def sanity_check_datasets(
        self,
        *tokenized_datasets,
        max_token_length: int | None = None,
    ):

        unk_token_idx = self.tokenizer.encode(self.tokenizer.unk_token)[0]

        for tokenized_dataset in tokenized_datasets:
            for dp in tokenized_dataset:
                input_ids = dp["input_ids"]
                
                # Check for empty input_ids
                if len(input_ids) == 0:
                    raise ValueError("Some datapoints have empty input_ids.")
                
                # Check max token length
                if max_token_length is not None and len(input_ids) > max_token_length:
                    raise ValueError("Some datapoints exceed max_token_length.")
                
                # Check for unknown tokens
                if unk_token_idx in input_ids:
                    raise ValueError("Unknown tokens found in dataset.")

    def search_mech(
        self,
        rxn: str,
        max_node_budget:int = 50,
        search_algo: str = "best_first",
        k_n_tuple: tuple[int, int] = (5, 10), # (Max moves kept, beam width)
        search_temperature: float = 1.0,
        cost_of_move: float = 0, # Punish longer mechanisms
        max_depth: int = 10,
        conditions_are_reactants: bool = True,
        verbose: bool = False,
        stop_at_best: bool = True, # If False, the algorithm will only stop at max_budget and might return more paths.
    ):
        reac, cond, prod = rxn.split('>')
        if conditions_are_reactants and cond != '':
            reac = '.'.join(reac.split('.')+cond.split('.'))

        return search(
            reac,
            prod,
            partial(self.human_to_input, format=self.inference_input_format, already_canonicalized=False),
            max_node_budget = max_node_budget,
            search_algo = search_algo,
            k_n_tuple = k_n_tuple,
            decoder_only = self.is_decoder_only,
            tokenizer = self.tokenizer,
            model = self.model,
            search_temperature = search_temperature,
            cost_of_move = cost_of_move,
            max_depth = max_depth,
            verbose = verbose,
            stop_at_best = stop_at_best, # If False, the algorithm will only stop at max_budget and might return more paths.
        )


