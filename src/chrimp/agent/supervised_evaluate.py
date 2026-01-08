import os
from tqdm import tqdm
import pandas as pd
import re
import math
from colorama import Fore, Style
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from torch.utils.data import DataLoader

from chrimp.world.mechsmiles import MechSmiles
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.AllChem import GetRDKitFPGenerator
from rdkit.DataStructs import TanimotoSimilarity

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

fpgen = GetRDKitFPGenerator()
answers_dict = {
    None: f"{Fore.LIGHTBLACK_EX} ❔{Fore.RESET}",
    1: f"{Fore.GREEN} ✔️ {Fore.RESET}",
    0: f"{Fore.RED} ❌{Fore.RESET}",
    0.1: f"{Fore.LIGHTBLACK_EX}0.1{Fore.RESET}",
    0.2: f"{Fore.LIGHTBLACK_EX}0.2{Fore.RESET}",
    0.3: f"{Fore.LIGHTBLACK_EX}0.3{Fore.RESET}",
    0.4: f"{Fore.LIGHTBLACK_EX}0.4{Fore.RESET}",
    0.5: f"{Fore.LIGHTBLACK_EX}0.5{Fore.RESET}",
    0.6: f"{Fore.LIGHTBLACK_EX}0.6{Fore.RESET}",
    0.7: f"{Fore.LIGHTBLACK_EX}0.7{Fore.RESET}",
    0.8: f"{Fore.LIGHTBLACK_EX}0.8{Fore.RESET}",
    0.9: f"{Fore.LIGHTBLACK_EX}0.9{Fore.RESET}",
}

def canonicalize_unmap(smiles, verbose=False):
    mol = MolFromSmiles(smiles)

    if mol is None:
        if verbose:
            print(f"SMILES {smiles} gave me a None")
        return smiles
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return MolToSmiles(mol)

def canonicalize_unmap_list(
    smiles: str,
    sort: bool = True,
    verbose: bool = False,
):
    """
    Individually canonicalizes a SMILES string. Each species in the SMILES becomes an element in the list
    """
    smiles_list = smiles.split('.')

    canonicalized_list = [canonicalize_unmap(smile, verbose=verbose) for smile in smiles_list]
    if sort:
        canonicalized_list.sort()
    return canonicalized_list



@lru_cache(maxsize=1024)
def evaluation_metrics_similarity(
    smiles,
    ref_smiles,
    authorize_not_all_species=False,
    authorize_duplicate_species=False,
    dampening_coeff=0.9,
    step_one_over_x=10,
    verbose=False,
):
    """
    Returns a score between 0 and 1, where 1 means the two SMILES are identical.
    Careful, in the case of non-identical SMILES, the score is dampened by the `dampening_coeff`,
    to avoid false positives due to the RDKit's fingerprinting not taking H's into account.
    For example, "CC(=[OH+])OC=O" and "CC(=O)OC=[OH+]" have a similarity of 1 w/o being similar.

    Args:
        smiles (str): The SMILES string to check.
        ref_smiles (str): The reference SMILES string.
        dampening_coeff (float): A coefficient to dampen the similarity score when the SMILES are not identical.
        step_one_over_x (int): A step for the similarity score, to avoid false positives.
        verbose (bool): Whether to print verbose messages.
    """
    global fpgen

    assert math.floor(dampening_coeff/step_one_over_x)*step_one_over_x < 1, f"Your dampening_coeff ({dampening_coeff}) is not low enough or your step_one_over_x ({step_one_over_x}) not high enough"

    if smiles == ref_smiles: # Always true and might gain some time as string comparison take a negligible time
        return 1

    can_list_smiles = canonicalize_unmap_list(smiles, sort=False, verbose=verbose)
    can_list_ref_smiles = canonicalize_unmap_list(ref_smiles, sort=False, verbose=verbose)

    try:
        if authorize_duplicate_species: # Set is the appropriate object here (We don't care about stoichiometry)
            can_set_smiles = set(can_list_smiles)
            can_set_ref_smiles = set(can_list_ref_smiles)
            
            if authorize_not_all_species: # We check the subset
                if can_set_smiles.issubset(can_set_ref_smiles):
                    return 1
            
            else: # We check the equality
                if can_set_smiles == can_set_ref_smiles:
                    return 1

        else: # Counter is the appropriate object here
            can_counter_smiles = Counter(can_list_smiles)
            can_counter_ref_smiles = Counter(can_list_ref_smiles)

            if authorize_not_all_species: # We check the subset
                if can_counter_smiles.keys() <= can_counter_ref_smiles.keys():
                    if all([smiles_count <= can_counter_ref_smiles[smiles] for smiles, smiles_count in can_counter_smiles.items()]):
                        return 1

            else: # We check the equality
                if can_counter_smiles.keys() == can_counter_ref_smiles.keys():
                    if all([smiles_count == can_counter_ref_smiles[smiles] for smiles, smiles_count in can_counter_smiles.items()]):
                        return 1

        # If we reach this point, the SMILES are not identical, we compute the Tanimoto similarity
        return math.floor(TanimotoSimilarity(
            fpgen.GetFingerprint(MolFromSmiles('.'.join(can_list_smiles))),
            fpgen.GetFingerprint(MolFromSmiles('.'.join(can_list_ref_smiles))),
        )*dampening_coeff*step_one_over_x)/step_one_over_x
    except Exception as e:
        if verbose:
            print(f"In molecule set legal assessment: error {e}")
        return 0

def evaluation_metrics_reac(
    msmi_string,
    reac_string,
    authorize_sub_reac = True, # We can chose to not use all reactants
    authorize_duplicate_species = False, # We can use multiple times species in reac (more times than they appear)
    verbose=False,
):
    msmi_reac, msmi_arrows = msmi_string.split('|', 1) if ("|" in msmi_string) else (msmi_string, "")

    try:
        msmi_reac_ = MolToSmiles(MolFromSmiles(msmi_reac))
        msmi_reac_legal = 1
        msmi_reac_correct = evaluation_metrics_similarity(
            smiles=msmi_reac_,
            ref_smiles=reac_string,
            authorize_not_all_species=authorize_sub_reac,
            authorize_duplicate_species=authorize_duplicate_species,
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            print(f"In molecule set legal assessment: error {e}")
        msmi_reac_legal = 0
        msmi_reac_correct = 0

    return msmi_reac_legal, msmi_reac_correct


def evaluation_metrics_indices(
    msmi_string: str,
):
    if not '|' in msmi_string:
        return (0, 0, 0)

    pattern_mapped = r"(?<=:)\d+(?=\])"
    all_maps_list = re.findall(pattern_mapped, msmi_string)
    all_maps = set(all_maps_list)
    every_index_unique = float(len(all_maps_list) == len(all_maps))
    search_result = re.search(r'\|(.*)', msmi_string)
    all_maps_used = set(re.findall(r'\d+', search_result.group(1))) if search_result else set()
    at_least_two_mapped_atoms_used = float(len(all_maps_used) >= 2)

    try:
        if all_maps_used.issubset(all_maps):
            all_indices_used_were_mapped = 1
        else:
            all_indices_used_were_mapped = 0
    except Exception as e:
        all_indices_used_were_mapped = 0


    return at_least_two_mapped_atoms_used, all_indices_used_were_mapped, every_index_unique

def evaluation_metrics_products(
    msmi_string: str,
    prod_string: str| None, # None in the case of wo groundtruth with 'reac' format
    authorize_sub_prod: bool,
    verbose=False,
):
    msmi = MechSmiles(msmi_string)

    try:
        _ = MolToSmiles(MolFromSmiles(msmi.prod))
        product_stable = 1

    except Exception as e:
        if verbose:
            print(f"In product stability assessement: error {e}")
        product_stable = 0

    if product_stable and prod_string is not None:
        product_reached = evaluation_metrics_similarity(
            prod_string,
            msmi.prod,                      # Remark that here the reference is msmi.prod, so the logic is reversed
            authorize_not_all_species=authorize_sub_prod, # We want the products to be a subset of what the MechSMILES reaches (True) or an exact match (False)
            authorize_duplicate_species=False, # However we want all individual products to be predicted
            verbose=verbose,
        )
    else:
        product_reached = 0

    return product_stable, product_reached


def evaluation_metrics_wo_groundtruth(
    total_generation,
    authorize_sub_reac = True, # We can chose to not use all reactants
    authorize_duplicate_species = False, # We can use multiple times species in reac (more times than they appear)
    authorize_sub_prod = True,
    format = "retro",
    verbose=False,
):
    """
    Few metrics that can be heuristically tested without knowing the ground truth

    Hierarchy of what to check:
    - msmi_reac_legal
    - msmi_reac_correct
    - every_index_unique
    - at_least_two_mapped_atoms_used
    - all_indices_used_were_mapped
    - mechsmiles_legal
    - product_stable
    - product_reached
    """

    for mandatory in ['[reac]', '[mech]', '[eos]'] + (['prod'] if format != "reac" else []):
        if not (mandatory in total_generation):
            if verbose:
                print(f"Incorrect format of the generation {mandatory} not in {total_generation}")
            #return defaultdict(int) # Everthing is false
            return {
                "evaluated_msmi": 0,
                "msmi_reac_legal": 0,
                "msmi_reac_correct": 0,
                "at_least_two_mapped_atoms_used": 0,
                "all_indices_used_were_mapped": 0,
                "every_index_unique": 0,
                "product_stable": 0,
                "product_reached": 0,
            }

    if format == "retro":
        pattern_gen = r"\[prod](.*?)\[reac](.*?)\[mech](.*)\[eos]"
        match = re.match(pattern_gen, total_generation)
        prod_string, reac_string, msmi_string = match.group(1), match.group(2), match.group(3)

    elif format == "forward":
        pattern_gen = r"\[reac](.*?)\[prod](.*?)\[mech](.*)\[eos]"
        match = re.match(pattern_gen, total_generation)
        reac_string, prod_string, msmi_string = match.group(1), match.group(2), match.group(3)

    elif format == "reac":
        pattern_gen = r"\[reac](.*?)\[mech](.*)\[eos]"
        match = re.match(pattern_gen, total_generation)
        reac_string, msmi_string = match.group(1), match.group(2)
        prod_string = None

    else:
        raise ValueError('Direction must be in ["retro", "forward", "reac"]')

    msmi_reac_legal = 0
    msmi_reac_correct = 0
    at_least_two_mapped_atoms_used = 0
    all_indices_used_were_mapped = 0
    every_index_unique = 0
    product_stable = 0 
    product_reached = 0

    msmi_reac_legal, msmi_reac_correct = evaluation_metrics_reac(
        msmi_string,
        reac_string,
        authorize_sub_reac=authorize_sub_reac,
        authorize_duplicate_species=authorize_duplicate_species,
        verbose=verbose,
    )

    if msmi_reac_legal:
        (
            at_least_two_mapped_atoms_used,
            all_indices_used_were_mapped,
            every_index_unique,
        ) = evaluation_metrics_indices(
            msmi_string
        )

        mechsmiles_legal = (msmi_reac_legal and all_indices_used_were_mapped and every_index_unique)
        if mechsmiles_legal:
            product_stable, product_reached = evaluation_metrics_products(
                msmi_string,
                prod_string,
                authorize_sub_prod=authorize_sub_prod,
                verbose=verbose,
            )

    return {
        "evaluated_msmi": msmi_string,
        "msmi_reac_legal": msmi_reac_legal,
        "msmi_reac_correct": msmi_reac_correct,
        "at_least_two_mapped_atoms_used": at_least_two_mapped_atoms_used,
        "all_indices_used_were_mapped": all_indices_used_were_mapped,
        "every_index_unique": every_index_unique,
        "product_stable": product_stable,
        "product_reached": product_reached,
    }

def evaluation_metrics_w_groundtruth(
    total_generation: str,
    ground_truth_msmi: str,
    authorize_sub_reac: bool = True, # We can chose to not use all reactants
    authorize_duplicate_species: bool = False, # We can use multiple times species in reac (more times than they appear)
    authorize_sub_prod: bool = True,
    format:str = "retro",
    verbose:bool = False,
):

    metrics = evaluation_metrics_wo_groundtruth(
        total_generation,
        authorize_sub_reac = authorize_sub_reac,
        authorize_duplicate_species = authorize_duplicate_species,
        authorize_sub_prod=authorize_sub_prod,
        format=format,
        verbose = verbose,
    )

    ground_truth_msmi = ground_truth_msmi.replace("[eos]", "")

    ground_truth_prod_sim = 0
    ground_truth_equiv = 0
    ground_truth_exact = 0

    if metrics["msmi_reac_correct"] and metrics["product_stable"]:
        
        msmi_string = metrics["evaluated_msmi"]

        msmi = MechSmiles(msmi_string)
        gt_msmi = MechSmiles(ground_truth_msmi)

        prod_ground_truth_prod_similarity = evaluation_metrics_similarity(
            msmi.prod,
            gt_msmi.prod,
            authorize_not_all_species=False,
            authorize_duplicate_species=False,
            verbose=verbose,
        )

        ground_truth_prod_sim = prod_ground_truth_prod_similarity
        ground_truth_equiv = float(metrics["msmi_reac_correct"] == 1 and prod_ground_truth_prod_similarity == 1)
        ground_truth_exact = float(msmi_string == ground_truth_msmi)
    
    metrics["ground_truth_prod_sim"] = ground_truth_prod_sim
    metrics["ground_truth_equiv"] = ground_truth_equiv
    metrics["ground_truth_exact"] = ground_truth_exact

    return metrics

def evaluate_task_hf(
    dataset,
    task_col,
    model_path,
    tokenizer_path,
    num_beams=20,
    num_lines_eval=-1,
    batch_size=8,
    parquet_file_path=None,
    decoder_only=True,
    evaluate_on_train_set=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer.padding_side = "left"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer.padding_side = "right"
    model.eval()
    model.to(device)

    if evaluate_on_train_set:
        print(f"{Fore.RED}Evaluating on train set for debug, don't forget to change that back{Fore.RESET}")
        val_dataset = load_dataset(dataset, split='train')
    else:
        val_dataset = load_dataset(dataset, split='val')


    # Needs modifs to store as a parquet, but good to visualize:
    if num_lines_eval > 0:
        val_dataset = val_dataset.select(range(min(num_lines_eval, len(val_dataset))))

    inputs_list = []
    predictions = []
    references = []

    def collate_fn(batch):
        inputs = tokenizer(
            [ex[task_col] for ex in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            #max_length=tokenizer.model_max_length, #OverflowError: int too big to convert
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs, [ex["mech_smi"].replace(' ', '') for ex in batch]

    loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    for inputs, gold_outputs in tqdm(loader, desc="Evaluating (generation)"):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            num_beams=num_beams,
            #num_beam_groups=num_beam_groups,
            num_return_sequences=num_beams,
            # diversity_penalty=1.0 if num_beam_groups != 1 else 0,
            # do_sample=True,
        )

        if decoder_only:

            for i in range(input_ids.shape[0]):
                outputs = generated_ids[i*num_beams:(i+1)*num_beams]
                texts = [
                    '[mech]'.join(tokenizer.decode(out).split("[mech]")[1:]).replace(" ", "") for out in outputs
                ]

                texts = [t.split('[eos]')[0] + '[eos]' if 'eos' in t else t for t in texts]
                predictions.append(texts)
                references.append(gold_outputs[i] + tokenizer.eos_token)
                inputs_list.append(tokenizer.decode(input_ids[i]).replace(" ", "").split('[pad]')[-1]) # Remove the left padding

        else:
            decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode([g_ids[1:] for g_ids in generated_ids]) # 1: to skip the default [pad] in t5

            for i in range(len(decoded_inputs)):
                references.append(gold_outputs[i])
                inputs_list.append(decoded_inputs[i].replace(' ', ''))
                predictions.append([decoded_preds[i*num_beams+j].replace(' ', '').split('[pad]')[0] for j in range(num_beams)])

    
    def _eval_one_beam(task):
        # Unpack one flattened task
        k, input_text, gold_output, predictions, verbose = task

        results_beam = []
        for j, text in enumerate(predictions):
            try:
                res = evaluation_metrics_w_groundtruth(
                    total_generation=input_text + text,
                    ground_truth_msmi=gold_output,
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

        # Return with k to reorder
        return k, results_beam


    # Create all tasks
    tasks = []
    k = 0
    for i, (input_text, gold_output) in enumerate(zip(inputs_list, references)):
        tasks.append((k, input_text, gold_output, predictions[i], (parquet_file_path is None)))
        k += 1

    total = len(tasks)
    max_workers = ((cpus - 2) if (cpus := os.cpu_count()) is not None else 1) # We let a few cpus resting on purpose (better perf. on my laptop)

    results_buffer = []   # (k, row)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        #futures = [ex.submit(_eval_one, t) for t in tasks]
        futures = [ex.submit(_eval_one_beam, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=total, desc="Evaluating (scoring)"):
            try:
                results_buffer.append(fut.result())
            except Exception as e:
                print(f"Uncatched error {e}")
                exit()

    results_buffer.sort(key=lambda x: x[0])  # by k

    # Extract only the row dicts
    results = [row for _, beam_row in results_buffer for row in beam_row]

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
        df = pd.DataFrame(results)
        df.to_parquet(parquet_file_path, index=False)
        print(f"\n✅ Saved evaluation results to {parquet_file_path}")

    return None

def free_evaluation(
    reac_smiles,
    prod_smiles,
    model_path,
    num_beams,
    canonicalize_smiles=True,
    evaluation_metrics=False,
    verbose_errors=False,
    ground_truth=None,
):

    raise NotImplementedError("Needs rewriting with new evaluation metrics")

    if canonicalize_smiles:
        reac_smiles = MolToSmiles(MolFromSmiles(reac_smiles,sanitize=False))
        prod_smiles = MolToSmiles(MolFromSmiles(prod_smiles,sanitize=False))

    model = init_model(checkpoint=model_path)
    model.model.eval()

    input = f"[prod]{prod_smiles}[reac]{reac_smiles}[mech]"
    _, texts_gen = free_generation(
        model=model,
        input=input,
        max_new_tokens=512,
        num_beams=num_beams,
        verbose=(not evaluation_metrics),
    )
    if evaluation_metrics:
        eval_metrics = {
            "msmi_reac_legal": [],
            "msmi_reac_correct": [],
            "at_least_two_mapped_atoms_used": [],
            "all_indices_used_were_mapped": [],
            "product_stable": [],
            "product_reached": [],
            "ground_truth_similar": [],
        }
        for i, text in enumerate(texts_gen):
            (
                msmi_reac_legal,
                msmi_reac_correct,
                at_least_two_mapped_atoms_used,
                all_indices_used_were_mapped,
                product_stable,
                product_reached,
            ) = evaluation_metrics_wo_groundtruth(input+texts_gen[i], verbose=verbose_errors)

            eval_metrics["msmi_reac_legal"].append(msmi_reac_legal)
            eval_metrics["msmi_reac_correct"].append(msmi_reac_correct)
            eval_metrics["at_least_two_mapped_atoms_used"].append(at_least_two_mapped_atoms_used)
            eval_metrics["all_indices_used_were_mapped"].append(all_indices_used_were_mapped)
            eval_metrics["product_stable"].append(product_stable)
            eval_metrics["product_reached"].append(product_reached)

            if ground_truth is not None:
                eval_metrics["ground_truth_similar"].append(
                    texts_gen[i] == ground_truth
                )

        print(f"{Fore.LIGHTBLACK_EX}input: {input}{Fore.RESET}")
        for i, text in enumerate(texts_gen):
            print(f"{Fore.LIGHTBLACK_EX}{i+1:>5.0f}: {Fore.LIGHTYELLOW_EX}{text}{Fore.RESET}")
            #print(f"{Fore.LIGHTBLACK_EX}{i+1:>3.0f}: {input}{Fore.LIGHTYELLOW_EX}{text}{Fore.RESET}")

        all_gens = "".join([f"{i+1:>4.0f} " for i in range(num_beams)])
        all_msmi_reac_legal = "".join([f"[{answers_dict[eval_metrics['msmi_reac_legal'][i]]}]" for i in range(num_beams)])
        all_msmi_reac_correct = "".join([f"[{answers_dict[eval_metrics['msmi_reac_correct'][i]]}]" for i in range(num_beams)])
        all_at_least_two_mapped_atoms_used = "".join([f"[{answers_dict[eval_metrics['at_least_two_mapped_atoms_used'][i]]}]" for i in range(num_beams)])
        all_all_indices_used_were_mapped = "".join([f"[{answers_dict[eval_metrics['all_indices_used_were_mapped'][i]]}]" for i in range(num_beams)])
        all_prod_stable = "".join([f"[{answers_dict[eval_metrics['product_stable'][i]]}]" for i in range(num_beams)])
        all_prod_reached = "".join([f"[{answers_dict[eval_metrics['product_reached'][i]]}]" for i in range(num_beams)])


        print("----- Score checks -----")
        print(f"                              {all_gens}")
        print(f"MechSMILES reac legal          {all_msmi_reac_legal}")
        print(f"MechSMILES reac correct        {all_msmi_reac_correct}")
        print(f"At least two mapped atoms used {all_at_least_two_mapped_atoms_used}")
        print(f"All indices used mapped        {all_all_indices_used_were_mapped}")
        print(f"MechSMILES legal               {all_prod_stable}")
        print(f"Product reached                {all_prod_reached}")

        if ground_truth is not None:
            print("----- Match GT ------")
            print(f"Text is the exact same         {all_ground_truth_similar}")

if __name__ == "__main__":
    # Try it interactively
    # Free generation (Change the input to test, be careful that the input respects the expected format from the tokenizer)
    

    #interesting_list = [
    #    ("O=C(CC(OC)OC)C.NNC", "OC(CC(OC)OC)(NNC)C"),
    #    ("CC(=O)C.N", "CC(O)(N)C"),
    #    ("CC([O-])([NH3+])C", "CC(O)(N)C"),
    #    ("C1CCCCC1=O.N", "C1CCCCC1(O)N"),
    #    (r"C(C)#CCCC=C(C)CCC=C.O1CCOC1=O", "CC([O+]=C1OCCO1)=C1CCC2C[CH-]CCC12C"), # Zlatko's demonic
    #    ("O=C(NCC1CCCN1)NC1=CC(C(F)(F)F)=CC(C(F)(F)F)=C1.O=C1C=C(Cl)C(=O)C=C1Cl", "O=C(NCC1CCC[NH+]1C1C(=O)C(Cl)=CC([O-])=C1Cl)NC1=CC(C(F)(F)F)=CC(C(F)(F)F)=C1"),
    #    ("C=C.C=C(C)C=C", "C1C=C(C)CCC1"),
    #    ("CCC(=O)N[C+]1C(Br)=CC(Br)=CC1Br.[Br-]","Br.CCC(=O)NC1=C(Br)C=C(Br)C=C1Br"),
    #    ("BrBr.CCC(=O)NC1=CC=C(Br)C=C1Br","Br.CCC(=O)NC1=C(Br)C=C(Br)C=C1Br"),
    #    ("BrBr.CCC(=O)NC1=C(Br)C=C(Br)C=C1","Br.CCC(=O)NC1=C(Br)C=C(Br)C=C1Br"),
    #    ("CCC(=O)N[C+]1C(Br)=CC(Br)=CC1Br.[Br-]","Br.CCC(=O)NC1=C(Br)C=C(Br)C=C1Br"),
    #    ("Oc1cc(Cl)c(cc1Cl)S(=O)(=O)O[Na].O","Oc1cc(Cl)ccc1Cl.OS(=O)(=O)O[Na]")
    #]

    #for reac_smiles, prod_smiles in interesting_list:

    #    free_evaluation(
    #        reac_smiles,
    #        prod_smiles,
    #        #os.path.join(os.path.dirname(__file__), "checkpoints", "last_model.pth"),
    #        os.path.join(os.path.dirname(__file__), "checkpoints", "model_epoch20.pth"),
    #        16,
    #        evaluation_metrics=True,
    #        #verbose_errors=True,
    #        #ground_truth=ground_truth,
    #    )

    #task = "ori_ori_1"
    task = "equ_equ_n"
    #task = "equ_ori_n"

    #model_type = "llama"
    #run = 48
    #checkpoint = 61584

    #run = 63
    #checkpoint = 126022

    #run = 65
    #checkpoint = 388791

    run = 71
    #checkpoint = 72240
    #checkpoint = 103200
    checkpoint = 206400
    model_type = "t5"

    run = 72
    checkpoint = 412800

    run = 73
    #checkpoint = 1330076
    #checkpoint = 1813740
    checkpoint = 2418320

    run = 82
    checkpoint = 125640
    checkpoint = 157050
    checkpoint = 219870
    checkpoint = 345510

    run = 83
    checkpoint = 133280
    checkpoint = 346528
    checkpoint = 399840
    checkpoint = 453152

    run = 84
    checkpoint = 229704
    checkpoint = 267988
    checkpoint = 306272
    checkpoint = 344556
    checkpoint = 382840
    checkpoint = 421124
    checkpoint = 650828

    run = 85
    checkpoint = 459672
    checkpoint = 536284
    checkpoint = 574590

    run = 86
    checkpoint = 650624

    run = 87
    checkpoint = 381100
    checkpoint = 533540

    #dataset_name = "pmechdb"
    #num_beams = 20

    dataset_name = "mechuspto31k"
    num_beams=5

    if dataset_name == 'pmechdb':
        full_dataset_name =  "SchwallerGroup/pmechdb_manually_curated_multi_elem_retro"
    elif dataset_name == "mechuspto31k":
        full_dataset_name =  "SchwallerGroup/uspto-31k_retro"


    evaluate_task_hf(
        dataset = full_dataset_name,
        task_col = f"{task}_task",
        model_path=os.path.join(os.path.dirname(__file__), f"checkpoints_hf_train_debug_{run}", f"checkpoint-{checkpoint}"),
        #model_path=os.path.join(os.path.dirname(__file__), f"checkpoints_hf_train_debug_{run}", f"checkpoint-{checkpoint}"),
        #model_path=os.path.join(os.path.dirname(__file__), "checkpoints_grpo", "model_30_102660", "checkpoint-1000"),
        tokenizer_path=os.path.join(os.path.dirname(__file__), "..", "tokenizer", "mechsmiles_tokenizer_folder"),
        num_beams=num_beams,
        num_lines_eval=-1,
        #num_lines_eval=1000,
        #parquet_file_path= os.path.join("data", "evaluations", "baybe_sweep_train", f"eval_model_{run}_{checkpoint}_{task}_top{num_beams}.parquet"),
        parquet_file_path= os.path.join("data", "evaluations", "hf_train", dataset_name, f"{model_type}_eval_model_{run}_{checkpoint}_{task}_top{num_beams}.parquet"),
        decoder_only = False if "t5" in model_type else True,
        batch_size=160//num_beams,
    )
