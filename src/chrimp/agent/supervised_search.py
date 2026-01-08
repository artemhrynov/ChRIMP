import os
from chrimp.visualization.mechanism_visualizer import MechanismVisualizer
import torch
from tqdm import tqdm
from colorama import Fore
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


import networkx as nx
from rdkit import Chem

from chrimp.world.mechsmiles import (
    MechSmiles,
    MechSmilesContextError,
    MechSmilesInitError,
)


def canonicalize_special(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    return Chem.MolToSmiles(mol)


def try_canonicalize_prod_msmi(msmi: MechSmiles):
    try:
        msmi.unhide_cond()
        return msmi.ms_prod.can_smiles, True
    except:  # noqa: E722 (Do not use bare except) # noqa: E722 (Do not use bare except)
        return "", False


def initialize_search_graph(start_smiles):
    """
    Initialize a search graph with start and goal nodes.

    Args:
        start_smiles: Canonicalized SMILES string for the start molecule

    Returns:
        G: NetworkX graph with initialized start and goal nodes
    """
    G = nx.DiGraph()

    # Add start node
    G.add_node(
        start_smiles,
        smiles=start_smiles,
        minimal_cost=0.0,
        dist_to_start=0.0,
        best_parent=None,
        expanded=False,
        goal=False,
    )

    return G


def search(
    start_smiles: str,
    goal_smiles: str,
    human_to_input,
    max_node_budget: int = 512,
    search_algo: str = "best_first",
    k_n_tuple: tuple = (4, 10),
    decoder_only: bool = False,
    tokenizer="",  # Tokenizer of string
    model="",  # Model or string
    search_temperature: float = 1.0,
    cost_of_move: float = 0,
    max_depth: int = 10,
    verbose: bool = False,
    save_mech_folder: str | None = None,
    show_images: bool = False,
    stop_at_best: bool = True,  # If False, the algorithm will only stop at max_budget and might return more paths.
):
    # Init the model
    top_k = k_n_tuple[0]
    num_beams = k_n_tuple[1]

    if save_mech_folder is not None and not os.path.isdir(save_mech_folder):
        os.makedirs(save_mech_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if isinstance(model, str):
        if decoder_only:
            model = AutoModelForCausalLM.from_pretrained(model)
            tokenizer.padding_side = "left"
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model)
            tokenizer.padding_side = "right"

    model.eval()
    model.to(device)

    def collate_fn(input):
        inputs = tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100_000,  # Way over training limit
            # max_length=tokenizer.model_max_length, #OverflowError: int too big to convert
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    # Init the graph

    try:
        start_smiles = canonicalize_special(start_smiles)
        goal_smiles = canonicalize_special(goal_smiles)
    except Chem.rdchem.AtomValenceException:
        print(
            f"{Fore.RED}Could not canonicalize_special:\n{start_smiles = }\n{goal_smiles = }"
        )
        return []

    curr_node_budget = 0
    G = initialize_search_graph(start_smiles)

    goal_set = set(goal_smiles.split("."))

    # Start the search
    curr_smiles = start_smiles
    curr_node = G.nodes[curr_smiles]
    curr_cost = 0.0
    curr_dist = 0
    max_cost_to_goal = np.inf

    pbar = tqdm(total=max_node_budget, desc="Search progress")

    while (
        curr_cost < max_cost_to_goal or (not stop_at_best)
    ) and curr_node_budget < max_node_budget:
        if verbose:
            print(f'{Fore.LIGHTYELLOW_EX}expanding "{curr_smiles}"{Fore.RESET}')
            print(f'{Fore.LIGHTYELLOW_EX}goal is   "{goal_smiles}"{Fore.RESET}')
        curr_node_budget += 1
        pbar.update(1)
        curr_node["expanded"] = True

        try:
            curr_expansion = human_to_input(f"{curr_smiles}>>{goal_smiles}")
        except ValueError:
            print(
                f'Weird behavior in human_to_input for:\n"{curr_smiles}>>{goal_smiles}"'
            )
            return []

        inputs = collate_fn(curr_expansion)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            num_beams=num_beams,
            # num_beam_groups=num_beam_groups,
            num_return_sequences=num_beams,
            do_sample=False,
            # diversity_penalty=1.0 if num_beam_groups != 1 else 0,
            return_dict_in_generate=True,
            output_scores=True,
        )

        if decoder_only:
            raise NotImplementedError("Not yet implemented for decoder only")
            # Something like:
            # for i in range(input_ids.shape[0]):
            #    outputs = generated_ids[i*num_beams:(i+1)*num_beams]
            #    texts = [
            #        '[mech]'.join(tokenizer.decode(out).split("[mech]")[1:]).replace(" ", "") for out in outputs
            #    ]

            #    texts = [t.split('[eos]')[0] + '[eos]' if 'eos' in t else t for t in texts]
            #    #references.append(gold_outputs[i] + tokenizer.eos_token)
            #    inputs_list.append(tokenizer.decode(input_ids[i]).replace(" ", "").split('[pad]')[-1]) # Remove the left padding
            #    predictions.append(texts)

        else:
            decoded_preds = tokenizer.batch_decode(
                generated_ids[0], skip_special_tokens=True
            )

        raw_beam_scores = generated_ids.sequences_scores
        beam_scores = -torch.pow(
            -(raw_beam_scores - cost_of_move), 1.0 / search_temperature
        )
        if verbose:
            print(f"{curr_cost = }, {max_cost_to_goal = }")
            print(f"{beam_scores=}")
        decoded_preds_msmi = []
        for pred, score in zip(decoded_preds, beam_scores):
            try:
                msmi_str = pred.replace(" ", "")
                decoded_preds_msmi.append(
                    (MechSmiles(msmi_str, context=curr_smiles), msmi_str, score)
                )
            except MechSmilesContextError:
                if verbose:
                    print(f"context error {msmi_str=}\n{curr_smiles=}")
                continue
            except MechSmilesInitError:
                if verbose:
                    print("init error")
                continue

        decoded_preds_can_bool = [
            (try_canonicalize_prod_msmi(x), mss, s) for x, mss, s in decoded_preds_msmi
        ]

        next_smiles = set()
        next_smiles_scores = []
        for (pred, could_can), msmi_str, score in decoded_preds_can_bool:
            if not could_can or pred in next_smiles:
                continue
            next_smiles_scores.append((pred, msmi_str, score))

        # Keep the top-k nodes
        next_smiles_scores = next_smiles_scores[:top_k]

        # Treat the nodes one by one and add them in the graph:
        for smiles, msmi_str, score in next_smiles_scores:
            if verbose:
                print(f"{msmi_str=}")
            edge_cost = -score.item()
            new_cost = curr_node["minimal_cost"] + edge_cost

            # If SMILES is already in the graph, look if values need update
            if smiles in G.nodes:
                if new_cost < G.nodes[smiles]["minimal_cost"]:
                    # Found a better path to this node
                    G.nodes[smiles]["minimal_cost"] = new_cost
                    G.nodes[smiles]["dist_to_start"] = curr_dist + 1
                    G.nodes[smiles]["best_parent"] = curr_smiles
                    if goal_set.issubset(set(smiles.split("."))):
                        if new_cost < max_cost_to_goal:
                            max_cost_to_goal = new_cost
                            print(
                                f"{Fore.GREEN}Improved path cost ({max_cost_to_goal:.4f}) ({edge_cost:.4f}){Fore.RESET}"
                            )
                            G.nodes[smiles]["goal"] = True
                    # Update or add edge
                    if G.has_edge(curr_smiles, smiles):
                        G[curr_smiles][smiles]["cost"] = edge_cost
                        G[curr_smiles][smiles]["msmi_str"] = msmi_str
                    else:
                        G.add_edge(
                            curr_smiles, smiles, cost=edge_cost, msmi_str=msmi_str
                        )

            # If SMILES is not in the graph yet, connect it to its parent correctly
            else:
                try:
                    can_next_set = set(
                        [canonicalize_special(x) for x in smiles.split(".")]
                    )
                except Exception:
                    continue

                if goal_set.issubset(can_next_set):
                    if new_cost < max_cost_to_goal:
                        max_cost_to_goal = new_cost
                        print(
                            f"{Fore.GREEN}Found at least one path ({max_cost_to_goal:.4f}) ({edge_cost:.4f}){Fore.RESET}"
                        )

                    G.add_node(
                        smiles,
                        smiles=smiles,
                        minimal_cost=curr_cost + edge_cost,
                        dist_to_start=curr_dist + 1,
                        best_parent=curr_smiles,
                        expanded=False,
                        goal=True,
                    )
                else:
                    G.add_node(
                        smiles,
                        smiles=smiles,
                        minimal_cost=curr_cost + edge_cost,
                        dist_to_start=curr_dist + 1,
                        best_parent=curr_smiles,
                        expanded=False,
                        goal=False,
                    )

                G.add_edge(curr_smiles, smiles, cost=edge_cost, msmi_str=msmi_str)

        # Select unexpanded node with lowest cost
        unexpanded = [
            n
            for n in G.nodes
            if not (
                G.nodes[n]["goal"]
                or G.nodes[n]["expanded"]
                or G.nodes[n]["dist_to_start"] > max_depth
            )
        ]
        if not unexpanded:
            print(f"{Fore.RED}Not anything remaining to expand{Fore.RESET}")
            break
        if search_algo == "best_first":
            curr_smiles = min(unexpanded, key=lambda n: G.nodes[n]["minimal_cost"])
        elif search_algo == "breadth_first":
            curr_smiles = min(unexpanded, key=lambda n: G.nodes[n]["dist_to_start"])
        curr_node = G.nodes[curr_smiles]
        curr_cost = G.nodes[curr_smiles]["minimal_cost"]
        curr_dist = G.nodes[curr_smiles]["dist_to_start"]

    pbar.close()

    all_goals = [n for n in G.nodes if (G.nodes[n]["goal"])]

    if len(all_goals) > 0:
        print(f"{Fore.GREEN}Solved!{Fore.RESET}")

        # Sort all goal nodes by minimal_cost to get alternatives
        sorted_goals = sorted(all_goals, key=lambda n: G.nodes[n]["minimal_cost"])

        all_alternatives = []  # List of lists, each containing path msmis

        for alt_idx, final_goal_smiles in enumerate(sorted_goals):
            # Reconstruct path for this alternative
            reversed_path = [G.nodes[final_goal_smiles]]
            while reversed_path[-1]["best_parent"] is not None:
                reversed_path.append(G.nodes[reversed_path[-1]["best_parent"]])
            path = reversed_path[::-1]

            all_msmis = []

            # Print header for this alternative
            if alt_idx == 0:
                print(
                    f"\n{Fore.GREEN}=== Best Path (Cost: {G.nodes[final_goal_smiles]['minimal_cost']:.2f}) ==={Fore.RESET}"
                )
            else:
                print(
                    f"\n{Fore.YELLOW}=== Alternative {alt_idx} (Cost: {G.nodes[final_goal_smiles]['minimal_cost']:.2f}) ==={Fore.RESET}"
                )

            print(f"{Fore.GREEN}Start:{Fore.RESET} {start_smiles}")

            for i in range(len(path) - 1):
                msmi_str = G[path[i]["smiles"]][path[i + 1]["smiles"]]["msmi_str"]
                all_msmis.append(msmi_str)
                print(msmi_str)

                if show_images:
                    msmi = MechSmiles(msmi_str)
                    # Save with alternative prefix
                    save_prefix = (
                        "whole_mechanism" if alt_idx == 0 else f"alternative_{alt_idx}"
                    )
                    msmi.show_reac(
                        save_path=os.path.join(
                            save_mech_folder, f"{save_prefix}_step_{i+1}"
                        )
                        if save_mech_folder is not None
                        else None
                    )

            if show_images:
                save_prefix = (
                    "whole_mechanism" if alt_idx == 0 else f"alternative_{alt_idx}"
                )
                msmi.show_prod(
                    save_path=os.path.join(
                        save_mech_folder, f"{save_prefix}_step_{len(path)}"
                    )
                    if save_mech_folder is not None
                    else None
                )

            print(f"{Fore.LIGHTYELLOW_EX}Goal:{Fore.RESET} {goal_smiles}")

            # Save the whole mechanism visualization for this alternative
            if save_mech_folder is not None:
                save_prefix = "whole_mech" if alt_idx == 0 else f"alternative_{alt_idx}"
                MechanismVisualizer(all_msmis).show(
                    save_path=os.path.join(save_mech_folder, f"{save_prefix}.svg")
                )

            all_alternatives.append(all_msmis)

        # Save start and product images once (shared across all alternatives)
        if save_mech_folder is not None:
            MechSmiles(f"{start_smiles}|").show_reac(
                save_path=os.path.join(save_mech_folder, "reac")
            )
            MechSmiles(f"{goal_smiles}|").show_reac(
                save_path=os.path.join(save_mech_folder, "prod")
            )

        return all_alternatives

    # if len(all_goals) > 0:
    #    print(f"{Fore.GREEN}Solved!{Fore.RESET}")

    #    final_goal_smiles = min(all_goals, key=lambda n: G.nodes[n]['minimal_cost'])
    #    reversed_path = [G.nodes[final_goal_smiles]]
    #    while not (reversed_path[-1]['best_parent'] is None):
    #        reversed_path.append(G.nodes[reversed_path[-1]['best_parent']])

    #    path = reversed_path[::-1]
    #    all_msmis = []
    #    print(f"{Fore.GREEN}Start:{Fore.RESET} {start_smiles}")
    #    for i in range(len(path)-1):
    #        msmi_str = G[path[i]['smiles']][path[i+1]['smiles']]['msmi_str']
    #        all_msmis.append(msmi_str)
    #        print(msmi_str)
    #        if show_images:
    #            msmi = MechSmiles(msmi_str)
    #            msmi.show_reac(save_path=os.path.join(save_mech_folder,f"step_{i+1}") if save_mech_folder is not None else None)

    #    if show_images:
    #        msmi.show_prod(save_path=os.path.join(save_mech_folder,f"step_{len(path)}") if save_mech_folder is not None else None)
    #    print(f"{Fore.LIGHTYELLOW_EX}Goal:{Fore.RESET} {goal_smiles}")

    #    if save_mech_folder is not None:
    #        # Not the cleanest, ideally we would need a proper way to show the reaction
    #        MechSmiles(f"{start_smiles}|").show_reac(save_path=os.path.join(save_mech_folder,f"reac"))
    #        MechSmiles(f"{goal_smiles}|").show_reac(save_path=os.path.join(save_mech_folder,f"prod"))

    #        MechanismVisualizer(all_msmis).show(save_path=os.path.join(save_mech_folder,f"whole_mech.svg"))

    #    return all_msmis

    else:
        print(f"{Fore.BLUE}Unsolved...{Fore.RESET}")
        return []
