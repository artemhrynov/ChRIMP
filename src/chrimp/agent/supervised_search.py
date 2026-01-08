import re
from chrimp.agent.supervised_evaluate import evaluation_metrics_wo_groundtruth
from chrimp.visualization.mechanism_visualizer import MechanismVisualizer
import torch
from tqdm import tqdm
from colorama import Fore
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


import networkx as nx
from rdkit import Chem

from chrimp.world.mechsmiles import MechSmiles, MechSmilesContextError, MechSmilesInitError


def canonicalize_special(smiles):
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return Chem.MolToSmiles(mol)



def try_canonicalize_prod_msmi(msmi: MechSmiles):
    try:
        msmi.unhide_cond()
        return  msmi.ms_prod.can_smiles, True
    except:
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
    start_smiles:str,
    goal_smiles:str,
    human_to_input,
    max_node_budget:int = 512,
    search_algo: str = "best_first",
    k_n_tuple: tuple = (4, 10),
    decoder_only: bool = False,
    tokenizer = "", # Tokenizer of string
    model = "", # Model or string
    search_temperature: float = 1.0,
    cost_of_move: float = 0,
    max_depth: int = 10,
    verbose: bool = False,
    save_mech_folder: str | None = None,
    show_images: bool = False,
    stop_at_best: bool = True, # If False, the algorithm will only stop at max_budget and might return more paths.
):
    # Init the model
    top_k = k_n_tuple[0]
    num_beams = k_n_tuple[1]

    if save_mech_folder is not None and not os.path.isdir(save_mech_folder):
        os.makedirs(save_mech_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if isinstance(model, str):
        if decoder_only:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer.padding_side = "left"
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer.padding_side = "right"

    model.eval()
    model.to(device)

    def collate_fn(input):
        inputs = tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100_000, # Way over training limit
            #max_length=tokenizer.model_max_length, #OverflowError: int too big to convert
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    # Init the graph

    try:
        start_smiles = canonicalize_special(start_smiles)
        goal_smiles = canonicalize_special(goal_smiles)
    except Chem.rdchem.AtomValenceException:
        print(f"{Fore.RED}Could not canonicalize_special:\n{start_smiles = }\n{goal_smiles = }")
        return []


    curr_node_budget = 0
    G = initialize_search_graph(start_smiles)

    goal_set = set(goal_smiles.split('.'))

    # Start the search
    curr_smiles = start_smiles
    curr_node = G.nodes[curr_smiles]
    curr_cost = 0.0
    curr_dist = 0
    max_cost_to_goal = np.inf

    pbar = tqdm(total=max_node_budget, desc="Search progress")

    while (curr_cost < max_cost_to_goal or (not stop_at_best)) and curr_node_budget < max_node_budget:
        if verbose:
            print(f'{Fore.LIGHTYELLOW_EX}expanding "{curr_smiles}"{Fore.RESET}')
            print(f'{Fore.LIGHTYELLOW_EX}goal is   "{goal_smiles}"{Fore.RESET}')
        curr_node_budget += 1
        pbar.update(1)
        curr_node['expanded'] = True

        try:
            curr_expansion = human_to_input(f"{curr_smiles}>>{goal_smiles}")
        except ValueError:
            print(f"Weird behavior in human_to_input for:\n\"{curr_smiles}>>{goal_smiles}\"")
            return []

        inputs = collate_fn(curr_expansion)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            num_beams=num_beams,
            #num_beam_groups=num_beam_groups,
            num_return_sequences=num_beams,
            do_sample=False,
            # diversity_penalty=1.0 if num_beam_groups != 1 else 0,
            return_dict_in_generate=True,
            output_scores=True,
        )

        references = []
        inputs_list = []
        predictions = []

        if decoder_only:

            raise NotImplementedError("Not yet implemented for decoder only")
            #for i in range(input_ids.shape[0]):
            #    outputs = generated_ids[i*num_beams:(i+1)*num_beams]
            #    texts = [
            #        '[mech]'.join(tokenizer.decode(out).split("[mech]")[1:]).replace(" ", "") for out in outputs
            #    ]

            #    texts = [t.split('[eos]')[0] + '[eos]' if 'eos' in t else t for t in texts]
            #    #references.append(gold_outputs[i] + tokenizer.eos_token)
            #    inputs_list.append(tokenizer.decode(input_ids[i]).replace(" ", "").split('[pad]')[-1]) # Remove the left padding
            #    predictions.append(texts)

        else:
            decoded_preds = tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)

        raw_beam_scores = generated_ids.sequences_scores
        beam_scores = -torch.pow(-(raw_beam_scores - cost_of_move), 1.0 / search_temperature) 
        if verbose:
            print(f"{curr_cost = }, {max_cost_to_goal = }")
            print(f"{beam_scores=}")
        decoded_preds_msmi = []
        for pred, score in zip(decoded_preds, beam_scores):
            try:
                msmi_str = pred.replace(' ', '')
                decoded_preds_msmi.append((MechSmiles(msmi_str, context=curr_smiles), msmi_str, score))
            except MechSmilesContextError:
                if verbose:
                    print(f"context error {msmi_str=}\n{curr_smiles=}")
                continue
            except MechSmilesInitError:
                if verbose:
                    print("init error")
                continue

        decoded_preds_can_bool = [(try_canonicalize_prod_msmi(x), mss, s) for x, mss, s in decoded_preds_msmi]

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
            new_cost = curr_node['minimal_cost'] + edge_cost
            
            # If SMILES is already in the graph, look if values need update
            if smiles in G.nodes:
                if new_cost < G.nodes[smiles]['minimal_cost']:
                    # Found a better path to this node
                    G.nodes[smiles]['minimal_cost'] = new_cost
                    G.nodes[smiles]['dist_to_start'] = curr_dist + 1
                    G.nodes[smiles]['best_parent'] = curr_smiles
                    if goal_set.issubset(set(smiles.split('.'))):
                        if new_cost < max_cost_to_goal:
                            max_cost_to_goal = new_cost
                            print(f"{Fore.GREEN}Improved path cost ({max_cost_to_goal:.4f}) ({edge_cost:.4f}){Fore.RESET}")
                            G.nodes[smiles]['goal'] = True
                    # Update or add edge
                    if G.has_edge(curr_smiles, smiles):
                        G[curr_smiles][smiles]['cost'] = edge_cost
                        G[curr_smiles][smiles]['msmi_str'] = msmi_str
                    else:
                        G.add_edge(curr_smiles, smiles, cost=edge_cost, msmi_str=msmi_str)

            # If SMILES is not in the graph yet, connect it to its parent correctly
            else:
                try:
                    can_next_set = set([canonicalize_special(x) for x in smiles.split('.')])
                except Exception:
                    continue

                if goal_set.issubset(can_next_set):
                    if new_cost < max_cost_to_goal:
                        max_cost_to_goal = new_cost
                        print(f"{Fore.GREEN}Found at least one path ({max_cost_to_goal:.4f}) ({edge_cost:.4f}){Fore.RESET}")

                    G.add_node(
                        smiles,
                        smiles=smiles,
                        minimal_cost=curr_cost+edge_cost,
                        dist_to_start=curr_dist+1,
                        best_parent=curr_smiles,
                        expanded=False,
                        goal=True,
                    )
                else:
                    G.add_node(
                        smiles,
                        smiles=smiles,
                        minimal_cost=curr_cost+edge_cost,
                        dist_to_start=curr_dist+1,
                        best_parent=curr_smiles,
                        expanded=False,
                        goal=False,
                    )

                G.add_edge(curr_smiles, smiles, cost=edge_cost, msmi_str=msmi_str)

        # Select unexpanded node with lowest cost
        unexpanded = [n for n in G.nodes if not (G.nodes[n]['goal'] or G.nodes[n]['expanded'] or G.nodes[n]['dist_to_start']>max_depth)]
        if not unexpanded:
            print(f"{Fore.RED}Not anything remaining to expand{Fore.RESET}")
            break
        if search_algo == "best_first":
            curr_smiles = min(unexpanded, key=lambda n: G.nodes[n]['minimal_cost'])
        elif search_algo == "breadth_first":
            curr_smiles = min(unexpanded, key=lambda n: G.nodes[n]['dist_to_start'])
        curr_node = G.nodes[curr_smiles]
        curr_cost = G.nodes[curr_smiles]['minimal_cost']
        curr_dist = G.nodes[curr_smiles]['dist_to_start']

    pbar.close()

    all_goals = [n for n in G.nodes if (G.nodes[n]['goal'])]

    if len(all_goals) > 0:
        print(f"{Fore.GREEN}Solved!{Fore.RESET}")
        
        # Sort all goal nodes by minimal_cost to get alternatives
        sorted_goals = sorted(all_goals, key=lambda n: G.nodes[n]['minimal_cost'])
        
        all_alternatives = []  # List of lists, each containing path msmis
        
        for alt_idx, final_goal_smiles in enumerate(sorted_goals):
            # Reconstruct path for this alternative
            reversed_path = [G.nodes[final_goal_smiles]]
            while not (reversed_path[-1]['best_parent'] is None):
                reversed_path.append(G.nodes[reversed_path[-1]['best_parent']])
            path = reversed_path[::-1]
            
            all_msmis = []
            
            # Print header for this alternative
            if alt_idx == 0:
                print(f"\n{Fore.GREEN}=== Best Path (Cost: {G.nodes[final_goal_smiles]['minimal_cost']:.2f}) ==={Fore.RESET}")
            else:
                print(f"\n{Fore.YELLOW}=== Alternative {alt_idx} (Cost: {G.nodes[final_goal_smiles]['minimal_cost']:.2f}) ==={Fore.RESET}")
            
            print(f"{Fore.GREEN}Start:{Fore.RESET} {start_smiles}")
            
            for i in range(len(path)-1):
                msmi_str = G[path[i]['smiles']][path[i+1]['smiles']]['msmi_str']
                all_msmis.append(msmi_str)
                print(msmi_str)
                
                if show_images:
                    msmi = MechSmiles(msmi_str)
                    # Save with alternative prefix
                    save_prefix = "whole_mechanism" if alt_idx == 0 else f"alternative_{alt_idx}"
                    msmi.show_reac(
                        save_path=os.path.join(save_mech_folder, f"{save_prefix}_step_{i+1}") 
                        if save_mech_folder is not None else None
                    )
            
            if show_images:
                save_prefix = "whole_mechanism" if alt_idx == 0 else f"alternative_{alt_idx}"
                msmi.show_prod(
                    save_path=os.path.join(save_mech_folder, f"{save_prefix}_step_{len(path)}") 
                    if save_mech_folder is not None else None
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
            MechSmiles(f"{start_smiles}|").show_reac(save_path=os.path.join(save_mech_folder, f"reac"))
            MechSmiles(f"{goal_smiles}|").show_reac(save_path=os.path.join(save_mech_folder, f"prod"))
        
        return all_alternatives

    #if len(all_goals) > 0:
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


if __name__ == "__main__":
    import os
    #full_reaction = "CCN=C=O.O=CC1=C(O)C(F)=CC(C)=C1>>CCNC(=O)OC1=C(O)C(F)=CC(C)=C1"
    #full_reaction = "CC[N-]C(=O)[OH+]C1=C(C=O)C=C(C)C=C1F>>CCNC(=O)OC1=C(O)C(F)=CC(C)=C1"
    #full_reaction = "C#CC1CC1.C1CCOC1.C[CH2][Mg][Br].[N-]=[N+]=NC1=NC2=C(CCOC3=CC(Br)=CC=C32)S1>>BrC1=CC=C2C(=C1)OCCC1=C2N=C(N2N=NC=C2C2CC2)S1.C1CCOC1.C[CH2][Mg][Br]"


    # Flower test set (dist = 3)
    #full_reaction = "CC(=O)[O-].CC(=O)[O-].CC(C)([OH2+])N1CCC(CN2C(=O)C3(COC4=CC5=C(C=C43)CCO5)C3=CC=CC=C32)CC1.CO.O=C(O)O.[BH3-]C#N.[Na+].[Na+]>>CC(=O)[O-].CC(=O)[O-].CC(C)([OH2+])N1CCC(CN2C(=O)C3(COC4=CC5=C(C=C43)CCO5)C3=CC=CC=C32)CC1.CO.O=C(O)O.[BH3-]C#N.[Na+].[Na+]"
    #full_reaction = "CC1=CC(N)=CC=C1.ClCCl.O=C(Cl)CBr.[Na+].[OH-]>>CC1=CC=CC(NC(=O)CBr)=C1.ClCCl.O.[Cl-].[Na+]"

    # Flower test set (dist = 4) 
    #full_reaction = "CN(C)C=O.C[S-].ClC1=CC(C2=CC=CC=C2)=NC=N1.O.[Na+]>>CN(C)C=O.CSC1=CC(C2=CC=CC=C2)=NC=N1.O.[Cl-].[Na+]"

    #Weird things happening when we shuffle the aromatic double bonds of the reactant
    #full_reaction = "CN(C)C=O.C[S-].ClC1=CC(C2=CC=CC=C2)=NC=N1.O.[Na+]>>CN(C)C=O.CSC1C=C(C2=CC=CC=C2)N=CN=1.O.[Cl-].[Na+]"
    #full_reaction = "CN(C)C=O.C[S-].ClC1C=C(C2=CC=CC=C2)N=CN=1.O.[Na+]>>CN(C)C=O.CSC1=CC(C2=CC=CC=C2)=NC=N1.O.[Cl-].[Na+]"
    #full_reaction = "CN(C)C=O.C[S-].ClC1C=C(C2=CC=CC=C2)N=CN=1.O.[Na+]>>CN(C)C=O.CSC1C=C(C2=CC=CC=C2)N=CN=1.O.[Cl-].[Na+]"

    ## Flower test set (dist = 5)
    #full_reaction = "CCOC(=O)C(=O)NC1=CC=CC(OC)=C1C(N)=O.O>>CCO.COC1=C(C(N)=O)C(NC(=O)C(=O)O)=CC=C1"

    #full_reaction = "CCOC(=O)C[NH+]1C(=O)C(=O)N(C)C1=O.[OH-]>>CCOC(=O)CN1C(=O)C(=O)N(C)C1=O.O"


    # PA routes paper (fig. 1):
    multistep = [
        "CCc1cccc(CC)c1N.BrCC=O>>CCc1cccc(CC)c1N(CC=O)",
        "ClC(=O)CCl.CCc1cccc(CC)c1N(CC=O)>>CCc1cccc(CC)c1N(CC=O)C(=O)CCl",
    ]

    ## PA routes paper (fig. S1 a)
    #multistep = [
    #    "COC(=O)c1ccc(CCNC2)c2c1.CN>>CNC(=O)c1ccc(CCNC2)c2c1",
    #    "CNC(=O)c1ccc(CCNC2)c2c1.C1CCC1N1CCNC(Cl)C1.CC(=O)N>>CNC(=O)c1ccc(c2c1)CCN(C2)CC(=O)N1CCN(CC1)C1CCC1"
    #]

    ## PA routes paper (fig. S1 a (corrected))
    #multistep = [
    #    "COC(=O)c1ccc(CCNC2)c2c1.CN.C[O-]>>CNC(=O)c1ccc(CCNC2)c2c1",
    #    "CNC(=O)c1ccc(CCNC2)c2c1.C1N(CCN(C1)C1CCC1)C(=O)CCl>>CNC(=O)c1ccc(c2c1)CCN(C2)CC(=O)N1CCN(CC1)C1CCC1"
    #]

    ## PA routes paper (fig. S1 b)
    #multistep = [
    #    "C1CCC1N1CCNC(Cl)C1.COC(=O)c1ccc(CCNC2)c2c1.CC(=O)N>>COC(=O)c1ccc(c2c1)CCN(C2)CC(=O)N1CCN(CC1)C1CCC1",
    #    "COC(=O)c1ccc(c2c1)CCN(C2)CC(=O)N1CCN(CC1)C1CCC1.CN>>CNC(=O)c1ccc(c2c1)CCN(C2)CC(=O)N1CCN(CC1)C1CCC1"
    #]

    ## PARoute paper (fig. S2)
    #multistep = [
    #    "Oc1cccc(c1)C1CCCC1.BrCc1ccccc1>>c1ccccc1COc1cccc(c1)C1CCCC1",
    #    "c1ccccc1COc1cccc(c1)C1CCCC1.BrN1C(=O)CCC1(=O)>>c1ccccc1COc1ccc(Br)c(c1)C1CCCC1",
    #]

    # oMeBench preprint fig. 1
    #multistep = [
    #    "CC=O.C=O.[Na]O>>O=CCCO"
    #]


    # PaRoutes_n5_unrecognized_by_namerxn
    #multistep = [
    #    #"Cc1ccc(cc1)S(=O)(=O)N(Cc2c(cc(c(=O)n2CC3CCCCC3)Br)C(=O)OC)CC(=O)OC.CO>CO.[Na+].Cl>COC(=O)c1c(c2cc(c(=O)n(c2cn1)CC3CCCCC3)Br)O",
    #    #"CC(=C(C=CC(=O)OC)C(=O)OC)NCC1CCCCC1.C1CC(=O)N(C1=O)Br>CO.CO.[Na+]>Cc1c(cc(c(=O)n1CC2CCCCC2)Br)C(=O)OC",
    #    #"CN(C(=O)c1c(=O)c(cn(n1)c2cccc(c2)C(F)(F)F)OC)OC.C[Mg+]>C1CCOC1.Br>CC(=O)c1c(=O)c(cn(n1)c2cccc(c2)C(F)(F)F)OC",
    #    #"CN(C)C(OC)OC.COCC(=O)C(=NNc1cccc(c1)C(F)(F)F)C(=O)OC>>COc1cn(nc(c1=O)C(=O)OC)c2cccc(c2)C(F)(F)F",
    #    #"COC(=O)c1ccc(nc1Cl)CNC=O>CCOC(=O)C.Cc1ccccc1.O.O(Cl)Cl.[Na+].[P+5]>COC(=O)c1ccc2cncn2c1Cl",
    #    #"CCC1Cc2cc(c(cc2C=N1)OC)OCc3ccccc3.CCOC=C(C(=O)C)C(=O)OCC>CCO>CCC1Cc2cc(c(cc2C3N1C=C(C(=O)C3)C(=O)OCC)OC)OCc4ccccc4",
    #    #"CCC(=Cc1ccc(c(c1)OCc2ccccc2)OC)[N+](=O)[O-]>[HH].[HH].[HH].[HH].[Li+].C1CCOC1.O.O.[Na+].[Al+3]>CCC(Cc1ccc(c(c1)OCc2ccccc2)OC)N",
    #    #"C[Mg+].c1ccc(cc1)CN2CC(Oc3c2cc(c(c3)C=O)Cl)C(=O)N4CCC(CC4)(Cc5ccc(cc5)F)C#N>C1CCOC1.Br>CC(c1cc2c(cc1Cl)N(CC(O2)C(=O)N3CCC(CC3)(Cc4ccc(cc4)F)C#N)Cc5ccccc5)O",
    #    #"CC(C)(C)OC(=O)N1CCN(CC1)C[C@H](CN(Cc2ccccc2)Cc3ccccc3)OC>B(OC(=O)C)(OC(=O)C)OC(=O)C.CC#N.C=O.C1COCCO1.C(Cl)Cl.N.O.[Na+].Cl>CN1CCN(CC1)C[C@H](CN(Cc2ccccc2)Cc3ccccc3)OC",
    #    #"CCOC(=O)C#CC.COc1cc(cc2c1[nH]c(c2)C(=S)N)Oc3ccc(nc3)S(=O)(=O)C>CCCCP(CCCC)CCCC.C1CCOC1>CCOC(=O)CC1CN=C(S1)c2cc3cc(cc(c3[nH]2)OC)Oc4ccc(nc4)S(=O)(=O)C",
    #    #"c1cc(ccc1c2ccnc(c2c3ccncc3)NN)Cl.c1cn(cn1)C(=O)n2ccnc2>C1CCOC1.O>c1cc(ccc1c2ccn3c(c2c4ccncc4)n[nH]c3=O)Cl",
    #    #"CC(c1ccc(c(c1)O)N)C(=O)O.C(#N)Br>O.O.[Na+]>CC(c1ccc2c(c1)oc(n2)N)C(=O)O",
    #    #"CO.C[Si](C)(C)OC[C@@H]1[C@H]([C@@H]([C@H](C(=O)O1)O[Si](C)(C)C)O[Si](C)(C)C)O[Si](C)(C)C.c1cc(ccc1Cc2cc(ccc2Cl)Br)COC3CC3>[Li]CCCC.Cc1ccccc1.Cc1ccccc1.CS(=O)(=O)O.C1CCOC1>COC1([C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O)c2ccc(c(c2)Cc3ccc(cc3)COC4CC4)Cl",
    #    #"CC(=O)c1c(=O)c(cn(n1)c2cc(c(cc2F)N3CCOCC3)F)OC.CN(C)C(OC)OC.c1ccc(cc1)NN>CCOC(=O)C>COc1cn(nc(c1=O)c2ccnn2c3ccccc3)c4cc(c(cc4F)N5CCOCC5)F",
    #    #"CN(C(=O)c1c(=O)c(cn(n1)c2cc(c(cc2F)N3CCOCC3)F)OC)OC.C[Mg+]>C1CCOC1.Br>CC(=O)c1c(=O)c(cn(n1)c2cc(c(cc2F)N3CCOCC3)F)OC",
    #    #"CN(C)C(OC)OC.COCC(=O)C(=NNc1cc(c(cc1F)N2CCOCC2)F)C(=O)OC>>COc1cn(nc(c1=O)C(=O)OC)c2cc(c(cc2F)N3CCOCC3)F",
    #    #"Cc1ccc2ccccc2c1CC(=O)N(C)OC>C1CCOC1.O>Cc1ccc2ccccc2c1CC(=O)C",
    #    #"CS(=O)(=O)Cc1cccc2c1[nH]cc2.c1cc(c(cc1Cl)F)C(C2CC2C#N)O>C(Cl)Cl.C(=O)(C(F)(F)F)O.Cl.Cl.Cl.[In+3]>CS(=O)(=O)Cc1cccc2c1[nH]cc2C(c3ccc(cc3F)Cl)C4CC4C#N",
    #    #"CCOC(=O)/C=C/c1cc2c(c(c1N)Cl)CCN(CC2)C(=O)OCC>CO.C(Cl)(Cl)Cl.O=[Pt]=O>CCOC(=O)N1CCc2cc3c(c(c2CC1)Cl)NC(=O)CC3",
    #    #"CCOC(=O)C(c1ccc(cc1)OCc2ccccc2)C(=O)OCC>B.CCO.[Na+].Cl>c1ccc(cc1)COc2ccc(cc2)C(CO)CO",
    #]

    # PaRoutes_n5_recognized_by_namerxn
    #multistep = [
    #    #"[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][O][c]2[cH][c]([cH][cH][c]2[F])[CH]([CH2][C](=[O])[O]C)[CH]3[CH2][CH2]3)[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]>CO.C1CCOC1.O.O.[Na+]>[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][O][c]2[cH][c]([cH][cH][c]2[F])[CH]([CH2][C](=[O])[OH])[CH]3[CH2][CH2]3)[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]",
    #    #"[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2]Cl)[c]2[cH][c]([cH][cH][c]2[F])[O][CH3].[CH3][O][C](=[O])[CH2][CH]([c]1[cH][cH][c]([c]([cH]1)[OH])[F])[CH]2[CH2][CH2]2>CC#N.C(=O)(O)O.[Cs+].[Cs+]>[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][O][c]2[cH][c]([cH][cH][c]2[F])[CH]([CH2][C](=[O])[O][CH3])[CH]3[CH2][CH2]3)[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]",
    #    #"C[O][c]1[cH][c]([cH][cH][c]1[F])[CH]([CH2][C](=[O])[O][CH3])[CH]2[CH2][CH2]2>B(Br)(Br)Br.C(Cl)Cl.O>[CH3][O][C](=[O])[CH2][CH]([c]1[cH][cH][c]([c]([cH]1)[OH])[F])[CH]2[CH2][CH2]2",
    #    #"[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2]O)[c]2[cH][c]([cH][cH][c]2[F])[O][CH3].CS(=O)(=O)[Cl]>CCN(CC)CC.C(Cl)Cl>[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][Cl])[c]2[cH][c]([cH][cH][c]2[F])[O][CH3]",
    #    #"[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][O][Si](c2ccccc2)(c3ccccc3)C(C)(C)C)[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]>CCCC[N+](CCCC)(CCCC)CCCC.C1CCOC1.O.F>[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][OH])[c]2[cH][c]([cH][cH][c]2[F])[O][CH3]",
    #    #"[CH3][C]([CH3])([CH3])[CH2][Mg]Cl.[CH3][C]([CH3])([CH3])[Si]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)[O][CH2][c]3[cH][cH][c]([c]([n]3)Cl)[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]>CCOCC.CC(C)c1cccc(c1N2CC[N+](=[C-]2)c3c(cccc3C(C)C)C(C)C)C(C)C.C1CCOC1.Cl>[CH3][C]([CH3])([CH3])[CH2][c]1[c]([cH][cH][c]([n]1)[CH2][O][Si]([c]2[cH][cH][cH][cH][cH]2)([c]3[cH][cH][cH][cH][cH]3)[C]([CH3])([CH3])[CH3])[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]",
    #    #"[CH3][C]([CH3])([CH3])[Si]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)Cl.[CH3][O][c]1[cH][cH][c]([c]([cH]1)[c]2[cH][cH][c]([n][c]2[Cl])[CH2][OH])[F]>CN(C)C=O.c1cnc[nH]1.O>[CH3][C]([CH3])([CH3])[Si]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)[O][CH2][c]3[cH][cH][c]([c]([n]3)[Cl])[c]4[cH][c]([cH][cH][c]4[F])[O][CH3]",
    #    ##"[CH3][O][c]1[cH][cH][c]([c]([cH]1)[c]2[cH][cH][c]([n][c]2[Cl])[C](=[O])OC)[F]>B.CCO.C1CCOC1.C(=O)(O)O.[Na+].[Na+].Cl.Cl.Cl.[Ca+2]>[CH3][O][c]1[cH][cH][c]([c]([cH]1)[c]2[cH][cH][c]([n][c]2[Cl])[CH2][OH])[F]",
    #    ##"B([c]1[cH][c]([cH][cH][c]1[F])[O][CH3])(O)O.[CH3][O][C](=[O])[c]1[cH][cH][c]([c]([n]1)[Cl])Br>Cc1ccccc1.C(=O)(O)O.O.[Na+].[Na+]>[CH3][O][c]1[cH][cH][c]([c]([cH]1)[c]2[cH][cH][c]([n][c]2[Cl])[C](=[O])[O][CH3])[F]",
    #    #"[CH3][O][C](=[O])[c]1[cH][cH][c]([cH][n+]1[O-])[Br].O=P([Cl])(Cl)Cl>>[CH3][O][C](=[O])[c]1[cH][cH][c]([c]([n]1)[Cl])[Br]",
    #    #"[CH3][CH2][c]1[c]([cH][c]2[c]([cH][n][n][c]2[cH]1)Br)[O][CH3].[CH2]1[c]2[nH][n][c]([c]2[CH2][NH][CH2]1)[C](=[O])[NH][CH]3[CH2][CH2]3>Cc1ccccc1.CC(O)(C)C.c1ccc(cc1)/C=C/C(=O)/C=C/c2ccccc2.c1ccc(cc1)/C=C/C(=O)/C=C/c2ccccc2.c1ccc(cc1)/C=C/C(=O)/C=C/c2ccccc2.C(=O)(O)C(F)(F)F.[Na+].[Pd].[Pd]>[CH3][CH2][c]1[c]([cH][c]2[c]([cH][n][n][c]2[cH]1)[N]3[CH2][c]4[c]([n][nH][c]4[CH2][CH2]3)[C](=[O])[NH][CH]5[CH2][CH2]5)[O][CH3]",
    #    #"[CH3][CH2][c]1[cH][c]2[c]([cH][c]1[O][CH3])[c]([cH][n][n]2)O.O=P([Br])(Br)Br>CC#N.C(=O)(O)O.O.[Na+].[Na+]>[CH3][CH2][c]1[cH][c]2[c]([cH][c]1[O][CH3])[c]([cH][n][n]2)[Br]",
    #    ##"[CH3][CH2][c]1[cH][c]([c]([cH][c]1[O][CH3])[C](=[O])[CH3])[NH2].[N](=O)O>C(=O)(O)O.O.[Na+].[Na+].[Na+].Cl>[CH3][CH2][c]1[cH][c]2[c]([cH][c]1[O][CH3])[c]([cH][n][n]2)[OH]",
    #    ##"[CH3][CH2][c]1[cH][c]([c]([cH][c]1[O][CH3])[C](=[O])[CH3])[N+](=O)[O-]>CC(=O)O.N.O.[Fe]>[CH3][CH2][c]1[cH][c]([c]([cH][c]1[O][CH3])[C](=[O])[CH3])[NH2]",
    #    #"[CH3][CH2][c]1[cH][cH][c]([cH][c]1[O][CH3])[C](=[O])[CH3].[N+](=[O])(O)[O-]>CC(=O)O.O>[CH3][CH2][c]1[cH][c]([c]([cH][c]1[O][CH3])[C](=[O])[CH3])[N+](=[O])[O-]",
    #    #"[CH3][CH2][c]1[cH][cH][c]([cH][c]1[OH])[C](=[O])[CH3].[CH3]I>CC(=O)C.C(=O)(O)O.[K+].[K+]>[CH3][CH2][c]1[cH][cH][c]([cH][c]1[O][CH3])[C](=[O])[CH3]",
    #    ##"[CH3][CH2][c]1[cH][cH][c]([cH][c]1N)[C](=[O])[CH3].N(=O)O>C(=O)(N)N.[OH2].OS(=O)(=O)O.[Na+]>[CH3][CH2][c]1[cH][cH][c]([cH][c]1[OH])[C](=[O])[CH3]",
    #    ##"[CH3][CH2][c]1[cH][cH][c]([cH][c]1[N+](=O)[O-])[C](=[O])[CH3]>CC(=O)O.N.O.[Fe]>[CH3][CH2][c]1[cH][cH][c]([cH][c]1[NH2])[C](=[O])[CH3]",
    #    #"[CH3][CH2][c]1[cH][cH][c]([cH][cH]1)[C](=[O])[CH3].[N+](=[O])(O)[O-]>O.OS(=O)(=O)O>[CH3][CH2][c]1[cH][cH][c]([cH][c]1[N+](=[O])[O-])[C](=[O])[CH3]",
    #    #"[CH3][CH2][c]1[cH][cH][cH][cH][cH]1.[CH3][C](=[O])OC(=O)C>C(Cl)Cl.[Al+3].Cl.Cl.Cl.Cl>[CH3][CH2][c]1[cH][cH][c]([cH][cH]1)[C](=[O])[CH3]",
    #]

    # One of the above in basic conditions (found it needs basic cond. on the internet)
    #multistep = [
    #    "[CH3][O][C](=[O])[c]1[cH][cH][c]([cH][n+]1[O-])[Br].O=P([Cl])(Cl)Cl.[Na]O>>[CH3][O][C](=[O])[c]1[cH][cH][c]([c]([n]1)[Cl])[Br]",
    #]


    ## Other (Boc protection, Boc deprotection)
    #multistep = [
    #    "c1ccccc1N.CC(C)(C)OC(=O)OC(=O)OC(C)(C)C>>c1ccccc1NC(=O)OC(C)(C)C",
    #    "c1ccccc1NC(=O)OC(C)(C)C.C(F)(F)(F)C(=O)O>>c1ccccc1N",
    #]

    #import pandas as pd
    #import pickle
    #df = pd.read_csv("data/master_v6/master_v6_reactions.csv")
    #df = df[df['equilibrated']]
    ##multistep = df['smiles_no_bp']
    #multistep = [x for x in df['smiles_no_bp'].values if any([('@' in x), ('/' in x), ('\\' in x)])]
    #multistep = [re.sub(r'[@\\/]', '', x) for x in multistep]

    #folder_res = f"data/figures/master_v6_equ_min_model_17_stereo_removed"


    #import pandas as pd
    #import pickle
    #df = pd.read_csv("data/uspto/USPTO_MIT_sampling_test.csv")
    #multistep = df['reactions'].values
    ##folder_res = f"data/figures/USPTO_MIT_sampling_test_equ_min_model_17"
    #folder_res = f"data/figures/USPTO_MIT_sampling_test_equ_min_model_60"

    # Misc theo
    folder_res = None
    multistep = [
        #"C[O][CH]([O]C)[O][CH3].c1cc(c(c(c1O)[CH]=[O])Cl)F>CO.C(=O)([O-])[O-].[NH4+].[N+](=O)([O-])[O-].[Na+].[Na+]>C[O][CH](c1c(ccc(c1Cl)F)O)[O]C",
        #"CCCc1ccc(cc1)C=O.COC(OC)OC>CO.Cl>CCCc1ccc(cc1)C(OC)OC",
        #"C[OH].C(C(C(C([CH]=[O])O)O)[OH])O>Cl>C[O][CH]1C(C(C([O]1)CO)O)O",
        #"C[O:12][CH:13]([O:14]C)[O:16][CH3:17].COC(=O)CCCC[CH:9]=[O:10]>Cc1ccc(cc1)S(=O)(=O)O.CO>C[O:12][CH:9](CCCCC(=O)OC)[O:14]C",
        #"COC(=O)CC(=O)OC.O=C1C=CCC1.[OH-]>>COC(=O)C(C(=O)OC)C1C=C(O)CC1.[OH-]",
        #"[CH3:1]C(=O)c1c2ccccn2nc1c3ccccc3.[CH:20](=[O:19])C(=O)O>C(Cl)Cl.O>c1ccc(cc1)c2c(c3ccccn3n2)C(=O)[CH:1]=[CH:20]C(=O)O",
        #"CCC1(Cc2cc(c(c(c2[C]1=[O])Cl)Cl)OCC(=O)OC)CCC(=O)[CH3]>Cc1ccccc1.CC(=O)O.C1CCNC1>CCC12CCC(=O)[CH]=[C]1c3c(cc(c(c3Cl)Cl)OCC(=O)OC)C2",
        #"CC(=O)C.[OH-].O>>C=C(O)C"

        # Reductive amination
        #"CCCCC[CH:6]=[O:7].c1cc(ccc1C2CCC[NH:24]C2)Oc3ccc(cn3)C(=O)N>[BH4-].[Na+].Cl>CCCCC[CH2:6][N:24]1CCCC(C1)c2ccc(cc2)Oc3ccc(cn3)C(=O)N",
        #"c1cc(cc(c1)O)[CH:8]=[O:7].C1CC[NH:4]CC1>[BH4-].Cl>c1cc(cc(c1)O)[CH2:8][N:4]2CCCCC2",

        # Acetal deprotection
        #"[CH3:1][O:2][CH:3](CN1CC2CC1CO2)[O:12][CH3:13].[ClH:14]>O>C1C2CN(C1CO2)C[CH:3]=[O:2]",
        #"CCn1c(nc(n1)[CH:4]([O:3][CH2:2][CH3:1])[O:5][CH2:6][CH3:7])C>O.Cl>CCn1c(nc(n1)[CH:4]=[O:3])C",
        #"CC(C)CC(CS)C(=O)NC[CH:3]([O:2][CH3:1])[O:15][CH3:16]>CC#N.O.Cl>CC(C)CC(CS)C(=O)NC[CH:3]=[O:2]",

        # N-Boc deprotection
        #"[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[NH:7]C1CCCN(C1=O)C>C(=O)(C(F)(F)F)O.O>CN1CCCC(C1=O)[NH2:7]",

        # Hydrogenation
        #"[C:4](=[C:2](F)F)(C(F)(F)F)F>[HH]>[CH:4]([CH:2](F)F)(C(F)(F)F)F"
        "[CH2:1]=[N+:2]1CCc2c(c(=O)n(c(=O)[nH]2)Cc3ccccc3)C1.[CH:25](=[O:24])[OH:26].[CH:28](=[O:27])[O-:29].[OH2:23].[ClH:21].[ClH:22]>>[CH3:1][N:2]1CCc2c(c(=O)n(c(=O)[nH]2)Cc3ccccc3)C1"
    ]


    results = []
    conditions_are_reactants = True

    for i, full_reaction in enumerate(multistep):
        print(f"Reaction {i+1}/{len(multistep)}")
        if folder_res is not None and os.path.exists(os.path.join(folder_res, f'rxn_{i+1}', 'res.pkl')):
            print("Already computed")
            continue
        print(f"full_reation = \"{full_reaction}\"")
        full_reaction = re.sub(r':\d+\]', ']', full_reaction)
        full_reaction = full_reaction.replace("[HH]", "[H][H]")
        reac, cond, prod = full_reaction.split('>')
        if conditions_are_reactants and cond != '':
            reac = '.'.join(reac.split('.')+cond.split('.'))

        res = search(
            reac,
            prod,
            k_n_tuple=(5, 10),
            max_node_budget=100,
            search_algo="best_first",
            #model_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints_hf_train_debug_20", "checkpoint-159536"), # trained on min_min_1
            #model_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints_hf_train_debug_17", "checkpoint-179478"), # trained on equ_min_n
            #model_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints_hf_train_debug_14", "checkpoint-598260"), # Trained on equ_equ_n
            model_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "checkpoints_hf_train_debug_60", "checkpoint-39910"), # model 17 finetuned with ozonolysis
            tokenizer_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", "tokenizer", "mechsmiles_tokenizer_folder"),
            search_temperature=1.0,
            #verbose=True,
            max_depth=8,
            #save_mech_folder=f"data/figures/paroutes/recognized_tmp{i+1}/",
            #save_mech_folder=f"data/figures/boc_round_trip/rxn_{i+1}/",
            save_mech_folder=os.path.join(folder_res, f"rxn_{i+1}/") if folder_res is not None else None,
            #save_mech_folder=os.path.join(folder_res, f"rxn_of_interest/"),
            #stop_at_best=False,
            #show_images=True,
        )

        if folder_res is not None:
            with open(os.path.join(folder_res, f'rxn_{i+1}', 'res.pkl'), 'wb') as f:
                pickle.dump(res, f)

    if folder_res is not None:
        for i, full_reaction in enumerate(multistep):
            with open(os.path.join(folder_res, f'rxn_{i+1}', 'res.pkl'), 'rb') as f:
                results.append(pickle.load(f))

        df['mechanism'] = results
        df['solved'] = df['mechanism'].apply(lambda x: len(x)>0)
        df.to_csv("data/uspto/USPTO_MIT_sampling_test_w_res_model_60.csv", index=False)

    elif len(res) > 0:
        for r in res:
            from chrimp.visualization.mechanism_visualizer import MechanismVisualizer
            MechanismVisualizer(r).show(max_msmi_in_one_row = 3)



