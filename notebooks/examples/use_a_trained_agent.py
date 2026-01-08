# %% [markdown]
# # Use a trained agent

# %% Imports
from chrimp.agent.t5_agent import T5Agent
from chrimp.visualization.mechanism_visualizer import MechanismVisualizer

# %% [markdown]
# First, let's load an agent. For this example we'll take the T5 model trained on the reaction without by-products task (equ_min_n), on the FlowER dataset, and finetuned on suzukis and ozonolyses

# %% Load an agent
agent = T5Agent(inference_input_format = "retro") # "forward", "retro" or "reac"
agent.load_model(
    tokenizer_path="SchwallerGroup/MechSMILES_tokenizer",
    model_path="SchwallerGroup/chrimp_T5_equ_min_n_flower_ft_ozsu"
)

# %% Load a reaction we want to seach a mechanism for
#rxn = "CN.CC(=O)Cl>>CNC(=O)C"
rxn = "C=CCOCC(NC(=O)C(CC1=CC=CC=C1)N1C(=O)C2=CC=CC=C2C1=O)C(=O)O.CO.CSC.ClCCl.O.O=[O+][O-]>>O=CCOCC(NC(=O)C(CC1=CC=CC=C1)N1C(=O)C2=CC=CC=C2C1=O)C(=O)O"
msmi_list = agent.search_mech(rxn, max_node_budget=100)

# If we have at least one answer, look at the model preferred one
if len(msmi_list) > 0:
    MechanismVisualizer(msmi_list[0]).show(max_msmi_in_one_row = 3)

