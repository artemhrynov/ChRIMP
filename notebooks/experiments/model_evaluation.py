# %% Imports
import re
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from colorama import Fore
import cairosvg
import io
import os
from PIL import Image

from chrimp.notebook_helpers.data_evaluation_helpers import aggregate_top_k
from chrimp.world.mechsmiles import MechSmiles


# %%
def compare_two_mechsmiles(
    msmi1, msmi2, title=None, show_plot=True, string_distance=True
):
    """
    Compare two MechSmiles objects by showing them side-by-side.
    """
    svg1 = msmi1.show(return_svg=True)
    svg2 = msmi2.show(return_svg=True)

    # Show them both side-by-side, on two panes of a matplotlib figure
    png_bytes1 = cairosvg.svg2png(bytestring=svg1, scale=4)
    png_bytes2 = cairosvg.svg2png(bytestring=svg2, scale=4)

    # Open PNG bytes as PIL images
    img1 = Image.open(io.BytesIO(png_bytes1))
    img2 = Image.open(io.BytesIO(png_bytes2))

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
    ax1.imshow(img1)
    ax1.axis("off")
    ax1.set_title("Ground Truth MechSMILES")
    ax2.imshow(img2)
    ax2.axis("off")
    ax2.set_title("Predicted MechSMILES")
    plt.tight_layout()
    print(f"MechSMILES:\n{msmi1.value}\n{msmi2.value}")
    print(f"Reactants:\n{msmi1.smiles}\n{msmi2.smiles}")
    plt.figtext(
        0.5,
        0.02,
        f"Lehvenstein between\nMechSMILES: {lev_dist(msmi1.value, msmi2.value)}\nReactants: {lev_dist(msmi1.smiles, msmi2.smiles)}",
        ha="center",
        fontsize=10,
    )
    if title is not None:
        plt.suptitle(title)
    if show_plot:
        plt.show()


# %% Important variables

##model = "model_48_61584"
##model = "model_47_123168"
##model = "model_57_112904"
##model = "model_63_126022"
# model = "model_65_388791"
#
# model = "model_72_412800"
# model = "model_73_1330076"
# model = "model_73_1813740"
# model = "model_73_2418320"
# model = "model_82_125640"
# model = "model_82_157050"
# model = "model_82_219870"
# model = "model_82_345510"
# model = "model_83_133280"
# model = "model_83_346528"
# model = "model_83_399840"
# model = "model_83_453152"
# model = "model_84_229704"
# model = "model_84_267988"
# model = "model_84_306272"
# model = "model_84_344556"
# model = "model_84_382840"
# model = "model_84_421124"
# model = "model_84_650828"
#
##model = "model_85_574590"
###model = "model_85_459672" #equ_equ_n
###model = "model_85_536284" #equ_equ_n
##
##model = "model_86_650624" #equ_equ_n
##model = "model_87_381100" #equ_equ_n
# model = "model_87_533540" #equ_equ_n

# top_k = 20
top_k = 10
# top_k = 5
#
# dataset_name = "pmechdb"
##dataset_name = "mechuspto31k"
#
# if dataset_name == 'pmechdb':
#    full_dataset_name =  "SchwallerGroup/pmechdb_manually_curated_multi_elem_retro"
# elif dataset_name == "mechuspto31k":
#    full_dataset_name =  "SchwallerGroup/uspto-31k_retro"
#
#
# model_type = "llama" # "llama", "t5" or "gpt"
##model_type = "gpt"
# model_type = "t5"
#
##folder = "manu"
##folder = "manu_and_10k_comb"
##folder = "manu_and_30k_comb"
# folder = "hf_train"
#
# grpo  = None # None or int
# grpo_string = f"_grpo_{grpo}" if grpo is not None else ""
#
# results_folder = os.path.join("data", "evaluations", folder, dataset_name, f"{model}{grpo_string}_top{top_k}_summary")
results_folder = None  # Just to observe
#
# if results_folder is None:
#    pass
# elif (not os.path.exists(results_folder) or  not os.path.isdir(results_folder)):
#    os.makedirs(results_folder)
#    os.makedirs(os.path.join(results_folder, "svg"))
# else:
#    print(f"{Fore.RED}Warning: {results_folder} already exists. Results will be overwritten.{Fore.RESET}")
#    ans = input("Do you want to overwrite the folder? (y/[n]): ")
#    while ans.lower() not in ["y", "n", ""]:
#        print("Please answer with 'y' or 'n': ")
#        ans = input("Do you want to overwrite the folder? (y/[n]): ")
#
#    if ans.lower() == "n" or ans == "":
#        print("Exiting without overwriting results.")
#        exit()

# task = "min_min_1"
task = "equ_equ_n"
# task = "equ_min_n"
# task = "spe_min_n"

# model_n = 34
# model_n = 37
# model_n = 36
# model_n = 38

# model_n = 14
# model_n = 72
model_n = 73

#
multiple_length_routes = task.split("_")[2] == "n"
# %% Data load
# df = pd.read_parquet(f"data/evaluations/{folder}/{dataset_name}/{model_type}_eval_{model}{grpo_string}_{task}_top{top_k}.parquet")
# df = pd.read_parquet(f"data/evaluations/model_11/test_10.parquet")
# df = pd.read_parquet(f"data/evaluations/test_fw.parquet")
df = pd.read_parquet(f"data/evaluations/model_{model_n}/test_retro.parquet")


if len(df.columns) == 2 and "0" in df.columns and "1" in df.columns:
    # Some runs have been stored in a weird format, we need to reformat it properly
    all_data = []

    for beam in df["1"]:
        all_data.extend(beam)

    df = pd.DataFrame(all_data)

df.loc[0]
df["input"].values[0]
df["generated_text"].values[0]
df["gold_output"].values[0]
# dataset_add_info = load_dataset(full_dataset_name, split="val")
# df_add_info = dataset_add_info.to_pandas()

# df_add_info = pd.read_parquet(f"data/evaluations/model_11/test_10_questions.parquet")
df_add_info = pd.read_parquet(
    f"data/evaluations/model_{model_n}/test_retro_questions.parquet"
)

# Filter because of truncation
sanity_checked_input = df["input"].unique()
df_add_info = df_add_info[df_add_info["input"].isin(sanity_checked_input)].reset_index(
    drop=True
)

df["beam_index"].value_counts()
df_best = df[df["beam_index"] == 0].reset_index(drop=True)
df_add_info["output"][0]
df_add_info_safecopy = df_add_info.copy()

len(df_best)
len(df_add_info)


# %% Try to save the data if the sizes do not match

# indices = [132595, 132596, 132604]
df_add_info = df_add_info_safecopy.copy()
# for idx in indices:
#    df_add_info = df_add_info.drop(idx)
# df_add_info = df_add_info.reset_index()

df_add_info["output"].values[0].replace(" ", "")
df_best.loc[0]["gold_output"].replace("[eos]", "")

changes_this_loop = True
while changes_this_loop:
    changes_this_loop = False
    for i in range(len(df_add_info)):
        if i >= len(df_best):
            # df_add_info has more rows than df_best, drop the rest
            print(f"Dropping row {i} from df_add_info (exceeds df_best length)")
            df_add_info = df_add_info.drop(i).reset_index(drop=True)
            changes_this_loop = True
            break
        if df_add_info.loc[i]["output"].replace(" ", "") != df_best.loc[i][
            "gold_output"
        ].replace("[eos]", ""):
            print(f"Mismatch at index {i}, dropping from df_add_info")
            df_add_info = df_add_info.drop(i).reset_index(drop=True)
            changes_this_loop = True
            break

# %%

if not len(df) == len(df_add_info) * top_k:
    print(f"{Fore.RED}Dataframes should have the same length{Fore.RESET}")
    print(f"{len(df)=} != {len(df_add_info)*top_k=}")
    print(
        f"Correct beam is most probably {len(df['beam_index'].unique())} == {len(df)/len(df_add_info)}"
    )
    print("If these two numbers are not equal or non-integer, there is an error")
    _ = input("Enter to exit")
    exit()

df["input"].value_counts()

for i in range(len(df_add_info)):
    if df["input"].values[top_k * i] != df_add_info["input"].values[i]:
        print(i)
        print(df["input"].values[top_k * i])
        print(df_add_info["input"].values[i])
        break

df_add_info.columns

# Multiply the column time top_k
df["step_idx_retro"] = [
    df_add_info["metadata"].values[i // top_k]["step_idx_retro"] for i in range(len(df))
]
# assert len(df) == len(df_add_info)*top_k, "Dataframes should have the same length"

df_add_info.columns

# Multiply the column time top_k
df["step_idx_retro"] = [
    df_add_info["metadata"].values[i // top_k]["step_idx_retro"] for i in range(len(df))
]

# df = pd.read_parquet(f"data/evaluations/equ_equ_before_rd_seed/gpt2_eval_last_model_equ_equ_1_top{top_k}.parquet")

print(df.columns)

# Remove lines with problematic gold_output
len_problem_output = len(df) - len(df[df["gold_output"] != "[eos]"])
if len_problem_output > 0:
    print(
        f"{Fore.RED}Careful, had to remove {len_problem_output} outputs because then don't end with [eos]"
    )
df = df[df["gold_output"] != "[eos]"]
print(len(df))
df["ready_for_search"] = (
    (df["product_stable"] == 1) & (df["msmi_reac_correct"] == 1)
).astype(float)

filtered_df = df[(df["ground_truth_equiv"]) & (df["ready_for_search"] != 1)]
filtered_df["ground_truth_equiv"]

for i in range(len(filtered_df)):
    msmi_gt_str = filtered_df["gold_output"].values[i].replace("[eos]", "")
    msmi_pred_str = filtered_df["generated_text"].values[i].replace("[eos]", "")
    msmi_gt = MechSmiles(msmi_gt_str)
    msmi_pred = MechSmiles(msmi_pred_str)
    compare_two_mechsmiles(msmi_gt, msmi_pred, title="Plausible move different from GT")


# %% Main performance

df_prechecked = df[df["ready_for_search"] == 1].copy()

# Retrieve the dataframe of all input-output pairs that didn't even get one pred in the df_prechecked
mask = (
    df.groupby(["input", "gold_output"])["ready_for_search"].transform(
        lambda s: (s != 1).all()
    )  # True if no 1-s in the group
)
df_no_heur_answer = df[mask].reset_index(drop=True)

perc_results = []
perc_results_prechecked = []
num_questions = len(df[["input", "gold_output"]].value_counts())

print(f"{num_questions = }")

for top_k in range(1, top_k + 1):
    res = aggregate_top_k(
        df,
        {"ground_truth_equiv": (lambda x: (x == 1).any())},
        k=top_k,
        filtering_strategy="keep_below_k",
    )
    perc = res / num_questions
    perc_results.append(perc)
    print(f"Aggregate top-{top_k:02.0f}: {perc:.2%}")

    res = aggregate_top_k(
        df_prechecked,
        {"ground_truth_equiv": (lambda x: (x == 1).any())},
        k=top_k,
        filtering_strategy="keep_k_first",
    )
    perc = res / num_questions
    perc_results_prechecked.append(perc)
    print(f"Aggregate top-{top_k:02.0f}: {perc:.2%} (after precheck)")


# %% More details

percentage_usable_for_search = []
percentage_legal_but_not_correct_reac = []
equiv_correct_answer = []
exact_correct_answer = []

for i in range(1, top_k + 1):
    res = aggregate_top_k(
        df,
        conditions={
            "product_stable": (lambda x: (x == 1).any()),
            "msmi_reac_correct": (lambda x: (x < 1).any()),
        },
        k=i,
        filtering_strategy="keep_only_kth",
    )
    percentage_legal_but_not_correct_reac.append(res / num_questions)

    res = aggregate_top_k(
        df,
        conditions={
            "product_stable": (lambda x: (x == 1).any()),
            "msmi_reac_correct": (lambda x: (x == 1).any()),
        },
        k=i,
        filtering_strategy="keep_only_kth",
    )

    percentage_usable_for_search.append(res / num_questions)

    res = aggregate_top_k(
        df,
        conditions={
            "ground_truth_equiv": (lambda x: (x == 1).any()),
        },
        k=i,
        filtering_strategy="keep_only_kth",
    )

    equiv_correct_answer.append(res / num_questions)

    res = aggregate_top_k(
        df,
        conditions={
            "ground_truth_exact": (lambda x: (x == 1).any()),
            # "ground_truth_similar": (lambda x: (x==1).any()),
        },
        k=i,
        filtering_strategy="keep_only_kth",
    )

    exact_correct_answer.append(res / num_questions)

# %% Figure elem. step performance

top_reaching_index = perc_results_prechecked.index(max(perc_results_prechecked))

x_axis = range(1, top_k + 1)

plt.figure()
sns.set_palette("magma", n_colors=2)
sns.lineplot(x=x_axis, y=[x * 100 for x in perc_results], zorder=2)
sns.scatterplot(
    x=x_axis, y=[x * 100 for x in perc_results], label="Normal order", zorder=2
)
sns.lineplot(x=x_axis, y=[x * 100 for x in perc_results_prechecked], zorder=1)
sns.scatterplot(
    x=x_axis,
    y=[x * 100 for x in perc_results_prechecked],
    label=f"Heuristic reorder (among top-{top_k})",
    zorder=1,
)

sns.scatterplot(
    x=[top_reaching_index + 1],
    y=[max(perc_results_prechecked) * 100],
    color=sns.color_palette()[1],
    marker="*",
    s=300,
    label=f"Max reached (@ top-{top_reaching_index+1})",
)  # color="#eec611"
plt.axhline(y=100, ls="--", color="k")
plt.ylim([30, 105])
plt.yticks(range(30, 105, 5))
plt.title("Correct predictions")
plt.ylabel("Percentage equivalent to GT in top-k")
plt.xlabel("Top-k")
plt.xticks(x_axis)
plt.legend(loc="lower right")

if results_folder is not None:
    plt.savefig(os.path.join(results_folder, "svg", "top_k_perf.svg"))
    plt.savefig(os.path.join(results_folder, "top_k_perf.png"))
    plt.clf()
    plt.close()
else:
    plt.show()

# %% Figure elem. step performance (zoom 1)
top_reaching_index = perc_results_prechecked.index(max(perc_results_prechecked))

x_axis = range(1, top_k + 1)

plt.figure()
sns.set_palette("magma", n_colors=2)
sns.lineplot(x=x_axis, y=[x * 100 for x in perc_results], zorder=2)
sns.scatterplot(
    x=x_axis, y=[x * 100 for x in perc_results], label="Normal order", zorder=2
)
sns.lineplot(x=x_axis, y=[x * 100 for x in perc_results_prechecked], zorder=1)
sns.scatterplot(
    x=x_axis,
    y=[x * 100 for x in perc_results_prechecked],
    label=f"Heuristic reorder (among top-{top_k})",
    zorder=1,
)

sns.scatterplot(
    x=[top_reaching_index + 1],
    y=[max(perc_results_prechecked) * 100],
    color=sns.color_palette()[1],
    marker="*",
    s=300,
    label=f"Max reached (@ top-{top_reaching_index+1})",
)  # color="#eec611"
plt.axhline(y=100, ls="--", color="k")
plt.ylim([80, 102])
plt.yticks(range(80, 105, 5))
plt.title("Correct predictions")
plt.ylabel("Percentage equivalent to GT in top-k")
plt.xlabel("Top-k")
plt.xticks(x_axis)
plt.legend(loc="lower right")

if results_folder is not None:
    plt.savefig(os.path.join(results_folder, "svg", "top_k_perf_zoom.svg"))
    plt.savefig(os.path.join(results_folder, "top_k_perf_zoom.png"))
    plt.clf()
    plt.close()
else:
    plt.show()


# %% Plot 2
sns.set_palette("magma", n_colors=3)
sns.lineplot(
    x=x_axis,
    y=[
        (x_i + y_i) * 100
        for x_i, y_i in zip(
            percentage_usable_for_search, percentage_legal_but_not_correct_reac
        )
    ],
    zorder=1,
    ls="--",
)
sns.scatterplot(
    x=x_axis,
    y=[
        (x_i + y_i) * 100
        for x_i, y_i in zip(
            percentage_usable_for_search, percentage_legal_but_not_correct_reac
        )
    ],
    label="Sum of legal MechSMILES",
)

sns.lineplot(x=x_axis, y=[x * 100 for x in percentage_usable_for_search], zorder=1)
sns.scatterplot(
    x=x_axis,
    y=[x * 100 for x in percentage_usable_for_search],
    label="MechSMILES legal + correct reac",
)

sns.lineplot(
    x=x_axis, y=[x * 100 for x in percentage_legal_but_not_correct_reac], zorder=1
)
sns.scatterplot(
    x=x_axis,
    y=[x * 100 for x in percentage_legal_but_not_correct_reac],
    label="MechSMILES legal + incorrect reac",
)

plt.ylabel("Percentage [%]")
plt.xlabel("k-th prediction")
plt.ylim([0, 105])
plt.axhline(y=100, ls="--", color="#333333")
plt.title('"Legal arrow-pushing" predictions')
plt.legend()
plt.xticks(x_axis)

if results_folder is not None:
    plt.savefig(os.path.join(results_folder, "svg", "top_kth_correct_reac.svg"))
    plt.savefig(os.path.join(results_folder, "top_kth_correct_reac.png"))
    plt.clf()
    plt.close()
else:
    plt.show()

# %% Plot 3
sns.set_palette("magma", n_colors=3)

plt.fill_between(
    x_axis,
    [x * 100 for x in exact_correct_answer],
    [x * 100 for x in equiv_correct_answer],
    color="gray",
    alpha=0.3,
    label="Gap between exact and equiv",
)

sns.lineplot(x=x_axis, y=[x * 100 for x in percentage_usable_for_search], zorder=1)
sns.scatterplot(
    x=x_axis,
    y=[x * 100 for x in percentage_usable_for_search],
    label="Correct reac. (stable prod.)",
)

sns.lineplot(x=x_axis, y=[x * 100 for x in equiv_correct_answer], zorder=1)
sns.scatterplot(
    x=x_axis, y=[x * 100 for x in equiv_correct_answer], label="Equiv. to ground-truth"
)

sns.lineplot(x=x_axis, y=[x * 100 for x in exact_correct_answer], zorder=1)
sns.scatterplot(
    x=x_axis, y=[x * 100 for x in exact_correct_answer], label="Exact ground-truth"
)

plt.ylabel("Percentage over val. set [%]")
plt.xlabel("k-th prediction")
plt.ylim([0, 105])
plt.axhline(y=100, ls="--", color="#333333")
plt.title("Stats on k-th predictions")
plt.legend(loc="right")
plt.xticks(x_axis)

if results_folder is not None:
    plt.savefig(
        os.path.join(results_folder, "svg", "top_kth_correct_reac_equiv_exact.svg")
    )
    plt.savefig(os.path.join(results_folder, "top_kth_correct_reac_equiv_exact.png"))
    plt.clf()
    plt.close()
else:
    plt.show()

# %%
df_top1 = df[df["beam_index"] == 0]


df_heuristic_top1 = (
    df_prechecked.sort_values("beam_index")  # ensure it is sorted
    .groupby(["input", "gold_output"], as_index=False, group_keys=False)
    .head(1)  # keep the k smallest per group
)

df_heuristic_top1_or_top1 = (  # 1st heuristic or top-1 is no answer in top-k passes the filters
    df.sort_values(
        [
            "ready_for_search",
            "beam_index",
        ],  # ensure it is sorted (But ready for search first)
        ascending=[False, True],  # False for ready_for_search to sort 1 before 0
    )
    .groupby(["input", "gold_output"], as_index=False, group_keys=False)
    .head(1)  # keep the k smallest per group
)

n = 3

df_heuristic_topn_or_topn = (  # n first heuristic or top-n is no answer in top-k passes the filters
    df.sort_values(
        [
            "ready_for_search",
            "beam_index",
        ],  # ensure it is sorted (But ready for search first)
        ascending=[False, True],  # False for ready_for_search to sort 1 before 0
    )
    .groupby(["input", "gold_output"], as_index=False, group_keys=False)
    .head(n)  # keep the k smallest per group
)

print(df_heuristic_top1)
df_heuristic_top1_correct = df_heuristic_top1[
    df_heuristic_top1["ground_truth_equiv"] == 1
]
df_heuristic_top1_incorrect = df_heuristic_top1[
    df_heuristic_top1["ground_truth_equiv"] != 1
]
df_no_heur_top1 = df_no_heur_answer[df_no_heur_answer["beam_index"] == 0]

# %% Look at the successrate of the heuristic top-1 predictions as a function of the step index
if multiple_length_routes:
    distance_to_goal_success_rate = dict()
    distance_to_goal_success_rate_topn = dict()
    distance_to_goal_success_rate_heur_or_top = dict()
    distance_to_goal_success_rate_heur_or_topn = dict()
    distance_to_goal_len = dict()

    for i in range(0, int(df_top1["step_idx_retro"].max())):
        print(i)
        df_top1_step = df_top1[df_top1["step_idx_retro"] == i]

        distance_to_goal_len[i] = len(df_top1_step)

        correct_fraction = df_top1_step["ground_truth_equiv"].mean()
        correct_fraction_topn = aggregate_top_k(
            df[df["step_idx_retro"] == i],
            conditions={
                "ground_truth_equiv": (lambda x: (x == 1).any()),
            },
            k=n,
            filtering_strategy="keep_below_k",
        ) / len(df_top1_step)
        distance_to_goal_success_rate[i] = correct_fraction
        distance_to_goal_success_rate_topn[i] = correct_fraction_topn

    for i in range(0, int(df_heuristic_top1_or_top1["step_idx_retro"].max())):
        print(i)
        df_heuristic_top1_or_top1_step = df_heuristic_top1_or_top1[
            df_heuristic_top1_or_top1["step_idx_retro"] == i
        ]
        df_heuristic_topn_or_topn_step = df_heuristic_topn_or_topn[
            df_heuristic_topn_or_topn["step_idx_retro"] == i
        ]

        correct_fraction = df_heuristic_top1_or_top1_step["ground_truth_equiv"].mean()
        correct_fraction_topn = aggregate_top_k(
            df_heuristic_topn_or_topn_step,
            conditions={
                "ground_truth_equiv": (lambda x: (x == 1).any()),
            },
            k=n,
            filtering_strategy="keep_k_first",
        ) / len(df_heuristic_top1_or_top1_step)
        print(f"{correct_fraction_topn = }")
        distance_to_goal_success_rate_heur_or_top[i] = correct_fraction
        distance_to_goal_success_rate_heur_or_topn[i] = correct_fraction_topn

    sns.histplot(df_top1, x="step_idx_retro")
    plt.yscale("log")

    if results_folder is not None:
        plt.savefig(
            os.path.join(results_folder, "svg", "distribution_distance_from_prod.svg")
        )
        plt.savefig(os.path.join(results_folder, "distribution_distance_from_prod.png"))
        plt.clf()
        plt.close()
    else:
        plt.show()

# %%
# Here double graph:
# x-axis is the step_idx_retro
# Left y-axis is a scatterplot of success_fraction
# Right y-axis is a barplot of the number of questions

if multiple_length_routes:
    # Prepare data for plotting
    steps = list(distance_to_goal_success_rate.keys())
    success_rates_top1 = [s * 100 for s in distance_to_goal_success_rate.values()]
    success_rates_topn = [s * 100 for s in distance_to_goal_success_rate_topn.values()]
    success_rates_heur_or_top1 = [
        s * 100 for s in distance_to_goal_success_rate_heur_or_top.values()
    ]
    success_rates_heur_or_topn = [
        s * 100 for s in distance_to_goal_success_rate_heur_or_topn.values()
    ]
    counts = list(distance_to_goal_len.values())

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xticks(steps, [str(x + 1) for x in steps])  # More correct human index
    ax2 = ax1.twinx()

    # Choose colors from the magma palette
    palette = sns.color_palette("magma", as_cmap=True)
    bar_color = sns.color_palette("magma")[0]
    scatter_color = sns.color_palette("magma")[4]
    scatter_color2 = sns.color_palette("magma")[3]
    lim = 10

    # Barplot (right y-axis)
    sns.barplot(x=steps[:lim], y=counts[:lim], ax=ax1, color=bar_color)
    ax1.set_ylabel("Number of Questions tested", color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)

    # Scatterplot (left y-axis)
    sns.lineplot(
        x=steps[:lim],
        y=success_rates_heur_or_topn[:lim],
        ax=ax2,
        ls="-",
        marker="o",
        color=scatter_color,
        label=f"Top-{n}",
    )
    sns.lineplot(
        x=steps[:lim],
        y=success_rates_heur_or_top1[:lim],
        ax=ax2,
        ls="-",
        marker="o",
        color=scatter_color2,
        label="Top-1",
    )

    ax2.set_ylabel("Success Rate [%]", color=scatter_color)
    ax2.axhline(y=100, color=scatter_color, ls="--")
    ax2.set_ylim([0, 105])
    ax2.tick_params(axis="y", labelcolor=scatter_color)
    ax1.set_xlabel("Distance to Goal (# Elem. steps)")

    ## Move ax1 (success rate) to the right
    # ax1.yaxis.set_label_position("right")
    # ax1.yaxis.tick_right()

    ## Move ax2 (barplot, counts) to the left
    # ax2.yaxis.set_label_position("left")
    # ax2.yaxis.tick_left()

    plt.legend(loc="right")
    plt.title("Success Rate and Number of Questions by Step Index")
    plt.tight_layout()

    if results_folder is not None:
        plt.savefig(
            os.path.join(
                results_folder,
                "svg",
                "success_rate_vs_distance_to_goal_heur_or_top_n.svg",
            )
        )
        plt.savefig(
            os.path.join(
                results_folder, "success_rate_vs_distance_to_goal_heur_or_topn.png"
            )
        )
        plt.clf()
        plt.close()
    else:
        plt.show()


# %%
# Rate the retrieval @k of the ground-truth path

df_add_info["rxn_idx"] = df_add_info["metadata"].apply(lambda x: x["rxn_idx"])
df_add_info["rxn_length"] = df_add_info["metadata"].apply(lambda x: x["rxn_length"])
df_add_info["retro_step_idx"] = df_add_info["metadata"].apply(
    lambda x: x["step_idx_retro"]
)

df_add_info
all_test_indices = df_add_info["rxn_idx"].unique()
all_test_length = df_add_info["rxn_length"].unique()
all_test_length.sort()


if not os.path.isfile(f"long_computation_data_model_{model_n}.pkl"):
    num_rxns_per_length = np.zeros(max(all_test_length))
    num_success_rxns_per_length = [[] for _ in range(max(all_test_length))]

    total_num_rxns = 0
    total_success_rxns = []

    if multiple_length_routes:
        for index in tqdm(all_test_indices[:]):
            sub_df = df_add_info[df_add_info["rxn_idx"] == index]
            if not len(sub_df) == sub_df["rxn_length"].values[0]:
                print(f"{Fore.RED}ERROR{Fore.RESET}")
                continue
            length = len(sub_df)
            num_rxns_per_length[length - 1] += 1  # 0-index is length 1
            total_num_rxns += 1
            successful = True
            min_beam_width_necessary = 0

            for input in sub_df["input"].values:
                # print(input)
                df_beam = df[df["input"] == input]
                df_beam_equiv_gt = df_beam[df_beam["ground_truth_equiv"] == 1]
                if len(df_beam_equiv_gt) == 0:
                    successful = False
                    break
                else:
                    min_index = df_beam_equiv_gt["beam_index"].min()
                    min_beam_width = min_index + 1
                    min_beam_width_necessary = max(
                        min_beam_width_necessary, min_beam_width
                    )

            if successful:
                num_success_rxns_per_length[length - 1].append(
                    int(min_beam_width_necessary)
                )
                total_success_rxns.append(int(min_beam_width_necessary))

    long_computation_data = (
        num_rxns_per_length,
        num_success_rxns_per_length,
        total_num_rxns,
        total_success_rxns,
    )

    with open(f"long_computation_data_model_{model_n}.pkl", "wb") as f:
        pickle.dump(long_computation_data, f)

else:
    with open(f"long_computation_data_model_{model_n}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
        (
            num_rxns_per_length,
            num_success_rxns_per_length,
            total_num_rxns,
            total_success_rxns,
        ) = loaded_data


# safe_num_rxns_per_length = num_rxns_per_length.copy()
# safe_num_success_rxns_per_length = num_success_rxns_per_length.copy()
# safe_total_num_rxns = total_num_rxns
# safe_total_success_rxns = total_success_rxns.copy()


# %%
## Look at why it is so low for top ones
# sub_df_length_1 = df_add_info[df_add_info['rxn_length']==1]
# inputs_rxns_length_1 = sub_df_length_1['input'].unique()
# filtered_df = df[(df['beam_index'] == 0) & (df['input'].isin(inputs_rxns_length_1))]
##filtered_df = df[(df['input'].isin(inputs_rxns_length_1))]
#
# for index in range(23):
#    if not filtered_df['generated_text'].values[index] == filtered_df['gold_output'].values[index] + '[eos]':
#        print(f"{index = }")
#        if filtered_df['ground_truth_equiv'].values[index]:
#            print("different but equivalent")
#        else:
#            print(filtered_df['generated_text'].values[index][:-5])
#            print(filtered_df['gold_output'].values[index])
#
#
# df_add_info[df_add_info['output'].apply(lambda x: x.startswith("BrC1=CC=C2C(=C1)OCCC1=C2N=C([N:1]=[N+:2]=[N-:3])S1.C1CC1[C:4]#[CH:5]"))]['input'].values


# %% Plot
thresholds = list(set(total_success_rxns))
thresholds.sort(reverse=True)
print(thresholds)

# Max interesting length (aggr. what surpasses)
max_interesting_length = 7

aggr_num_rxns_per_length = np.append(
    num_rxns_per_length[:max_interesting_length],
    np.sum(num_rxns_per_length[max_interesting_length:]),
)

aggr_num_success_rxns_per_length = num_success_rxns_per_length[
    :max_interesting_length
].copy() + [sum(num_success_rxns_per_length[max_interesting_length:], [])]

# Prepare data for plotting
data = []
for threshold in thresholds:
    for length_idx, (total, success_list) in enumerate(
        zip(aggr_num_rxns_per_length, aggr_num_success_rxns_per_length)
    ):
        length = length_idx + 1  # Length starts at 1

        if total > 0:  # Avoid division by zero
            count = len([x for x in success_list if x <= threshold])
            percentage = (count / total) * 100
            data.append(
                {
                    "Length": length,
                    "Percentage": percentage,
                    "Threshold": threshold,
                }
            )

# Create DataFrame
df_plot = pd.DataFrame(data)

# Create the plot
# sns.set_palette('magma', n_colors=max_interesting_length+1)
palette = sns.color_palette("magma_r", n_colors=max_interesting_length + 1)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_plot,
    x="Length",
    y="Percentage",
    hue="Threshold",
    marker="o",
    palette=palette,
)

plt.xlabel("Length of reaction (elem. steps)", fontsize=12)
plt.ylabel("Percentage of Successes [%]", fontsize=12)
plt.title("Retrieval@k of the ground-truth sequence of elem. steps", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title="Beam width", fontsize=10)
plt.tight_layout()
plt.xlim(0.5, max_interesting_length + 1.5)
plt.xticks(
    range(1, len(aggr_num_rxns_per_length) + 1),
    labels=[
        str(x) + ("+" if x > max_interesting_length else "")
        for x in range(1, len(aggr_num_rxns_per_length) + 1)
    ],
)
plt.axhline(ls="--", y=100)
plt.ylim(90, 101)
plt.show()

# %% Plot Global
thresholds = range(1, 11)

# Prepare data for plotting
chrimp_t5_data = []
for threshold in thresholds:
    total, success_list = total_num_rxns, total_success_rxns
    if total > 0:  # Avoid division by zero
        count = len([x for x in success_list if x <= threshold])
        percentage = (count / total) * 100
        chrimp_t5_data.append(round(percentage, 2))

# Create DataFrame
df_plot = pd.DataFrame(data)
data_flower_paper = {
    "Top-k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "FlowER-large": [92.5, 96.89, 98.15, 98.45, 98.61, 98.65, 98.68, 98.69, 98.7, 98.7],
    "FlowER": [88.97, 94.75, 96.48, 97.11, 97.39, 97.57, 97.65, 97.68, 97.69, 97.7],
    #'G2S': [92.51, 95.94, 97.03, 97.49, 97.76, 97.88, 98.0, 98.08, 98.13, 98.19],
    #'G2S+H': [89.22, 93.7, 95.1, 95.79, 96.19, 96.45, 96.66, 96.79, 96.89, 96.98],
    #'MT': [88.31, 95.64, 97.02, 97.41, 97.59, 97.59, 97.59, 97.59, 97.59, 97.59],
    "ChRIMP T5 (ours)": chrimp_t5_data,
}

print(chrimp_t5_data)


# %% Comparison FlowER
sns.set_palette("magma", n_colors=len(data_flower_paper.keys()))

df_flower_paper = pd.DataFrame(data_flower_paper)
df_flower_paper

df_melted = df_flower_paper.melt(
    id_vars="Top-k", var_name="Method", value_name="Accuracy"
)
df_melted = df_melted[df_melted["Top-k"] <= 5]

# Create the plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_melted, x="Top-k", y="Accuracy", hue="Method", marker="o")

# sns.lineplot(data=df_plot, x='Beam size', y='Percentage', marker='o')

plt.xlabel("Beam width (k)", fontsize=12)
plt.ylabel("Percentage of Successes [%]", fontsize=12)
plt.title("Retrieval@k of the ground-truth sequence of elem. steps", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0.5, 5.5)
plt.ylim(85, 100)  # Same as FlowER paper
plt.yticks(range(85, 101, 5))
plt.savefig("comparison_flower_paper.svg")
plt.show()

# %% Show results on mech-USPTO-31k
data_mech_uspto_31k_paper = {
    "Top-k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "ChRIMP T5 (ours)": chrimp_t5_data[:10],
}

sns.set_palette("magma", n_colors=len(data_mech_uspto_31k_paper.keys()))

data_mech_uspto_31k_paper = pd.DataFrame(data_mech_uspto_31k_paper)

df_melted = data_mech_uspto_31k_paper.melt(
    id_vars="Top-k", var_name="Method", value_name="Accuracy"
)
df_melted = df_melted[df_melted["Top-k"] <= 10]

# Create the plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_melted, x="Top-k", y="Accuracy", hue="Method", marker="o")

plt.xlabel("Beam width (k)", fontsize=12)
plt.ylabel("Percentage of Successes [%]", fontsize=12)
plt.title("Retrieval@k of the ground-truth sequence of elem. steps", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0.5, 10.5)
plt.ylim(70, 100)  # Same as reactron paper
plt.yticks(range(70, 105, 5))
plt.savefig("results_mech_uspto_31k_paper.svg")
plt.show()
# %% Comparison Reactron
data_reactron_paper = {
    "Top-k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #'Reactron': [94.8, 95.8, 95.9, None, None, None, None, None, None, None],
    #'Transformer M': [56.7, 56.7, 56.7, None, None, None, None, None, None, None],
    #'Graph2SMILES M': [43.5, 43.5, 43.5, None, None, None, None, None, None, None],
    "ChRIMP T5 (ours)": chrimp_t5_data[:10],
}

sns.set_palette("magma", n_colors=len(data_reactron_paper.keys()))

df_reactron_paper = pd.DataFrame(data_reactron_paper)

df_melted = df_reactron_paper.melt(
    id_vars="Top-k", var_name="Method", value_name="Accuracy"
)
df_melted = df_melted[df_melted["Top-k"] <= 10]

# Create the plot
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_melted, x="Top-k", y="Accuracy", hue="Method", marker="o")

# sns.lineplot(data=df_plot, x='Beam size', y='Percentage', marker='o')
plt.axhline(
    y=95.9,
    color=sns.color_palette("magma", n_colors=4)[0],
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    xmin=0.2,
    xmax=1.0,
)
plt.axhline(
    y=56.7,
    color=sns.color_palette("magma", n_colors=4)[1],
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    xmin=0.2,
    xmax=1.0,
)
plt.axhline(
    y=43.5,
    color=sns.color_palette("magma", n_colors=4)[2],
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    xmin=0.2,
    xmax=1.0,
)

plt.xlabel("Beam width (k)", fontsize=12)
plt.ylabel("Percentage of Successes [%]", fontsize=12)
plt.title("Retrieval@k of the ground-truth sequence of elem. steps", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xlim(0.5, 10.5)
plt.ylim(0, 100)  # Same as reactron paper
plt.yticks(range(0, 110, 10))
plt.savefig("comparison_reactron_paper.svg")
plt.show()

# %%
if multiple_length_routes:
    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Choose colors from the magma palette
    palette = sns.color_palette("magma", as_cmap=True)
    bar_color = sns.color_palette("magma")[0]
    scatter_color = sns.color_palette("magma")[4]
    scatter_color2 = sns.color_palette("magma")[3]

    # Barplot (right y-axis)
    sns.barplot(x=steps, y=counts, ax=ax1, color=bar_color)
    ax1.set_ylabel("Number of Questions in validation set", color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)

    # Scatterplot (left y-axis)
    sns.lineplot(
        x=steps,
        y=success_rates_topn,
        ax=ax2,
        marker="o",
        color=scatter_color2,
        ls="--",
        label=f"top-{n}",
    )

    sns.lineplot(
        x=steps,
        y=success_rates_top1,
        ax=ax2,
        marker="o",
        color=scatter_color,
        ls="--",
        label="top-1",
    )

    ax2.set_ylabel("Success Rate [%]", color=scatter_color)
    ax2.axhline(y=0, color="#333333", ls="--")
    ax2.set_ylim([0, 105])
    ax2.tick_params(axis="y", labelcolor=scatter_color)
    ax1.set_xlabel("Distance to Goal (# Elem. steps)")

    # Move ax1 (success rate) to the right
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    # Move ax2 (barplot, counts) to the left
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()

    plt.legend(loc="right")
    plt.title("Success Rate and Number of Questions by Step Index")
    plt.tight_layout()

    if results_folder is not None:
        plt.savefig(
            os.path.join(
                results_folder, "svg", "success_rate_vs_distance_to_goal_topn.svg"
            )
        )
        plt.savefig(
            os.path.join(results_folder, "success_rate_vs_distance_to_goal_topn.png")
        )
        plt.clf()
        plt.close()
    else:
        plt.show()


# %% Plot distribution of failure modes for top-1 predictions
sns.set_palette("magma", n_colors=6)
dict_results = {
    "Equiv. to ground-truth": len(df_heuristic_top1_correct) / num_questions,
    "Incorrect arrow-flow": len(df_heuristic_top1_incorrect) / num_questions,
    # "No heur df": len(df_no_heur_top1) / num_questions,
}

# Assume these are the proportions for each subcategory
df_no_heur_illegal_smiles = df_no_heur_top1[(df_no_heur_top1["msmi_reac_legal"] != 1)]
df_no_heur_top1_legal_reac = df_no_heur_top1[(df_no_heur_top1["msmi_reac_legal"] == 1)]

df_no_heur_nothing = df_no_heur_top1_legal_reac[
    (df_no_heur_top1_legal_reac["msmi_reac_correct"] != 1)
    & (df_no_heur_top1_legal_reac["product_stable"] == 0)
]
df_no_heur_not_correct_reac = df_no_heur_top1_legal_reac[
    (df_no_heur_top1_legal_reac["msmi_reac_correct"] != 1)
    & (df_no_heur_top1_legal_reac["product_stable"] == 1)
]
df_no_heur_prod_arrow_push_not_legal = df_no_heur_top1_legal_reac[
    (df_no_heur_top1_legal_reac["msmi_reac_correct"] == 1)
    & (df_no_heur_top1_legal_reac["product_stable"] == 0)
]

no_heur_a, label_a = (
    len(df_no_heur_not_correct_reac) / num_questions,
    "Incorrect reac. (stable prod.)",
)
no_heur_b, label_b = len(df_no_heur_illegal_smiles) / num_questions, "Illegal SMILES"
no_heur_c, label_c = (
    len(df_no_heur_nothing) / num_questions,
    "Incorrect reac. (unstable prod.)",
)
no_heur_d, label_d = (
    len(df_no_heur_prod_arrow_push_not_legal) / num_questions,
    "Correct reac. with illegal arrow-flow",
)


# Set up plot
labels = list(dict_results.keys()) + [f"No heuristics in top-{top_k}"]
values = list(dict_results.values())
x = np.arange(len(labels))

fig, ax = plt.subplots()
fig.set_size_inches(5, 10)

# Plot the first two bars
bars1 = ax.bar(x[0], values[0], label=labels[0])
bars2 = ax.bar(x[1], values[1], label=labels[1])

# Plot the stacked bar
bars3 = ax.bar(x[2], no_heur_a, label=label_a)
bars4 = ax.bar(x[2], no_heur_b, bottom=no_heur_a, label=label_b)
bars5 = ax.bar(x[2], no_heur_c, bottom=no_heur_a + no_heur_b, label=label_c)
bars6 = ax.bar(x[2], no_heur_d, bottom=no_heur_a + no_heur_b + no_heur_c, label=label_d)

for i, v in enumerate(values):
    ax.text(x[i], v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=16)

# Add percentage labels on stacked bar
heights = [no_heur_a, no_heur_b, no_heur_c, no_heur_d]
bottoms = [0, no_heur_a, no_heur_a + no_heur_b, no_heur_a + no_heur_b + no_heur_c]
categories = ["No heur A", "No heur B", "No heur C", "No heur D"]
for i, (h, b, cat) in enumerate(zip(heights, bottoms, categories)):
    ax.text(
        x[2],
        b + h / 2,
        f"{h:.2%}" if h >= 0.0005 else "",
        ha="center",
        va="center",
        fontsize=16,
    )


ax.set_title(f"Heuristic top-1 predictions among top-{top_k}")
ax.set_ylabel("Percentage of predictions")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
plt.tight_layout()

if results_folder is not None:
    plt.savefig(os.path.join(results_folder, "svg", "heuristic_top1_performance.svg"))
    plt.savefig(os.path.join(results_folder, "heuristic_top1_performance.png"))
    plt.clf()
    plt.close()
else:
    plt.show()


if results_folder is not None:
    exit()

# %% Show some examples
from Levenshtein import distance as lev_dist

# %% Now inspecting the part that couldn't find any pred suitable for search among the top-k
tuples_gt_pred = df_no_heur_not_correct_reac[["gold_output", "generated_text"]].values
tuples_gt_pred = [
    (re.sub(r"\[eos\]", "", a), re.sub(r"\[eos\]", "", b)) for a, b in tuples_gt_pred
]

for i in range(20):
    print()
    print(
        f"{i = }, distance_to_goal = {df_no_heur_not_correct_reac['step_idx_retro'].values[i]+1}"
    )
    print(df_no_heur_not_correct_reac["input"].values[i])
    msmi_gt = MechSmiles(tuples_gt_pred[i][0])
    msmi_pred = MechSmiles(tuples_gt_pred[i][1])

    compare_two_mechsmiles(msmi_gt, msmi_pred, title="Wrong reactant")


# %%
tuples_gt_pred = df_heuristic_top1_incorrect[["gold_output", "generated_text"]].values


tuples_gt_pred = [
    (re.sub(r"\[eos\]", "", a), re.sub(r"\[eos\]", "", b)) for a, b in tuples_gt_pred
]

df_heuristic_top1_incorrect.columns

for i in range(20):
    print()
    print(
        f"{i = }, distance_to_goal = {df_heuristic_top1_incorrect['step_idx_retro'].values[i]+1}"
    )
    print(df_heuristic_top1_incorrect["input"].values[i])
    print(tuples_gt_pred[i][0])
    print(tuples_gt_pred[i][1])
    msmi_gt = MechSmiles(tuples_gt_pred[i][0])
    msmi_pred = MechSmiles(tuples_gt_pred[i][1])
    compare_two_mechsmiles(msmi_gt, msmi_pred, title="Plausible move different from GT")


# %% df_illegal_arrows

df_no_heur_prod_arrow_push_not_legal["generated_text"].values[:3]


# %% Inpect some reactions

index_rxn = df_add_info[(df_add_info["rxn_length"] == 4)]["rxn_idx"].values[30]
sub_df = df_add_info[df_add_info["rxn_idx"] == index_rxn]
len(sub_df)

print(
    f'"{sub_df["metadata"].values[0]["elem_reac_equ"]}>>{sub_df["metadata"].values[-1]["elem_prod_equ"]}"'
)

print(
    sub_df["metadata"].values[0]["elem_reac_equ"]
    + ">>"
    + ">>".join(f"{x['elem_prod_equ']}" for x in sub_df["metadata"].values)
)
