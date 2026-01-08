import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"

# Data structure: dictionary with model names as keys
# Each model has results for test sets Ozonolysis (Ozon.) and Suzuki (Suzu.)

data = {
    "Base": {
        "Test Ozonolysis": {
            "Ground-truth": 0,
            "Good alternative": 0,
            "Bad alternative": 0,
            "Not solved": 5,
        },
        "Test Suzuki": {
            "Ground-truth": 0,
            "Good alternative": 1,
            "Bad alternative": 4,
            "Not solved": 3,
        },
    },
    "Finetuned Ozonolysis": {
        "Test Ozonolysis": {
            "Ground-truth": 3,
            "Good alternative": 1,
            "Bad alternative": 0,
            "Not solved": 1,
        },
        "Test Suzuki": {
            "Ground-truth": 0,
            "Good alternative": 2,
            "Bad alternative": 3,
            "Not solved": 3,
        },
    },
    "Finetuned Suzuki": {
        "Test Ozonolysis": {
            "Ground-truth": 0,
            "Good alternative": 0,
            "Bad alternative": 0,
            "Not solved": 5,
        },
        "Test Suzuki": {
            "Ground-truth": 4,
            "Good alternative": 0,
            "Bad alternative": 0,
            "Not solved": 4,
        },
    },
    "Finetuned Ozon. + Suzu.": {
        "Test Ozonolysis": {
            "Ground-truth": 3,
            "Good alternative": 0,
            "Bad alternative": 0,
            "Not solved": 2,
        },
        "Test Suzuki": {
            "Ground-truth": 4,
            "Good alternative": 0,
            "Bad alternative": 0,
            "Not solved": 4,
        },
    },
}

# Define colors for each category
colors = {
    "Ground-truth": "#2d5016",  # Dark green
    "Good alternative": "#7cb342",  # Light green
    "Bad alternative": "#efb350",  # Light yellow
    "Not solved": "#424242",  # Dark gray/black
}

# Categories in order (bottom to top of stack)
categories = ["Not solved", "Bad alternative", "Good alternative", "Ground-truth"]

# Models in order
models = ["Base", "Finetuned Ozonolysis", "Finetuned Suzuki", "Finetuned Ozon. + Suzu."]

# Create figure with two subplots (now stacked vertically)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))


# Function to create horizontal stacked bars
def create_stacked_bars(ax, test_set_name):
    y_pos = np.arange(len(models))
    height = 0.6

    # Create stacked bars
    left = np.zeros(len(models))

    for category in categories:
        values = [data[model][test_set_name][category] for model in models]
        ax.barh(
            y_pos,
            values,
            height,
            label=category,
            left=left,
            color=colors[category],
            edgecolor="white",
            linewidth=1.5,
        )
        left += values

    # Customize the subplot
    ax.set_xlabel("Count", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.set_title(f"{test_set_name}", fontsize=12, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid for readability
    ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


# Create plots for both test sets
create_stacked_bars(axes[0], "Test Ozonolysis")
create_stacked_bars(axes[1], "Test Suzuki")

# Add a single legend for the entire figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=False,
    fontsize=10,
)

# Add overall title
fig.suptitle(
    "Model Performance on both Test Sets", fontsize=14, fontweight="bold", y=0.98
)

# Adjust layout to prevent overlap and make room for legend
plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Leave space at bottom for legend

# plt.show()

# Save the figure
plt.savefig("model_performance_comparison.svg", dpi=300, bbox_inches="tight")
plt.savefig("model_performance_comparison.pdf", bbox_inches="tight")
print("Figure saved successfully!")
