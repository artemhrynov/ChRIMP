# Acknoledgments:
# Pattern adapted from Schwaller et al. https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00576
# Code inspired from Hogru (Stephan Holzgruber) https://github.com/huggingface/transformers/issues/17862

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, Sequence, WhitespaceSplit
from transformers import PreTrainedTokenizerFast
import os

current_folder = os.path.dirname(os.path.abspath(__file__))


def create_tokenizer():
    VOCAB_PATH = os.path.join(current_folder, "vocabulary_smiles.txt")

    with open(VOCAB_PATH) as f:  # keep file order → index
        base_tokens = [tok.rstrip("\n") for tok in f]

    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, CLS_TOKEN, SEP_TOKEN = (
        "[pad]",
        "[unk]",
        "[bos]",
        "[eos]",
        "[cls]",
        "[sep]",
    )

    vocab = {tok: idx for idx, tok in enumerate(base_tokens)}

    SMI_REGEX_PATTERN = r"""(|\[[A-Z][a-z]?|\]|Br?|Cl?|N|O|S|P|F|I|H|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"""

    model = WordLevel(vocab=vocab, unk_token=UNK_TOKEN)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Sequence(
        [
            WhitespaceSplit(),
            Split(pattern=Regex(SMI_REGEX_PATTERN), behavior="isolated"),
        ]
    )

    tokenizer.save(os.path.join(current_folder, "smiles_tokenizer.json"))

    # tokenizer = Tokenizer.from_file("smiles_tokenizer.json")  # to get it back

    # Not sure if I will need this later
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
    )

    hf_tok.save_pretrained(os.path.join(current_folder, "smiles_tokenizer_folder"))
    hf_tok.push_to_hub(
        "SchwallerGroup/SMILES_tokenizer", private=True, token=os.getenv("HF_TOKEN")
    )


def check_tokenization_is_same(file_path, tokenizer, hf_pretrained=False):
    import re
    from tqdm import tqdm

    with open(file_path, "r") as f_read:
        all_lines = [re.sub("\n", "", line) for line in f_read.readlines()]

    n_success = 0

    for line in tqdm(all_lines):
        line_wo_spaces = re.sub(" ", "", line)

        if hf_pretrained:
            ids = hf_tok.encode(line_wo_spaces)
            retok_line = hf_tok.decode(ids)

        else:
            ids = tokenizer.encode(line_wo_spaces).ids
            retok_line = tokenizer.decode(ids)

        if line == retok_line:
            n_success += 1

    len_lines = len(all_lines)

    print(
        f"{Fore.LIGHTGREEN_EX if n_success == len_lines else Fore.LIGHTRED_EX}{n_success}/{len_lines} ({n_success/len_lines:.2%}) sucess in reconstruction{Fore.RESET}"
    )


if __name__ == "__main__":
    import re
    from colorama import Fore

    create_tokenizer()
    exit()

    examples_msmi = [
        "C1[CH:1]=[CH:2][CH:3]=[CH:4]1.[CH2:5]=[CH2:6]|((6,5),1);((1,2),3);((3,4),6)",
        "[H]C([H])([H])C(=O)[O:2][H:1]|((1, 2), 2)",
    ]

    tokenizer = Tokenizer.from_file(
        os.path.join(current_folder, "smiles_tokenizer.json")
    )
    hf_tok = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(current_folder, "smiles_tokenizer_folder")
    )

    check_tokenization_is_same(
        "data/pmechdb/manually_curated_train_equ_equ_decoderonly_max_512.txt", tokenizer
    )
    check_tokenization_is_same(
        "data/pmechdb/combinatorial_train_equ_equ_decoderonly_max_512.txt", tokenizer
    )

    check_tokenization_is_same(
        "data/pmechdb/manually_curated_train_equ_equ_decoderonly_max_512.txt",
        hf_tok,
        hf_pretrained=True,
    )
    check_tokenization_is_same(
        "data/pmechdb/combinatorial_train_equ_equ_decoderonly_max_512.txt",
        hf_tok,
        hf_pretrained=True,
    )

    for msmi in examples_msmi:
        # If Tokenizer
        ids = tokenizer.encode(msmi).ids
        text = tokenizer.decode(ids)

        print(f"Assessing {msmi}")

        if re.sub(" ", "", msmi) == re.sub(" ", "", text):
            print(
                f"{Fore.LIGHTGREEN_EX}Tokenization back and forth worked!{Fore.RESET}"
            )
        else:
            print(
                f"{Fore.LIGHTRED_EX}Tokenization back and forth did not work!{Fore.RESET}"
            )

            print(f"{msmi = }")
            print(f"{ids = }")
            print(f"{text = }")

        # If PreTrainedTokenizerFast

        ids = hf_tok.encode(msmi)
        if re.sub(" ", "", msmi) == re.sub(" ", "", hf_tok.decode(ids)):
            print(
                f"{Fore.LIGHTGREEN_EX}HF Tokenization back and forth worked!{Fore.RESET}"
            )
        else:
            print(
                f"{Fore.LIGHTRED_EX}HF Tokenization back and forth did not work!{Fore.RESET}"
            )

            print(f"{msmi = }")
            print(f"{ids = }")
            print(f"{hf_tok.decode(ids) = }")
