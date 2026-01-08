"""
Helper to convert PMechDB's format to MechSMILES
The conventions go as follow:

| Convention    | ionization A-B --> [A+] + [B-]| Attack [A-] + [B+] --> A-B| Bond-attack A-B + C --> [A+] + B-[C-] | Separation between arrows |
|:-------------:|:-----------------------------:|:-------------------------:|:-------------------------------------:|:-------------------------:|
| MechSMILES    |            ((a,b), b)         |          (a,b)            |              ((a,b), c)               |             ,             |
| PMechDB       |            "a,b=b"            |          "a=b"            |              "b,a=b,c"                |             ;             |
"""

import re
from colorama import Fore
from rdkit.Chem import MolFromSmiles, MolToSmiles, AddHs

from alphamollite.world.mechsmiles import MechSmiles
from alphamollite.world.molecule_set import ReusedVirtualTSException


def remove_mapping_and_canonicalize(smiles, addHs=False, sanitize=True, kekulize=False):
    mol = MolFromSmiles(smiles, sanitize=sanitize)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    if addHs:
        mol = AddHs(mol)

    return MolToSmiles(mol, kekuleSmiles=kekulize)


def safe_translate_and_validate(x):
    try:
        return (*translate_and_validate(x), "")
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except ReusedVirtualTSException:
        return ("", False, "Reused VTS")
    except NotImplementedError:
        return ("", False, "Not implemented")
    except Exception as e:
        print(f"Error with {x}\n{e}\n\n")
        return ("", False, "Unknown error")


def translate_and_validate(pmechdb_string):
    """Translate a PMechDB string into a MechSmiles one and verify that the results are the same"""

    pmechdb_string = re.sub(
        re.escape("[HH]"), "[H][H]", pmechdb_string
    )  # For some reason H2 written as [HH] is not canonicalized like [H][H]
    pmechdb_string = re.sub(
        re.escape("N(=O)=O"), "[N+](=O)[O-]", pmechdb_string
    )  # Rewrite nitro in a form that doesn't violate the octet
    new_pmechdb_string = re.sub(
        r"[@\\/]", "", pmechdb_string
    )  # ChRIMP doesn't support chirality yet

    if pmechdb_string != new_pmechdb_string:
        print(f"{Fore.YELLOW}Careful: ChRIMP doesn't support chirality yet{Fore.RESET}")
        pmechdb_string = new_pmechdb_string

    rxn_pmdb, arr_pmdb = pmechdb_string.split(" ")
    reac_pmdb, cond_pmdb, prod_pmdb = rxn_pmdb.split(
        ">"
    )  # I will not use conditions for now

    arrows_pmdb = arr_pmdb.split(";")
    # arrows_mechsmiles = []

    source_atoms = set()
    sink_atoms = set()

    source_to_arrow_sink = {}

    for arr in arrows_pmdb:
        atck_entity, acpt_entity = arr.split(
            "="
        )  # Entity because we don't know yet if it is an atom or a bond

        atck_entity = ",".join(
            [x for x in atck_entity.split(",") if x != ""]
        )  # Sanitize moves like 10=20, that is multiple times in the dataset
        acpt_entity = ",".join([x for x in acpt_entity.split(",") if x != ""])

        atck_entity_is_bond = "," in atck_entity
        acpt_entity_is_bond = "," in acpt_entity

        if acpt_entity_is_bond:
            # No such thing as attacking a bond in MechSMILES
            # We have to attack the atom in this bond that is not part of the attacker
            set_common_atoms = set(atck_entity.split(",")).intersection(
                set(acpt_entity.split(","))
            )
            assert len(set_common_atoms) == 1, "Didn't plan that yet"
            acpt_entity_split = acpt_entity.split(",")
            acpt_entity = (
                acpt_entity_split[0]
                if acpt_entity_split[0] not in set_common_atoms
                else acpt_entity_split[1]
            )
            acpt_entity_is_bond = False

            if atck_entity_is_bond:
                # We want to reorder it in such a way that the shared is second
                atck_entity_split = atck_entity.split(",")
                atck_entity_split = (
                    atck_entity_split
                    if atck_entity_split[0] not in set_common_atoms
                    else atck_entity_split[::-1]
                )
                atck_entity = ",".join(atck_entity_split)

        acpt = acpt_entity

        if not atck_entity_is_bond:  # atom to atom attack
            atck = atck_entity

            if atck in source_atoms:
                print(f"Source {atck} used multiple times")
            elif acpt in sink_atoms:
                print(f"Sink {acpt} used multiple times")
            else:
                source_atoms.add(atck)
                sink_atoms.add(acpt)

            source_to_arrow_sink[atck] = ((int(atck), int(acpt)), acpt)

        else:
            atck_entity_split = atck_entity.split(",")

            if acpt in atck_entity_split:  # ionization
                atck = (
                    atck_entity_split[0]
                    if atck_entity_split[0] != acpt
                    else atck_entity_split[1]
                )

                if atck in source_atoms:
                    print(f"Source {atck} used multiple times")
                elif acpt in sink_atoms:
                    print(f"Sink {acpt} used multiple times")
                else:
                    source_atoms.add(atck)
                    sink_atoms.add(acpt)

                source_to_arrow_sink[atck] = (((int(atck), int(acpt)), int(acpt)), acpt)

            else:  # bond attack
                atck, inter = atck_entity_split

                # print(f"In bond attack {atck=}, {inter=}, {acpt=}")

                if atck in source_atoms:
                    print(f"Source {atck} used multiple times")
                elif inter in source_atoms:
                    print(f"Source {inter} used multiple times")
                elif acpt in sink_atoms:
                    print(f"Sink {acpt} used multiple times")
                elif inter in sink_atoms:
                    print(f"Sink {inter} used multiple times")
                else:
                    source_atoms.add(atck)
                    source_atoms.add(inter)
                    sink_atoms.add(acpt)
                    sink_atoms.add(inter)

                source_to_arrow_sink[atck] = (
                    ((int(atck), int(inter)), int(acpt)),
                    acpt,
                )

    if (
        len(source_atoms - sink_atoms) == 1 and len(sink_atoms - source_atoms) == 1
    ):  # Unique source and sink, linear arrow flow
        first_source = list(source_atoms - sink_atoms)[0]
        last_sink = list(sink_atoms - source_atoms)[0]
        curr_source = first_source
        arrows_mechsmiles = []
        while curr_source != last_sink:
            arr, curr_source = source_to_arrow_sink[curr_source]
            arrows_mechsmiles.append(arr)

    elif (
        len(source_atoms - sink_atoms) == 0 and len(sink_atoms - source_atoms) == 0
    ):  # No source that is not also a sink, circular arrow flow
        # Here in principle, it doesn't matter form where we start, so let's choose a random pair
        print("Cyclic flow!")
        first_source = list(source_to_arrow_sink.keys())[0]
        last_sink = first_source
        curr_source = first_source
        arrows_mechsmiles = []
        cycled_started = False
        while curr_source != last_sink or not cycled_started:
            cycled_started = True
            arr, curr_source = source_to_arrow_sink[curr_source]
            arrows_mechsmiles.append(arr)

    else:
        print(f"{source_atoms=}")
        print(f"{sink_atoms=}")
        print(
            f"I don't understand the arrow-flow of {pmechdb_string}\n{source_to_arrow_sink=}"
        )

    arrows_mechsmiles_str = ";".join([str(a) for a in arrows_mechsmiles])
    mech_smi = MechSmiles(f"{reac_pmdb}|{arrows_mechsmiles_str}")
    mech_smi.unmap_nonreactive_atoms()
    mech_smi.minimize_indices()

    similar_prod = remove_mapping_and_canonicalize(
        prod_pmdb, addHs=True
    ) == remove_mapping_and_canonicalize(mech_smi.prod, addHs=True)

    if not similar_prod:
        # print(f"Comparing:\n{remove_mapping_and_canonicalize(prod_pmdb)}\n{remove_mapping_and_canonicalize(mech_smi.prod)}")
        # print(f"Not similar prod for {pmechdb_string}")
        pass

    return mech_smi.value, "" if similar_prod else "Not similar prod"
