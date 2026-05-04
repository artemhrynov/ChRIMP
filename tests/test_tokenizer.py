from tokenizers import Tokenizer

tok = Tokenizer.from_file(
    "src/chrimp/agent/tokenizer/mechsmiles_tokenizer.json"
)

examples = [
    # simple isolated stereo updates
    "TH(1,'invert',((5,6),))",
    "TH(1,'retain',((5,6),))",
    "TH(1,'clear',())",
    "TH(1,'unknown',())",

    # larger atom-map numbers
    "TH(12,'invert',((34,56),))",
    "TH(99,'retain',((10,11),))",

    # multiple ligand replacements
    "TH(1,'invert',((5,6),(7,8)))",
    "TH(12,'retain',((34,56),(78,90)))",

    # full MechSMILES with stereo update
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'invert',((5,6),))",

    # full MechSMILES with normal arrow and retain
    "[OH-:1].[C@:2]([H:3])([Cl:4])[Br:5]|(1,2)|TH(2,'retain',((5,1),))",

    # full MechSMILES with bond attack and stereo
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|((2,3),1)|TH(1,'invert',((4,6),))",

    # several arrows plus one stereo update
    "[OH-:1].[C@:2]([H:3])([Cl:4])[Br:5]|(1,2);((2,5),5)|TH(2,'invert',((5,1),))",

    # clear/unknown in full strings
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'clear',())",
    "[C@:1]([F:2])([Cl:3])([Br:4])[I:5]|(6,1)|TH(1,'unknown',())",

    # chirality symbols @ and @@
    "F[C@:1]([Cl:2])([Br:3])[I:4]|(5,1)|TH(1,'retain',((4,5),))",
    "F[C@@:1]([Cl:2])([Br:3])[I:4]|(5,1)|TH(1,'invert',((4,5),))",
]


for s in examples:
    enc = tok.encode(s)
    assert "[unk]" not in enc.tokens, f"Tokenizer failed on: {s}\n{enc.tokens}"

print("All examples passed.")