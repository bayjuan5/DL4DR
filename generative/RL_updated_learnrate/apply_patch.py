files = ["rl_finetune.py", "rl_eval_independent.py"]

old = '''        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue'''

new = '''        mol = Chem.MolFromSmiles(smi)
        # RDKit quirk: Chem.MolFromSmiles("") returns a valid, 0-atom Mol
        # object (not None) - it slips past "mol is None" and produces a
        # PyG graph with zero nodes. Batch.from_data_list then contributes
        # no rows to data.batch for that graph, so the DMPNN batch-index
        # count (batch_vec.max()+1) comes out one short of the actual batch
        # size, and DrugResponseModel.forward's dmpnn_tokens_list[i] loop
        # (indexed by genomic_img.shape[0]) runs off the end of the list.
        # An under-trained/early-RL policy emits empty decodes often enough
        # (immediate EOS) that this isn't a rare edge case here.
        if mol is None or mol.GetNumAtoms() == 0:
            continue'''

for fn in files:
    with open(fn) as f:
        src = f.read()
    if old not in src:
        print(f"{fn}: OLD BLOCK NOT FOUND -- file may differ from expected, skipping")
        continue
    count = src.count(old)
    if count != 1:
        print(f"{fn}: old block matches {count} times (expected 1) -- ambiguous, skipping")
        continue
    src2 = src.replace(old, new)
    with open(fn, "w") as f:
        f.write(src2)
    print(f"{fn}: patched OK")
