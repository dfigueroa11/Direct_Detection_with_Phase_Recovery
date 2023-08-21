import torch

def get_ER(tx, rx): 
    assert tx.dim() == 1 , "only for one dimentional inputs"
    assert rx.dim() == 1 , "only for one dimentional inputs"
    return torch.sum(tx != rx)/torch.numel(tx)