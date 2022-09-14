import os
import sys
import torch
import main_patho_lightning
from fvcore.nn import FlopCountAnalysis


if __name__ == '__main__':
    ap = main_patho_lightning.parser()
    hparams = ap.parse_args()
    hparams.executable = "main_lincls_lightning"

    #if not os.path.exists(hparams.output_path):
    #    os.makedirs(hparams.output_path)

    # checking if there is dset_paths.json in key2ai-med/meta_data/, if so dataset-paths can just be read
    hparams = main_patho_lightning.dset_shortcut(hparams)

    # automaticly setting the input size for transformers
    hparams = main_patho_lightning.transformer_input_size_sanity(hparams)

    # repeating experiments k times (allows for mean and std of AUC/ Accuracy, reported numbers are solid)
    results = []
    model, logger, trainer = None, None, None
    pretrained = hparams.pretrained
    # loading pretrained model, if specified           
    if pretrained != "":
        if not os.path.isfile(pretrained):
            if os.path.isdir(pretrained):
                hparams.pretrained = sorted([os.path.join(pretrained, p) for p in os.listdir(pretrained) if p.endswith('.pth')])[idx_experiment]
            else:
                raise FileNotFoundError(f'{hparams.pretrained} is neither file nor directory.')
    # Model initialization
    model = main_patho_lightning.construct_model(hparams)
    dloader = model.test_dataloader()
    model = model.model
    model = model.to(torch.device('cpu'))
    model.eval()
    for X, _ in dloader:
        X = X.to(torch.device('cpu'))
        flops = FlopCountAnalysis(model, X)
        print(f'{flops.total()/(10**9):.2f} GFLOPS')
        break

