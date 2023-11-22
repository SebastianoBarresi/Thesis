import argparse
import yaml
import wandb
from pathlib import Path
import pytorch_lightning as pl
import torch
from nn.pl_particlenet import Particlenet
from nn.pl_lorentznet import LorentzNet_TopTagging, LorentzNet_QuarkGluon
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping



def main(hparams, extra_args=None, experiment_name=None):
    torch.set_float32_matmul_precision('high')
    
    if hparams['model'] == 'ParticleNet':
        model = Particlenet(hparams)

    elif hparams['model'] == 'LorentzNet':
        if hparams['dataset'] == 'TopDataset':
            model = LorentzNet_TopTagging(hparams)
        elif hparams['dataset'] == 'QGDataset':
            model = LorentzNet_QuarkGluon(hparams)

    model = model.cuda()

    run_name = f"{experiment_name}{'_lorentz_' + str(extra_args.lorentz_coeff) if extra_args.lorentz_coeff != 0 else ''}{'_beta_' + str(extra_args.max_beta) if extra_args.max_beta is not None else ''}{'_seed_' + str(extra_args.seed) if extra_args.seed is not None else ''}"
    wandb_logger = WandbLogger(entity='jet-tagging', project='lorentz_invariance', tags=[experiment_name], name=run_name)
    logdir = Path('logs_invariance') / run_name
    logdir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=logdir, filename='best_model'),
    ]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=35,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        )
    
    if not (logdir / 'best_model.ckpt').exists():
        trainer.fit(model)

    print('\nTesting model\n')
    if (best_model := logdir / 'best_model.ckpt').exists():
        if hparams['model'] == 'ParticleNet':
            model = Particlenet.load_from_checkpoint(best_model)
        elif hparams['model'] == 'LorentzNet':
            if hparams['dataset'] == 'TopDataset':
                model = LorentzNet_TopTagging.load_from_checkpoint(best_model)
            elif hparams['dataset'] == 'QGDataset':
                model = LorentzNet_QuarkGluon.load_from_checkpoint(best_model)
    if not extra_args.evaluate_invariance:
        trainer.test(model)
    else:
        model.evaluate_invariance(logdir=logdir)



def arg2bool(arg):
    if isinstance(arg, bool):
       return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args():
    parser = argparse.ArgumentParser(description='JetNet')
    parser.add_argument('-c', '--config', default='config/particlenet.yaml', type=str, help='config file')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('-b', '--max_beta', type=float, help='Maximum beta for the Lorentz Augmentation, if 0: disable Lorentz Augmentation')
    parser.add_argument('--lorentz_coeff', type=float, default=0, help='Coefficient of the Lorentz Loss (0 means no Lorentz Loss).')
    parser.add_argument('--evaluate_invariance', type=arg2bool, nargs='?', default=False, const=True, help='Whether to evaluate the invariance of the model')
    parser.add_argument('--disable_wandb', type=arg2bool, nargs='?', default=False, const=True, help='Whether to disable wandb')
    parser.add_argument('-s', '--seed', type=int, help='Seed')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = get_args()
    
    with open(args.config) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    # update hparams with the args
    for arg, value in vars(args).items():
        if value is not None and arg != ['config', 'max_beta']:
            hparams[arg] = value
        if arg == 'max_beta' and value is not None:
            hparams['augmentation_params']['max_beta'] = value
    
    if args.disable_wandb:
        import wandb
        wandb.init(mode='disabled')

    main(hparams, extra_args=args, experiment_name=Path(args.config).stem)
