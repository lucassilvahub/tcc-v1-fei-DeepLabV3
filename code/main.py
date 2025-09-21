from skimage import io
import os, time
import torch
import numpy as np
import pandas as pd
from glob import glob
from project_utils import load_loss_weights, batch_mean_and_sd

from dataset import DatasetIcmbio
from trainer import Trainer
from models import build_model
import matplotlib.pyplot as plt
from project_utils import clear, convert_to_color, make_optimizer, seed_everything, visualize_augmentations
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def is_save_epoch(epoch, ignore_epoch=0):
    return params['save_epoch'] is not None and epoch % params['save_epoch'] == 0 and epoch != ignore_epoch
    
class LossFN:
    CROSS_ENTROPY = 'cross_entropy'
    FOCAL_LOSS = 'focal_loss'
    DICE = 'DICE'
    JACCARD = 'JACCARD'
    TVERSKY = 'TVERSKY'

class ModelChooser:
    SEGNET_MODIFICADA = 'segnet_modificada'
    UNET = 'unet'
    SEGFORMER = 'segformer'
    DEEPLABV3PLUS = 'deeplabv3plus'


class Callback():

    def __init__(self, patience = 10, min_value = 66):
        
        self.PATIENCE = patience
        self.COUNTER = 0
        self.MIN_LIMIT = min_value
        self.BEST_VALUE = 0
        self.BEST_TRAINER = []

    def patience_loss(self, epoch):
        if trainer.epoch_loss[epoch-1] < self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_loss[epoch-1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_loss[epoch-1] >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Loss Too High | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False
    
    def patience_acc(self, epoch):
        if trainer.epoch_acc[epoch-1] > self.BEST_VALUE:
            self.BEST_VALUE = trainer.epoch_acc[epoch-1]
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        elif trainer.epoch_acc[epoch-1] <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False
    
    def patience_acc_val(self, avg_acc):
        if avg_acc > self.MIN_LIMIT and avg_acc > self.BEST_VALUE:
            self.BEST_VALUE = avg_acc
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val acc < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_f1_val(self, f1):
        if f1 > self.MIN_LIMIT and f1 > self.BEST_VALUE:
            self.BEST_VALUE = f1
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val F1-Score < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_iou_val(self, iou):
        if iou > self.MIN_LIMIT and iou > self.BEST_VALUE:
            self.BEST_VALUE = iou
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New best epoch | Saving model...")
            return True
        elif self.BEST_VALUE <= self.MIN_LIMIT:
            print(f"PATIENCE :::: Val mIoU < {self.MIN_LIMIT} % | Skipping save...")
            return False
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: Waiting for {self.COUNTER} epoch(s) | Skipping save...")
            return False
    
    def patience_loss_val(self, avg_loss):
        if avg_loss >= self.MIN_LIMIT:
            print(f"PATIENCE :::: Accuracy Too low | Skipping Save...")
            return False
        elif avg_loss < self.BEST_VALUE:
            self.BEST_VALUE = avg_loss
            self.BEST_TRAINER = trainer
            self.COUNTER = 0
            print(f"PATIENCE ::: New Best Epoch | Saving Model...")
            return True
        else:
            self.COUNTER += 1
            print(f"PATIENCE ::: {self.COUNTER} Epoch(s) Without Improvement | Skipping Save...")
            return False


def weights_calculator_loss(params, train_labels):
    try:
        if params['loss']['name'] == LossFN.CROSS_ENTROPY:
            if params['loss']['params']['weights'] == 'equal':
                params['weights'] = torch.ones(params['n_classes'])
            elif params['loss']['params']['weights'] == 'calculate':
                if os.path.exists('./loss_weights.npy'):
                    loss_weights = load_loss_weights('./loss_weights.npy')
                    params['weights'] = torch.from_numpy(loss_weights['weights']).float()
                else:
                    import extra.weights_calculator as wc
                    loss_weights, _ = wc.WeightsCalculator(train_labels, params['classes'], dev=False).calculate_and_save()
                    params['weights'] = torch.from_numpy(loss_weights).float()
        elif params['loss']['name'] == LossFN.FOCAL_LOSS:
            params['weights'] = torch.ones(params['n_classes'])

        # Imprimindo os pesos das classes para a loss
        print(params['weights'])
    except Exception as e:
        print(e)
        raise e
    
    
if __name__=='__main__':

    # Registra o tempo de início do treinamento
    start_time = time.time()
    
    # Params
    params = {
        'root_dir': '../../dataset_35/', # Diretório raiz dos dados
        'cache': True,
        
        'window_size': (224, 224), # Tamanho das imagens de entrada da rede
        'bs': 40, # Batch size
        'n_classes': 8, # Número de classes
        'classes': ["Urbano", "Vegetação Densa", "Sombra", "Vegetação Esparsa", "Agricultura", "Rocha", "Solo Exposto", "Água"], # Nome das classes
        'maximum_epochs': 999, # Número de épocas de treinaento
        'save_epoch': 2, # Salvar o modelo a cada n épocas para evitar perder o treinamento caso ocorra algum erro ou queda de energia
        'print_each': 100, # Print each n iterations (apenas para acompanhar visualmente o treinamento)
        'augment': False,
        
        'cpu': None, # CPU ou GPU. Se None, será usado GPU. Não vai funcionar com CPU
        'device': 'cuda', # GPU
        'precision' : 'full', # Precisão dos cálculos. 'full' ou 'half'. 'full' é mais preciso, mas mais lento. 'half' é mais rápido, mas menos preciso. Default: 'full'

        'optimizer_params': {
            'optimizer': 'ADAM',
            'lr': 1e-3,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0,
            'epsilon': 1e-8,
            'momentum': 0.9
        },
        'lrs_params': {
            'type': 'Plateau',
            'lr_decay': 30,
            'milestones': [25, 35, 45],
            'gamma': 0.1
        },
        'weights': '', # Peso de cada classe para a loss. Será calculado automaticamente em seguida
        'loss': {
            'name': LossFN.TVERSKY, # Escolha entre 'CROSS_ENTROPY' ou 'FOCAL_LOSS' 'DICE'
            'params': {
                'weights': 'calculate', # Escolha entre 'equal' ou 'calculate'. Se 'equal', os pesos serão iguais. Se 'calculate', os pesos serão calculados pelo arquivo `extra\weights_calculator.py`
                'alpha': 0.5, # Somente para FOCAL_LOSS. Informe um valor float. Default: 0.5
                'gamma': 2.0, # Somente para FOCAL_LOSS. Informe um valor float. Default: 2.0
            }
        },
        'patience': 10,
        
        'model': {
            'name': ModelChooser.UNET, # Escolha entre 'SEGNET_MODIFICADA' ou 'UNET' ou 'SEGFORMER' DEEPLABV3PLUS
        },
        'results_folder': "output_35", # Pasta onde serão salvos os resultados
    }
    
    params['results_folder'] = f"output_35/K1x5noAug_{params['model']['name']}b45drop2_imgnet_{params['optimizer_params']['optimizer']}{params['optimizer_params']['weight_decay']}WD_{params['loss']['name']}1.0-0.5_noWeight"
    
    image_dir = os.path.join(params['root_dir'], 'images')
    label_dir = os.path.join(params['root_dir'], 'labels')
    edges_dir = os.path.join(params['root_dir'], 'edges')

    # Load image and label files from .txt
    train_images1 = pd.read_table('folds/fold1_images.txt',header=None).values
    train_images2 = pd.read_table('folds/fold2_images.txt',header=None).values
    train_images3 = pd.read_table('folds/fold3_images.txt',header=None).values
    train_images = [os.path.join(image_dir, f[0]) for f in np.concatenate([train_images1,train_images2,train_images3])]
    train_labels1 = pd.read_table('folds/fold1_labels.txt',header=None).values
    train_labels2 = pd.read_table('folds/fold2_labels.txt',header=None).values
    train_labels3 = pd.read_table('folds/fold3_labels.txt',header=None).values
    train_labels = [os.path.join(label_dir, f[0]) for f in np.concatenate([train_labels1,train_labels2,train_labels3])]
    #train_edges = pd.read_table('train_labels.txt',header=None).values
    #train_edges = [os.path.join(edges_dir, f[0]) for f in train_edges]
    
    val_images = pd.read_table('folds/fold4_images.txt',header=None).values
    val_images = [os.path.join(image_dir, f[0]) for f in val_images]
    val_labels = pd.read_table('folds/fold4_labels.txt',header=None).values
    val_labels = [os.path.join(label_dir, f[0]) for f in val_labels ]
    
    test_images = pd.read_table('folds/fold5_images.txt',header=None).values
    test_images = [os.path.join(image_dir, f[0]) for f in test_images]
    test_labels = pd.read_table('folds/fold5_labels.txt',header=None).values
    test_labels = [os.path.join(label_dir, f[0]) for f in test_labels]

    # Carregar os pesos de cada classe, calculados pelo arquivo `extra\weights_calcupator.py`
    weights_calculator_loss(params, train_labels)    

    # Create train and test sets
    train_dataset = DatasetIcmbio(train_images, train_labels, None, window_size = params['window_size'], cache = params['cache'], augmentation=params['augment'])
    val_dataset = DatasetIcmbio(val_images, val_labels, window_size = params['window_size'], cache = params['cache'], augmentation=False)
    test_dataset = DatasetIcmbio(test_images, test_labels, window_size = params['window_size'], cache = params['cache'], augmentation=False)

    # # Load dataset classes in pytorch dataloader handler object
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params['bs'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = params['bs'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params['bs'], shuffle=False)

    model = build_model(model_name=params['model']['name'], params=params)

    loader = {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader,
    }
    
    #cbkp=f"{params['results_folder']}/best_epoch.pth.tar" #None {params['results_folder']}
    #20250509_t1_unet_augTrue/focal_loss_calculate_ADAM_multi/
    cbkp=f"output_35/K1x10augRot90_unetb05noDrop_imgnet_ADAM0WD_focal_loss/best_epoch7573.pth.tar" #None {params['results_folder']}
    trainer = Trainer(model, loader, params, cbkp=None)
    # print(trainer.test(stride = 32, all = False))
    # _, all_preds, all_gts = trainer.test(all=True, stride=32)
    clear()
    
    patCB = Callback(patience = params['patience'], min_value = 60) 
    
    # Start the training.
    for epoch in range(trainer.last_epoch+1, params['maximum_epochs']):
        acc_train, f1score_train, mcc_train, iou_train = trainer.train()
        acc_val, f1score_val, mcc_val, iou_val = trainer.validate(stride=64)
        
        trainer.epoch_acc.append(acc_train)
        trainer.epoch_val_acc.append(acc_val)
        trainer.epoch_f1.append(f1score_train)
        trainer.epoch_val_f1.append(f1score_val)
        trainer.epoch_mcc.append(mcc_train)
        trainer.epoch_val_mcc.append(mcc_val)
        trainer.epoch_iou.append(iou_train)
        trainer.epoch_val_iou.append(iou_val)
        
        trainer.plot_metrics(params['results_folder'])
        
        if trainer.scheduler is not None:
            trainer.scheduler.step(iou_val)#f1score_val
        
        #if is_save_epoch(epoch, ignore_epoch=params['maximum_epochs']):
        if patCB.patience_iou_val(iou_val):
            
            # acc = trainer.test(stride = min(params['window_size']), all=False)
            # trainer.save('./segnet256_epoch_{}.pth.tar'.format(epoch))
            trainer.save(os.path.join(params['results_folder'], 'best_epoch.pth.tar'))
    
            #trainer.save(os.path.join(params['results_folder'], '{}_{}.pth.tar'.format(params['model']['name'], params['maximum_epochs'])))
        
        if patCB.COUNTER == patCB.PATIENCE:
            #trainer.save(os.path.join(params['results_folder'], 'last_epoch.pth.tar'))
        
            print(f"PATIENCE :::  Training Terminated | Best Epoch = {epoch-10} ")#| Loss = {trainer.epoch_loss[epoch-11]} | Acc = {trainer.epoch_acc[epoch-11]}
            #trainer = patCB.BEST_TRAINER
            break
    
    np.savez(os.path.join(params['results_folder'], 'metrics_train.npz'),
      acc_train = trainer.epoch_acc,
      acc_val = trainer.epoch_val_acc,
      f1score_train = trainer.epoch_f1,
      f1score_val = trainer.epoch_val_f1,
      iou_train = trainer.epoch_iou,
      iou_val = trainer.epoch_val_iou)
    
    # Registra o tempo de término do treinamento
    end_time = time.time()
    # Calcula o tempo gasto em horas
    training_time = end_time - start_time
    training_time_hours = training_time / 3600.0
    print("Tempo gasto treinando: {:.2f} horas".format(training_time_hours))
    
    trainer = Trainer(model, loader, params, cbkp=os.path.join(params['results_folder'], 'best_epoch.pth.tar'))
    
    # acc, all_preds, all_gts = trainer.test(all=True, stride=min(params['window_size']))
    all_preds = trainer.test(stride=64, all=True)#acc,  , all_gts, _mc_dropout, mc_runs=25
    #print(f'Global Accuracy: {acc}')
    training_time = time.time() - end_time
    training_time_hours = training_time / 3600.0
    print("Tempo gasto em inferências MCDropout: {:.2f} horas".format(training_time_hours))
    
    input_ids, label_ids, _ = test_loader.dataset.get_dataset()
    all_ids = [os.path.split(f)[1].split('.')[0] for f in input_ids]
    
    os.makedirs(os.path.join(params['results_folder'], 'inference'), exist_ok=True)
    for p, id_ in zip(all_preds, all_ids):
        img = convert_to_color(p)
        # plt.imshow(img) and plt.show()
        # io.imsave('./tmp/inference_tile_{}.png'.format(id_), img)
        io.imsave(os.path.join(params['results_folder'], 'inference', 'inference_tile_{}.png'.format(id_)), img)
         
