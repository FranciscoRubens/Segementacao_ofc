# PIPELINE DE TREINAMENTO, TESTE FINAL E AVALIAÇÃO EXTERNA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os, gc, joblib
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, recall_score, jaccard_score, mean_absolute_error
from imblearn.metrics import specificity_score
from sklearn.metrics import f1_score as dice_coefficient
from scipy.stats import wilcoxon
import itertools

# IMPORTAR ARQUITETURAS 
from Models.unet import UNet
from Models.wnet import WNet
from Models.attunet import AttentionUNet
from Models.unetplusplus import UNetPlusPlus
from Models.unet3plus import UNet3Plus

#Limpar a memória da GPU e coleta lixo do sistema.
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# CONFIGURAÇÃO BASE 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {device}")

base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, "Dataset")

dataset1_dirs = {"images": os.path.join(datasets_dir, "Archive/images"),
                 "masks": os.path.join(datasets_dir, "Archive/mask")}
dataset2_dirs = {"images": os.path.join(datasets_dir, "Dataset_and_code/images"),
                 "masks": os.path.join(datasets_dir, "Dataset_and_code/mask")}
dataset3_dirs = {"images": os.path.join(datasets_dir, "Panoramic_radiography_database/images"),
                 "masks": os.path.join(datasets_dir, "Panoramic_radiography_database/mask")}

# FUNÇÃO PARA CARREGAR CAMINHOS
def load_paths_auto(image_dir, mask_dir, mask_extensions=None):
    if mask_extensions is None:
        mask_extensions = ['.png', '.bmp', '.jpg', '.jpeg']
    image_paths, mask_paths = [], []
    for fname in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, fname)
        base_name = os.path.splitext(fname)[0]
        for ext in mask_extensions:
            candidate = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(candidate):
                image_paths.append(img_path)
                mask_paths.append(candidate)
                break
    return image_paths, mask_paths

# DATASET
class MaskDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))

        augmented = self.transform(image=img, mask=mask)
        input_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float()
        mask_tensor = (mask_tensor > 0.5).float()
        return input_tensor, mask_tensor

# TRANSFORMAÇÕES
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
    A.Resize(256, 512),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# PERDAS E MÉTRICAS
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return bce + d_loss

def enhanced_alignment_measure(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32)
    if np.all(y_true == y_true[0]) and np.all(y_pred == y_pred[0]):
        return 1.0 if y_true[0] == y_pred[0] else 0.0
    if y_pred.max() != y_pred.min():
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    if y_true.max() != y_true.min():
        y_true = (y_true - y_true.min()) / (y_true.max() - y_true.min())
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    numerator = 2 * (y_true - mean_true) * (y_pred - mean_pred)
    denominator = (y_true - mean_true)**2 + (y_pred - mean_pred)**2 + 1e-8
    align = numerator / denominator
    return np.clip(align, -1, 1).mean()

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    return {
        "Accuracy": accuracy_score(y_true_bool, y_pred_bool),
        "Specificity": specificity_score(y_true_bool, y_pred_bool),
        "Sensitivity": recall_score(y_true_bool, y_pred_bool),
        "E-measure": enhanced_alignment_measure(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "IoU": jaccard_score(y_true_bool, y_pred_bool),
        "Dice": dice_coefficient(y_true_bool, y_pred_bool)
    }

def get_model_output(model_name, outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs

# GRIDS DE HIPERPARÂMETROS 
param_grid_unet = {"learning_rate": [1e-3, 5e-4], "batch_size": [8, 16], "optimizer": ["Adam", "AdamW"]}
param_grid_att_unet = {"learning_rate": [1e-3, 5e-4], "batch_size": [8, 16], "optimizer": ["Adam","AdamW"]}
param_grid_unet3p = {"learning_rate": [1e-3, 5e-4], "batch_size": [4, 8], "optimizer": ["Adam", "AdamW"]}
param_grid_wnet = {"learning_rate": [1e-4, 5e-5], "batch_size": [8, 16], "optimizer": ["Adam", "AdamW"]}
param_grid_unetpp = {"learning_rate": [1e-3, 5e-4], "batch_size": [4, 8], "optimizer": ["Adam", "AdamW"]}

model_param_grids = {
    "UNet": (UNet, param_grid_unet),
    "UNetPlusPlus": (UNetPlusPlus, param_grid_unetpp),
    "UNet3Plus": (UNet3Plus, param_grid_unet3p),
    "AttUNet": (AttentionUNet, param_grid_att_unet),
    "WNet": (WNet, param_grid_wnet)
}

def create_model(model_name):
    if model_name == "AttUNet":
        return AttentionUNet().to(device)
    elif model_name == "UNet":
        return UNet(num_classes=1).to(device)
    elif model_name == "UNet3Plus":
        return UNet3Plus(in_channels=1, n_classes=1).to(device)
    elif model_name == "UNetPlusPlus":
        return UNetPlusPlus(in_channels=1, out_channels=1).to(device)
    elif model_name == "WNet":
        return WNet(in_ch=1, out_ch=1).to(device)

# ============================================================
# 1️⃣ GRID SEARCH + K-FOLD (Dataset1)
# ============================================================
dataset1_imgs, dataset1_masks = load_paths_auto(dataset1_dirs["images"], dataset1_dirs["masks"])

best_configs_all = {}

for model_name, (model_class, param_grid) in model_param_grids.items():
    print("\n" + "="*100)
    print(f" Treinando arquitetura: {model_name}")
    print("="*100)

    all_results = []
    best_history_per_fold = {}
    best_configs = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset1_imgs), 1):
        print(f"\n--- Fold {fold} ---")
        fold_train_img = [dataset1_imgs[i] for i in train_idx]
        fold_train_mask = [dataset1_masks[i] for i in train_idx]
        fold_test_img = [dataset1_imgs[i] for i in test_idx]
        fold_test_mask = [dataset1_masks[i] for i in test_idx]

        train_img, val_img, train_mask, val_mask = train_test_split(
            fold_train_img, fold_train_mask, test_size=0.2, random_state=42
        )

        best_val_dice = -1
        best_model_state = None
        best_config_fold = None

        test_dataset = MaskDataset(fold_test_img, fold_test_mask, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for params in ParameterGrid(param_grid):
            print(f"\n Config: {params}")

            train_dataset = MaskDataset(train_img, train_mask, transform=train_transform)
            val_dataset = MaskDataset(val_img, val_mask, transform=val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            

            model = create_model(model_name)
            if params["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
            elif params["optimizer"] == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
          
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            history = {"train_dice": [], "val_dice": []}

            for epoch in range(50):  # número de épocas do Grid Search
                model.train()
                train_preds, train_targets = [], []
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    seg_pred = get_model_output(model_name, model(inputs))
                    loss = combined_loss(seg_pred, targets)
                    loss.backward()
                    optimizer.step()

                    pred_bin = (seg_pred.detach().cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
                    train_preds.extend(pred_bin.flatten())
                    train_targets.extend(targets.cpu().numpy().flatten())
                train_metrics = calculate_metrics(train_targets, train_preds)
                history["train_dice"].append(train_metrics["Dice"])

                # Validação
                model.eval()
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        seg_pred = get_model_output(model_name, model(inputs))
                        pred_bin = (seg_pred.detach().cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
                        val_preds.extend(pred_bin.flatten())
                        val_targets.extend(targets.cpu().numpy().flatten())
                val_metrics = calculate_metrics(val_targets, val_preds)
                history["val_dice"].append(val_metrics["Dice"])
                scheduler.step(val_metrics["Dice"])

            # Seleção do melhor modelo do fold
            if val_metrics["Dice"] > best_val_dice:
                best_val_dice = val_metrics["Dice"]
                best_config_fold = params
                best_model_state = model.state_dict()
                best_history_per_fold[fold] = history
            # LIMPEZA DE MEMÓRIA APÓS CADA CONFIGURAÇÃO
            del model, optimizer, scheduler, train_loader, val_loader
            clear_gpu_memory()
        
        # 🔻 LIMPEZA ANTES DE SALVAR O CHECKPOINT
        clear_gpu_memory()
        # Salvar checkpoint do fold

        checkpoint_dir = os.path.join(base_dir, "checkpoints", model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(checkpoint_dir, f"best_model_{model_name}_fold{fold}.pt"))
        joblib.dump(best_config_fold, os.path.join(checkpoint_dir, f"best_params_{model_name}_fold{fold}.pkl"))

        print(f"Checkpoint salvo: {model_name} - Fold {fold}")
        print(f"Melhor Dice Val: {best_val_dice:.4f}")
        print(f"Melhor config: {best_config_fold}")

        # Avaliação no fold test
        model = create_model(model_name)   
        model.load_state_dict(best_model_state)
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                seg_pred = get_model_output(model_name, model(inputs))
                pred_bin = (seg_pred.detach().cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
                test_preds.extend(pred_bin.flatten())
                test_targets.extend(targets.cpu().numpy().flatten())
        test_metrics = calculate_metrics(test_targets, test_preds)
        test_metrics['Fold'] = fold
        test_metrics['Config'] = str(best_config_fold)
        all_results.append(test_metrics)

        # LIMPEZA APÓS TESTE DE FOLD
        del model, test_loader,test_dataset
        clear_gpu_memory()

        # Gráfico Dice
        best_history = best_history_per_fold[fold]
        graphs_root = os.path.join(base_dir, "graficos", model_name)
        os.makedirs(graphs_root, exist_ok=True)
        graph_path = os.path.join(graphs_root, f"dice_{model_name}_fold{fold}.png")
        plt.figure(figsize=(8,5))
        plt.plot(best_history["train_dice"], label="Treino")
        plt.plot(best_history["val_dice"], label="Validação")
        plt.xlabel("Época")
        plt.ylabel("Dice")
        plt.title(f"{model_name} - Fold {fold}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()
    
    # LIMPEZA GERAL APÓS CADA MODELO
    clear_gpu_memory()

    # Resultados finais do Grid Search
    results_root = os.path.join(base_dir, "resultados", model_name)
    os.makedirs(results_root, exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(results_root, f"gridsearch_{model_name}_folds_final.csv"), index=False)

    # Média e desvio
    metrics_cols = ["Accuracy","Specificity","Sensitivity","E-measure","MAE","IoU","Dice"]
    mean_metrics = df_results[metrics_cols].mean()
    std_metrics = df_results[metrics_cols].std()
    summary_df = pd.DataFrame({"Métrica": metrics_cols,
                               "Média": [mean_metrics[m] for m in metrics_cols],
                               "Desvio Padrão":[std_metrics[m] for m in metrics_cols]})
    summary_df.to_csv(os.path.join(results_root, f"media_desvio_{model_name}.csv"), index=False)

    # Melhor configuração geral da rede
    best_df = pd.DataFrame([(res['Config'], res['Dice']) for res in all_results], columns=["Config","Dice"])
    mean_dice_by_config = best_df.groupby("Config")["Dice"].mean().reset_index()
    best_overall_row = mean_dice_by_config.loc[mean_dice_by_config["Dice"].idxmax()]
    best_overall_config = eval(best_overall_row["Config"])
    best_configs_all[model_name] = best_overall_config
    joblib.dump(best_overall_config, os.path.join(results_root, f"best_config_{model_name}_overall.pkl"))

# ============================================================
# 2️⃣ TREINAMENTO FINAL COM DATASET1
# ============================================================

dataset1_train_dataset = MaskDataset(dataset1_imgs, dataset1_masks, transform=train_transform)

for model_name in ["UNet", "UNetPlusPlus", "UNet3Plus", "AttUNet", "WNet"]:
    
    # Carregar melhor configuração
    config_path = os.path.join(base_dir, "resultados", model_name, f"best_config_{model_name}_overall.pkl")
    best_config = joblib.load(config_path)
    print(f"\n===== Treinamento final: {model_name} =====")
    print(f"Usando melhor configuração: {best_config}")

    # DataLoader
    train_loader = DataLoader(dataset1_train_dataset, batch_size=best_config["batch_size"], shuffle=True, num_workers=2 if torch.cuda.is_available() else 0, pin_memory=torch.cuda.is_available())

    # Criar modelo e mover para device
    model = create_model(model_name).to(device)

    # Otimizador
    if best_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    elif best_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_config["learning_rate"])
    
    # Scheduler (reduz LR se o loss não melhorar)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Treinamento final
    for epoch in range(50):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            seg_pred = get_model_output(model_name, model(inputs))
            loss = combined_loss(seg_pred, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/50] - Loss: {avg_loss:.4f}")

    # Salvar modelo final
    final_root = os.path.join(base_dir, "resultados_finais", "Dataset1", model_name)
    os.makedirs(final_root, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_root, f"{model_name}_final_trained.pt"))

    # =====================================================
    # 🔹 Limpeza de memória após terminar cada modelo
    # =====================================================
    del model, optimizer, scheduler, train_loader
    clear_gpu_memory()

# ============================================================
# 3️⃣ AVALIAÇÃO EM DATASETS EXTERNOS
# ============================================================
final_models_paths = {
    model_name: os.path.join(base_dir, "resultados_finais", "Dataset1", model_name, f"{model_name}_final_trained.pt")
    for model_name in ["UNet", "UNetPlusPlus", "UNet3Plus", "AttUNet", "WNet"]
}

external_datasets = {"Dataset2": dataset2_dirs, "Dataset3": dataset3_dirs}

for dataset_name, dirs in external_datasets.items():
    imgs, masks = load_paths_auto(dirs["images"], dirs["masks"])
    test_dataset = MaskDataset(imgs, masks, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for model_name, model_path in final_models_paths.items():
        print(f"\n===== Avaliação externa: {dataset_name} - {model_name} =====")

        # Carregar e mover o modelo para device **apenas aqui**
        model = create_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        preds_list, targets_list = [], []

        # Criar pasta de máscaras
        masks_pred_dir = os.path.join(base_dir, "resultados_finais", dataset_name, model_name, "masks_pred")
        os.makedirs(masks_pred_dir, exist_ok=True)

        max_masks_to_save = 10
        saved_count = 0

        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                seg_pred = get_model_output(model_name, model(inputs))

                pred_bin = (seg_pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
                pred_bin = np.squeeze(pred_bin)

                if pred_bin.ndim == 3 and pred_bin.shape[0] == 1:
                    pred_bin = pred_bin[0]

            if saved_count < max_masks_to_save:
                mask_img = Image.fromarray(pred_bin * 255)
                mask_img.save(os.path.join(masks_pred_dir, f"mask_{i}.png"))
                saved_count += 1

            preds_list.extend(pred_bin.flatten())
            targets_list.extend(targets.cpu().numpy().flatten())

        metrics = calculate_metrics(targets_list, preds_list)
        metrics_df = pd.DataFrame([metrics])
        metrics_root = os.path.join(base_dir, "resultados_finais", dataset_name, model_name)
        os.makedirs(metrics_root, exist_ok=True)
        metrics_df.to_csv(os.path.join(metrics_root, "metrics.csv"), index=False)
        print(f"Métricas salvas: {metrics_root}")

        # Limpeza após avaliação de cada modelo
        del model, preds_list, targets_list
        torch.cuda.empty_cache()
        gc.collect()

    # Limpeza após terminar todo o dataset
    del test_loader, test_dataset
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# 4️⃣ TESTE DE WILCOXON
# ============================================================

wilcoxon_results = {}
for (model_a, model_b) in itertools.combinations(best_configs_all.keys(), 2):
    df_a = pd.read_csv(os.path.join(base_dir, "resultados", model_a, f"gridsearch_{model_a}_folds_final.csv"))
    df_b = pd.read_csv(os.path.join(base_dir, "resultados", model_b, f"gridsearch_{model_b}_folds_final.csv"))
    stat, p_val = wilcoxon(df_a["Dice"], df_b["Dice"])
    wilcoxon_results[f"{model_a} vs {model_b}"] = p_val

# Cria o DataFrame com coluna de significância
wilcoxon_df = pd.DataFrame(list(wilcoxon_results.items()), columns=["Comparação","p-value"])
wilcoxon_df["Significativo?"] = wilcoxon_df["p-value"].apply(lambda x: "Sim" if x <= 0.05 else "Não")

# Salva o CSV
wilcoxon_csv_path = os.path.join(base_dir, "resultados", "wilcoxon.csv")
wilcoxon_df.to_csv(wilcoxon_csv_path, index=False)

# Imprime os resultados no terminal
print("\n Wilcoxon salvo! Resultados:")
for idx, row in wilcoxon_df.iterrows():
    signif_text = "Diferença significativa" if row["Significativo?"] == "Sim" else "Diferença não significativa"
    print(f"{row['Comparação']}: {row['p-value']:.4f} ({signif_text})")
