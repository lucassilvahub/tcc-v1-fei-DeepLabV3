import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"

def generate_miou_report():
    log_file = OUTPUTS_DIR / "training_log.txt"
    
    if not log_file.exists():
        print("Arquivo de log não encontrado!")
        return
    
    fold_scores = []
    with open(log_file) as f:
        for line in f:
            if "Melhor mIoU fold" in line:
                score = float(line.strip().split(":")[-1])
                fold_scores.append(score)
    
    if not fold_scores:
        print("Nenhuma métrica encontrada!")
        return
    
    # Calcular estatísticas
    mean = np.mean(fold_scores)
    std = np.std(fold_scores)
    min_score = np.min(fold_scores)
    max_score = np.max(fold_scores)
    
    # Gerar relatório
    report_file = OUTPUTS_DIR / "miou_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE RESULTADOS - DeepLabV3 ResNet50\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Modelo: DeepLabV3 com ResNet50 (pretrained ImageNet)\n")
        f.write("Dataset: Petrópolis - Imagens raster01 a raster35\n")
        f.write("Cross-Validation: 5-Folds\n\n")
        
        f.write("-" * 60 + "\n")
        f.write("RESULTADOS POR FOLD (mIoU):\n")
        f.write("-" * 60 + "\n")
        for i, score in enumerate(fold_scores, 1):
            f.write(f"  Fold {i}: {score:.4f} ({score*100:.2f}%)\n")
        
        f.write("\n" + "-" * 60 + "\n")
        f.write("ESTATÍSTICAS GERAIS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Média (mIoU):       {mean:.4f} ({mean*100:.2f}%)\n")
        f.write(f"  Desvio Padrão:      {std:.4f} ({std*100:.2f}%)\n")
        f.write(f"  Mínimo:             {min_score:.4f} ({min_score*100:.2f}%)\n")
        f.write(f"  Máximo:             {max_score:.4f} ({max_score*100:.2f}%)\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nRelatório salvo em: {report_file}\n")
    
    # Imprimir também no console
    with open(report_file, "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    generate_miou_report()