import torch
import sys

print("="*60)
print("DIAGNÓSTICO GPU")
print("="*60)

print(f"\n1. Versão do Python: {sys.version}")
print(f"2. Versão do PyTorch: {torch.__version__}")
print(f"3. CUDA disponível no PyTorch: {torch.cuda.is_available()}")
print(f"4. Versão CUDA compilada no PyTorch: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"\n✅ GPU DETECTADA!")
    print(f"   - Número de GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   - Memória total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print(f"\n❌ GPU NÃO DETECTADA!")
    print(f"\nPossíveis causas:")
    print(f"   1. PyTorch instalado sem suporte CUDA")
    print(f"   2. Driver NVIDIA desatualizado ou não instalado")
    print(f"   3. CUDA Toolkit não compatível")
    
print("\n" + "="*60)
print("TESTES")
print("="*60)

# Teste rápido
try:
    x = torch.randn(3, 3)
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"✅ Tensor criado na GPU: {x.device}")
    else:
        print(f"⚠️  Tensor criado na CPU: {x.device}")
except Exception as e:
    print(f"❌ Erro ao criar tensor: {e}")

print("\n" + "="*60)