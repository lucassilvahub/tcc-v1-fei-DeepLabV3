# TCC - Segmentação Semântica da Mata Atlântica  

> **Processamento de Imagens RGB para Mapeamento de Cobertura e Uso do Solo**  
> **Centro Universitário FEI – Trabalho de Conclusão de Curso**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://lucassilvahub.github.io/tcc-fei/)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

## Sobre o Projeto  

Este trabalho de conclusão de curso (TCC) tem como objetivo desenvolver um **sistema de segmentação semântica** para análise de cobertura e uso do solo na **Mata Atlântica**.  

Por meio de **Deep Learning** e **Visão Computacional**, o sistema processa **imagens RGB** de alta resolução (cobrindo aproximadamente **60.000 hectares**) e gera **mapas temáticos** de uso e cobertura da terra.  

### Objetivo Principal  

Transformar imagens RGB em mapas temáticos por meio de **redes neurais convolucionais (CNNs)** aplicadas à segmentação semântica.  

---

## Classes de Uso e Cobertura do Solo  

O modelo foi treinado para identificar **8 categorias principais**:  

| Classe              | Cor Padrão (RGB) | Exemplo Visual |
|---------------------|------------------|----------------|
| Mata Nativa         | Verde Escuro (#006400)       | ![#006400](https://via.placeholder.com/20/006400/006400.png) |
| Vegetação Densa     | Verde Claro (#00FF00)        | ![#00FF00](https://via.placeholder.com/20/00FF00/00FF00.png) |
| Ocupação Urbana     | Cinza (#808080)              | ![#808080](https://via.placeholder.com/20/808080/808080.png) |
| Solo Exposto        | Marrom (#A0522D)             | ![#A0522D](https://via.placeholder.com/20/A0522D/A0522D.png) |
| Corpos d’Água       | Azul (#0000FF)               | ![#0000FF](https://via.placeholder.com/20/0000FF/0000FF.png) |
| Agricultura         | Amarelo (#FFFF00)            | ![#FFFF00](https://via.placeholder.com/20/FFFF00/FFFF00.png) |
| Regeneração         | Verde Amarelado (#ADFF2F)    | ![#ADFF2F](https://via.placeholder.com/20/ADFF2F/ADFF2F.png) |
| Sombra              | Preto (#000000)              | ![#000000](https://via.placeholder.com/20/000000/000000.png) |

---

## Resultados Preliminares  

- **Acurácia média**: 83%  
- **Área total processada**: ~60.000 hectares  
- **Região de estudo**: Serra de Petrópolis (RJ)  
- **Classes identificadas**: 8 categorias  

---

## Dashboard Online  

Acesse o dashboard interativo do projeto:  
**[lucassilvahub.github.io/tcc-fei](https://lucassilvahub.github.io/tcc-fei/)**  

---

## Links Relacionados  

- [Repositório ICMBio (base de referência)](https://github.com/fabricioifc/icmbio)  
- [Encoders disponíveis no Segmentation Models](https://smp.readthedocs.io/en/latest/encoders.html)  
- [Documentos acadêmicos (FEI - SharePoint)](https://feiedu-my.sharepoint.com/my?id=%2Fpersonal%2Funiflsilva%5Ffei%5Fedu%5Fbr%2FDocuments%2FCollege%2FTCC&ga=1)  

---

## Referências  

- Xiao, Aoran, et al. *"Foundation models for remote sensing and earth observation: A survey."* **arXiv:2410.16602 (2024).**  
  Disponível em: [arXiv](https://arxiv.org/abs/2410.16602)  

---

## Contribuições  

Este é um projeto **acadêmico em desenvolvimento**.  
Sugestões, melhorias e feedbacks são muito bem-vindos.  

---

## Licença  

Este projeto está sob a **licença MIT**.  
Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.  

---

Desenvolvido com dedicação para a **preservação da Mata Atlântica**  
