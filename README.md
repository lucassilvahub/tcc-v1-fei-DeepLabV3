# TCC - Segmentação Semântica da Mata Atlântica

> **Processamento de Imagens RGB para Mapeamento de Cobertura e Uso do Solo**  
> **Centro Universitário FEI – Trabalho de Conclusão de Curso**

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://lucassilvahub.github.io/tcc-fei/)  
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Sobre o Projeto

Este trabalho de conclusão de curso (TCC) tem como objetivo desenvolver um **sistema de segmentação semântica** para análise de cobertura e uso do solo na **Mata Atlântica**. Por meio de **Deep Learning** e **Visão Computacional**, o sistema processa **imagens RGB** de alta resolução (cobrindo aproximadamente **60.000 hectares**) e gera **mapas temáticos** de cobertura do solo.

### Objetivo Principal

Transformar imagens RGB em mapas temáticos por meio de **redes neurais convolucionais (CNNs)** aplicadas à segmentação semântica.

---

## Classes de Uso e Cobertura do Solo

O modelo foi treinado para identificar **8 categorias principais**:

| Classe              | RGB              | HEX         |
|---------------------|------------------|-------------|
| Mata Nativa         | (0, 100, 0)       | `#006400`   |
| Vegetação Densa     | (0, 255, 0)       | `#00FF00`   |
| Ocupação Urbana     | (128, 128, 128)   | `#808080`   |
| Solo Exposto        | (160, 82, 45)     | `#A0522D`   |
| Corpos d’Água       | (0, 0, 255)       | `#0000FF`   |
| Agricultura         | (255, 255, 0)     | `#FFFF00`   |
| Regeneração         | (173, 255, 47)    | `#ADFF2F`   |
| Sombra              | (0, 0, 0)         | `#000000`   |

*Obs.: os códigos RGB e HEX seguem o padrão do sistema de mapeamento temático e podem ser usados diretamente em visualizações ou legendas.*

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

- Xiao, Aoran, et al. _"Foundation models for remote sensing and earth observation: A survey."_ **arXiv:2410.16602 (2024).**  
  Disponível em: [arXiv](https://arxiv.org/abs/2410.16602)

---

## Contribuições

Este é um projeto **acadêmico em desenvolvimento**. Sugestões, melhorias e feedbacks são muito bem-vindos.

---

## Licença

Este projeto está sob a **licença MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

---

Desenvolvido com dedicação para a **preservação da Mata Atlântica**
