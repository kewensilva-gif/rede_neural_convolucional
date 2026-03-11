# VisionNet CNN — Classificação de Imagens com Rede Neural Convolucional

> CNN em PyTorch para classificação de imagens RGB 64x64 em 5 classes, com módulo adicional de detecção e segmentação de objetos por cor usando OpenCV.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Classes do Dataset](#classes-do-dataset)
- [Arquitetura da CNN](#arquitetura-da-cnn)
- [Scripts](#scripts)
- [Módulo de Detecção e Segmentação](#módulo-de-detecção-e-segmentação)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura de Pastas](#estrutura-de-pastas)

---

## Sobre o Projeto

**VisionNet CNN** (nome sugerido para substituir "rede_neural_convolucional") é um projeto de visão computacional com dois componentes:

1. **CNN para classificação** — Rede neural convolucional que classifica imagens 64×64 RGB em 5 categorias de formas/símbolos
2. **Detecção e segmentação** — Pipeline baseada em OpenCV para detectar objetos avermelhados via máscara HSV, operações morfológicas e extração de contornos

---

## Classes do Dataset

| Classe | Label | Descrição |
|---|---|---|
| 0 | Lâmpada/Balão | Formas de lâmpada |
| 1 | Colcheias | Notas musicais (oitavas) |
| 2 | Floco | Flocos de neve |
| 3 | Hélice | Hélices/propulsores |
| 4 | TV | Televisores |

---

## Arquitetura da CNN

```
Input (3×64×64)
    ↓
Conv2d(3→16, 3×3) → ReLU → MaxPool(2×2)     → 16×32×32
    ↓
Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)    → 32×16×16
    ↓
Flatten                                        → 8192
    ↓
Linear(8192→64) → ReLU
    ↓
Linear(64→5) → Output
```

| Camada | Tipo | Saída |
|---|---|---|
| Conv1 | Conv2d(3, 16, kernel=3, pad=1) + ReLU + MaxPool | 16×32×32 |
| Conv2 | Conv2d(16, 32, kernel=3, pad=1) + ReLU + MaxPool | 32×16×16 |
| FC1 | Linear(32×16×16, 64) + ReLU | 64 |
| FC2 | Linear(64, num_classes) | 5 |

### Normalização

Valores pré-calculados do dataset de treino:
- **Mean:** `[0.813, 0.537, 0.564]`
- **Std:** `[0.124, 0.256, 0.253]`

---

## Scripts

| Script | Descrição |
|---|---|
| `cnn.py` | Definição da arquitetura CNN (`nn.Module`) |
| `main.py` | Treinamento (Adam, lr=0.001, 10 épocas, batch=8), salva `modelo.pth` |
| `evaluation.py` | Avaliação da acurácia no conjunto de treino |
| `test.py` | Inferência em imagens individuais + renomeia com a classe predita |
| `calc_normalize.py` | Calcula mean e std por canal do dataset para normalização |

### Parâmetros de Treinamento

| Parâmetro | Valor |
|---|---|
| Otimizador | Adam |
| Learning Rate | 0.001 |
| Loss | CrossEntropyLoss |
| Épocas | 10 |
| Batch Size | 8 |
| Imagem | 64×64 RGB |
| Device | CUDA (fallback CPU) |

---

## Módulo de Detecção e Segmentação

O diretório `detect-segment/` contém um pipeline separado de visão computacional:

| Etapa | Técnica |
|---|---|
| **Detecção** | Máscara HSV para objetos avermelhados |
| **Denoising** | Filtro mediano (median blur) |
| **Morfologia** | Operação de fechamento (closing) |
| **Segmentação** | Extração de contornos |
| **Exportação** | Imagens coloridas, binárias e view-binárias por objeto |

Processa 12 imagens de entrada e exporta os objetos segmentados individualmente.

---

## Instalação

### 1. Criar ambiente virtual

```sh
python -m venv venv
```

### 2. Ativar o ambiente virtual

- **Linux/macOS:**
  ```sh
  source venv/bin/activate
  ```
- **Windows (CMD):**
  ```sh
  venv\Scripts\activate
  ```

### 3. Instalar dependências

```sh
pip install -r requirements.txt
```

---

## Uso

### Treinar o modelo

```sh
python main.py
```

Treina a CNN e salva os pesos em `modelo.pth`.

### Avaliar acurácia

```sh
python evaluation.py
```

### Classificar imagens de teste

```sh
python test.py
```

Classifica imagens em `dataset-test/` e renomeia com o label predito.

### Calcular normalização do dataset

```sh
python calc_normalize.py
```

### Detecção e segmentação

```sh
cd detect-segment
python det_seg.py
```

---

## Estrutura de Pastas

```
VisionNet-CNN/
├── cnn.py                  # Arquitetura CNN (2 conv + 2 FC)
├── main.py                 # Script de treinamento
├── evaluation.py           # Avaliação de acurácia
├── test.py                 # Inferência + renomeação
├── calc_normalize.py       # Cálculo de mean/std do dataset
├── modelo.pth              # Pesos do modelo treinado
├── requirements.txt        # Dependências (torch, torchvision, numpy, pillow, opencv, matplotlib)
├── LICENSE                 # MIT License
├── dataset-train/          # Dados de treino (ImageFolder)
│   ├── 0_lampada/
│   ├── 1_colcheias/
│   ├── 2_floco/
│   ├── 3_helice/
│   └── 4_tv/
├── dataset-test/           # Imagens de teste (25 PNGs)
└── detect-segment/         # Módulo de detecção/segmentação
    ├── det_seg.py          # Pipeline OpenCV (HSV + morfologia + contornos)
    └── imagens/            # 12 imagens de entrada + pastas de saída
```

---

## Tecnologias

| Biblioteca | Uso |
|---|---|
| **PyTorch** | Framework de deep learning |
| **torchvision** | Transforms, ImageFolder, DataLoader |
| **OpenCV** | Detecção e segmentação por cor (HSV) |
| **NumPy** | Computação numérica |
| **Matplotlib** | Visualização |
| **Pillow** | Processamento de imagens |

---

## Licença

Este projeto está sob a licença [MIT](./LICENSE). © 2025 Kéwen Silva.

