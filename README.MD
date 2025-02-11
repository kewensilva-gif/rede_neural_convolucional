# Projeto de CNN com PyTorch

Este repositório contém um projeto de rede neural convolucional (CNN) utilizando PyTorch para classificação de imagens. O código inclui manipulação de imagens, carregamento de datasets e treinamentos de modelos.

## 📌 Requisitos

Certifique-se de ter o Python instalado (recomendado: Python 3.8+).

## 🚀 Configurando o Ambiente Virtual

Siga os passos abaixo para configurar e rodar o projeto corretamente:

### 1️⃣ Criar um ambiente virtual

No terminal ou prompt de comando, execute:

```sh
python -m venv venv
```

### 2️⃣ Ativar o ambiente virtual

- **Windows (CMD)**:
  ```sh
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```sh
  venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```sh
  source venv/bin/activate
  ```

### 3️⃣ Instalar dependências

Com o ambiente virtual ativado, instale as dependências do projeto executando:

```sh
pip install -r requirements.txt
```

### 4️⃣ Executar o projeto

Se o projeto possui um script principal (por exemplo, `train.py` ou `main.py`), execute:

```sh
python main.py
```

## 📂 Estrutura do Projeto

```
/project_root
│── cnn.py           # Definição do modelo CNN
│── dataset/         # Diretório para armazenar os dados
│── main.py          # Arquivo principal para treinar a CNN
│── requirements.txt # Lista de dependências
│── README.md        # Documentação do projeto
│── venv/            # Ambiente virtual (não incluso no repositório)
```

## 🔧 Dependências

O projeto utiliza as seguintes bibliotecas:

```
torch
torchvision
numpy
pillow
opencv-python
matplotlib
```

Caso precise adicionar novas dependências, instale-as com `pip install` e atualize `requirements.txt`:

```sh
pip freeze > requirements.txt
```

## 📌 Observações
- Certifique-se de que os dados estão organizados corretamente antes de executar o treinamento.
- Ajuste hiperparâmetros e estrutura da rede no arquivo `cnn.py` conforme necessário.

🚀 Bom desenvolvimento!

