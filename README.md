# taac-fishr-ei

TAAC, Checkpoint 2 (11 de novembro): limitação do Fishr sem etiquetas de ambiente e proposta Fishr-EI.
Divisão de tarefas no PDF desta pasta.

## Estrutura
```
external/    clones dos repos oficiais (fishr, eiil). NÃO editar, fora do git
shared/      results_io.py (formato único dos CSV), seeds.py, plotting.py
theory/      Pessoa A: formulação, literatura, protocolo, toy_2d.ipynb
cmnist/      Pessoa B: Colored MNIST
waterbirds/  Pessoa C: Waterbirds sobre features da ResNet-50
data/        dados e features, fora do git
results/     raw/ (um CSV por experiência), figures/ (PNG/PDF para os slides)
notebooks/   checkpoint2_summary.ipynb
```
Cada pessoa só escreve na sua pasta e em `results/`.

## Instalar
Ambiente principal (Python 3.10+), com conda:
```
conda env create -f environment.yml && conda activate taac-fishr-ei
```
ou com venv (Windows):
```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install wilds pandas matplotlib pillow tabulate jupyter ipykernel
```
Ambiente do repo oficial do Fishr (Pessoa B): `conda env create -f environment-fishr.yml`.

Repos externos:
```
git clone https://github.com/alexrame/fishr external/fishr
git clone https://github.com/ecreager/eiil external/eiil
```

## Correr, por ordem
1. Waterbirds (Pessoa C): ver [waterbirds/README.md](waterbirds/README.md).
2. Colored MNIST (Pessoa B): `cmnist/` (por preencher).
3. Resumo: `notebooks/checkpoint2_summary.ipynb` junta `results/raw/*.csv`.

Formato dos resultados: `dataset,method,setting,seed,train_acc,test_acc,worst_group_acc,partition_purity`.
