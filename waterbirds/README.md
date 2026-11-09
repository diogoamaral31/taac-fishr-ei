# Waterbirds (Pessoa C)

Limitação 1 em imagens reais: sem etiquetas de ambiente, o Fishr com divisão aleatória
colapsa para o ERM. Só se treina a última camada (regressão logística) sobre features fixas
da ResNet-50 (ImageNet, 2048-d).

## Instalar (ambiente principal, Python 3.10+)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install wilds pandas matplotlib pillow jupyter
```

## Correr, por ordem
```
python waterbirds/download_data.py          # ~2 GB para data/
python waterbirds/extract_features.py       # único passo pesado (GPU)
python waterbirds/train_last_layer.py --quick   # teste rápido do pipeline
python waterbirds/train_last_layer.py       # grelha completa, 5 seeds -> results/raw/waterbirds.csv
```
Depois abrir `analysis.ipynb` (gera a tabela em `results/figures/`).

## Grupos
`g = 2*y + place`: g0 terrestre/terra, g1 terrestre/água (minoritário),
g2 aquático/terra (minoritário), g3 aquático/água.
Treino (4795): 3498 / 184 / 56 / 1057.

## Regra de construção dos ambientes (para a professora)
Ambientes verdadeiros (`--minority_frac_env2`, por defeito 0.9), por seed:
1. Cada grupo é baralhado com uma seed fixa (1000 + seed).
2. Grupos maioritários (g0, g3): 50% para o ambiente 1, 50% para o ambiente 2.
3. Grupos minoritários (g1, g2): 10% para o ambiente 1, 90% para o ambiente 2.

Efeito: ambiente 1 com atalho muito forte (~99% fundo = tipo de pássaro) e ambiente 2 com
atalho mais fraco (~90%). É a variação da força do atalho entre ambientes que o Fishr usa.
Ambientes aleatórios: o treino inteiro baralhado e dividido em duas metades, sem usar
grupos, logo ambos os ambientes têm a mesma força de atalho (~95%).

## Métricas
- `worst_group_acc`: mínima exatidão nos 4 grupos do teste.
- `test_acc`: média das exatidões por grupo ponderada pela distribuição dos grupos no TREINO
  (convenção do group_DRO/DFR; o teste é balanceado por grupo, a média simples seria enganadora).
  A média simples fica em `grid_val.csv` (`test_acc_unweighted`).
- Seleção de modelo: lambda e l2 pela worst-group accuracy da validação (média das seeds).
  Grelha: lambda em {1e1..1e5}, l2 em {1e-4, 1e-3, 1e-2}. Hiperparâmetros NÃO são os do artigo
  (o artigo do Fishr não tem Waterbirds).

## Validação do pipeline
O ERM tem de mostrar exatidão média alta e worst-group muito mais baixa. Para um número
externo: ERM com features fixas no artigo do DFR (citar a tabela exata depois de confirmar
no PDF; não copiar de memória).
