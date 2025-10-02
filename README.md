# Secular Dynamics System

Implementação em Python da teoria secular linear (Laplace–Lagrange), com base no Capítulo 7 de Murray & Dermott (2000) — *Solar System Dynamics*.  
Este repositório permite calcular a evolução secular de dois corpos perturbadores e de um corpo teste, gerando gráficos e dados intermediários.

---

## Estrutura do projeto

secular-dynamics-system-v1/
├── data/
│ ├── inputPlanets.json # Dados dos planetas (massa, a, e, I, ω, Ω)
│ └── constants.json # Constantes físicas (G, M0, AU, etc.)
├── output/ # Resultados gerados (csv, npz, gráficos)
├── src/
│ ├── init.py
│ ├── main.py # Script principal de execução
│ ├── secular.py # Cálculos seculares, matrizes, modos puros etc.
│ ├── plotting.py # Rotinas de plotagem (Fig. 7.1, modos puros)
│ └── inspect_output.py # Script para inspecionar o arquivo .npz de saída
├── requirements.txt # Dependências Python do projeto
└── README.md # Documento explicativo (este arquivo)


---

## Funcionalidades principais

- Leitura de parâmetros de entrada via arquivos JSON (`data/`)  
- Construção das matrizes seculares \(A\) (excentricidade) e \(B\) (inclinação)  
- Cálculo de autovalores/autovetores — frequências próprias \(g\), \(s\)  
- Cálculo das constantes modais \(C_k\), \(D_k\)  
- Reconstrução temporal completa (todos os modos)  
- Geração de séries temporais e salvamento em `.npz` e `.csv`  
- Plotagem da evolução secular (excentricidade e inclinação), reproduzindo estilo da Fig. 7.1  
- Inspeção interativa do arquivo `.npz` para visualização dos arrays  

---

## Dependências

As bibliotecas necessárias estão em `requirements.txt`. Exemplo:

numpy
matplotlib
scipy
pandas


Para instalar:

pip install -r requirements.txt

## Para rodar

python -m src.main

