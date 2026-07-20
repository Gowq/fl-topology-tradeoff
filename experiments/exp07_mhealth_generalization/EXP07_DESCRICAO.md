# Experimento 07 — Generalização multimodal no MHEALTH

## 1. Objetivo

O Experimento 07 avalia se o comportamento observado anteriormente no dataset
OPPORTUNITY se generaliza para outro problema multimodal de reconhecimento de
atividades humanas. A pergunta central é como a topologia de aprendizado
federado — horizontal (HFL) ou vertical (VFL) — condiciona a relação entre:

- utilidade preditiva;
- privacidade diferencial;
- número de participantes;
- ataques adversariais;
- agregação robusta;
- redundância entre modalidades físicas.

O experimento está finalizado. A matriz possui **510/510 configurações válidas**,
com cinco seeds por célula, 25 rodadas fixas, execução CUDA e métricas finitas.

## 2. Dataset e separação dos dados

Foi utilizado o **MHEALTH**, um dataset multimodal de Human Activity
Recognition com:

- 10 sujeitos;
- 12 classes de atividade;
- 23 canais;
- frequência de 50 Hz;
- três dispositivos físicos sincronizados.

As modalidades são:

| Modalidade | Canais | Papel no VFL |
|---|---:|---|
| Peito | 5 | Um silo de atributos |
| Tornozelo esquerdo | 9 | Um silo de atributos |
| Braço/punho direito | 9 | Um silo de atributos |

A separação foi feita por pessoa:

- sujeitos 1–8: treinamento;
- sujeito 9: tuning e conjunto-raiz confiável do FLTrust;
- sujeito 10: teste final intocado.

A normalização é ajustada somente nos sujeitos de treinamento. As janelas têm
128 amostras, equivalentes a 2,56 segundos, com stride 64 e 50% de sobreposição.
Janelas de transição com menos de 80% de concordância de rótulo são removidas.

## 3. Topologias comparadas

### HFL — Horizontal Federated Learning

Cada cliente representa uma pessoa distinta e possui todas as modalidades. Os
clientes treinam cópias locais do mesmo modelo e enviam atualizações de pesos ao
servidor. A escala usa sujeitos reais e aninhados, com `N = {2, 4, 8}`; não são
criados clientes sintéticos.

### VFL — Vertical Federated Learning

As mesmas amostras são divididas por atributos. Cada dispositivo físico é um
silo com encoder próprio. Os embeddings produzidos pelos silos são concatenados
por um coordenador, que executa a classificação final.

### Controles adicionais

- `centralized`: todos os 23 canais em um único encoder;
- `vfl`: três silos correspondentes aos dispositivos físicos;
- `vfl_random`: canais divididos aleatoriamente em três silos;
- `vfl_leave_chest`: VFL sem os canais do peito;
- `vfl_leave_left_ankle`: VFL sem o tornozelo esquerdo;
- `vfl_leave_right_arm`: VFL sem o braço direito.

## 4. Modelo e treinamento

Cada grupo de canais passa por um encoder com convoluções 1D, GroupNorm, ReLU,
pooling, dropout e uma projeção linear. O coordenador concatena os embeddings e
usa duas camadas lineares para produzir as 12 classes.

Parâmetros comuns:

- 25 rodadas fixas;
- uma época local por rodada;
- batch size 64;
- SGD com momentum 0,9;
- clipping com norma máxima 5,0;
- cinco seeds: `42`, `123`, `456`, `789` e `2026`;
- sem early stopping.

O tuning avaliou 72 execuções, combinando:

- learning rate: `{0,001; 0,003; 0,01}`;
- dropout: `{0,1; 0,25}`;
- dimensão oculta: `{64; 128}`;
- três seeds de tuning;
- HFL e VFL separadamente.

Configurações escolhidas:

| Topologia | Learning rate | Dropout | Dimensão oculta |
|---|---:|---:|---:|
| HFL | 0,01 | 0,1 | 128 |
| VFL | 0,01 | 0,1 | 64 |

O sujeito de teste não foi usado no tuning.

## 5. Privacidade diferencial

A privacidade diferencial usa **DP-SGD com Opacus** e contador RDP.

Parâmetros principais:

- `delta = 1e-5`;
- norma máxima por amostra: `5,0`;
- multiplicador de ruído calibrado para o número exato de passos;
- composição ao longo das 25 rodadas;
- `epsilon = 0` significa treinamento sem DP.

### DP no HFL

Cada cliente treina localmente com `PrivacyEngine.make_private`. O sample rate
conservador é o maior entre os clientes. O orçamento é calibrado para todos os
passos locais de todas as rodadas.

### DP no VFL

Cada encoder e o coordenador recebem gradientes por amostra. O passo DP é
aplicado manualmente com clipping conjunto por amostra e ruído Gaussiano antes
do `optimizer.step`.

Uma mesma pessoa atravessa todos os mecanismos VFL. Por isso, a contabilidade
faz composição sequencial de:

- três encoders + coordenador no VFL físico: quatro mecanismos por passo;
- um encoder + coordenador no centralizado: dois mecanismos;
- dois encoders + coordenador nas ablações leave-one-out: três mecanismos.

Essa composição garante que HFL e VFL sejam comparados pelo orçamento total por
pessoa, e não por um ε independente para cada silo. Ela também torna o VFL mais
sensível a orçamentos muito restritos.

Budgets avaliados:

| Braço | Valores de ε |
|---|---|
| Generalização | `0`, `0,25`, `0,5`, `0,75`, `1`, `3`, `8`, `20` |
| Escala | `0`, `3`, `20` |
| Robustez | `0`, `3`, `20` |
| Cauda | `50`, `100`, `150`, `200` |
| Redundância | `0`, `3`, `20` |

O gasto final observado ficou entre 96,28% e aproximadamente 100% do
epsilon-alvo.

## 6. Ataques avaliados

### Model Replacement

Ataque disponível no HFL. Após o treinamento local, a atualização maliciosa é
multiplicada por:

```text
número total de clientes / número de clientes maliciosos
```

O objetivo é fazer com que a média agregada substitua o modelo global pela
direção proposta pelos clientes maliciosos, em vez de apenas deslocá-lo.

Foi usado:

- no braço de escala, com razão maliciosa de 50%;
- no braço de robustez, com razões de 25% e 50%;
- com FedAvg, FLTrust e FoolsGold;
- sob `ε = {0, 3, 20}`.

### Clean-label Sensor Backdoor

Ataque aplicável ao HFL e ao VFL. O rótulo não é alterado. Nos exemplos da
classe-alvo, o ataque adiciona um trigger localizado:

- modalidade: tornozelo esquerdo;
- canais: giroscópio, índices 3–5 dentro da modalidade;
- região: 20% final da janela temporal;
- amplitude adicionada: `+4,0`;
- classe-alvo: classe codificada como `0`.

No teste de ASR, o trigger é aplicado a exemplos que não pertencem à
classe-alvo. A métrica mede a fração desses exemplos que passa a ser prevista
como classe `0`.

Foram usadas razões maliciosas de 25% e 50%, seleção reprodutível por seed e
budgets `ε = {0, 3, 20}`.

## 7. Agregadores e coordenador

### FedAvg

Média das atualizações HFL ponderada pelo número de amostras de cada cliente.
É o baseline de agregação.

### FLTrust

Usado apenas no HFL. O sujeito 9 produz uma atualização-raiz confiável. Cada
update de cliente recebe confiança baseada no cosseno positivo em relação à
raiz e é normalizado pela norma da atualização-raiz. Se nenhuma atualização
tiver confiança positiva, o servidor aplica atualização zero.

### FoolsGold

Usado apenas no HFL. Mantém o histórico acumulado das atualizações, calcula
similaridade de cosseno entre clientes e reduz o peso de clientes com direções
muito semelhantes. A implementação inclui pardoning e transformação logit dos
pesos.

### Coordenador VFL

O coordenador não é um agregador de updates homólogos. Ele recebe embeddings de
modalidades diferentes e aprende uma cabeça de fusão. FLTrust e FoolsGold não
são aplicados ao VFL porque os encoders dos silos representam espaços de
atributos distintos.

## 8. Matriz experimental

| Braço | Objetivo | Configurações |
|---|---|---:|
| Generalização | Curva HFL/VFL × DP | 80 |
| Escala | HFL com 2, 4 e 8 pessoas, limpo e model replacement | 90 |
| Robustez HFL | 2 ataques × 2 razões × 3 agregadores × 3 budgets | 180 |
| Robustez VFL | Sensor backdoor × 2 razões × 3 budgets | 30 |
| Cauda | HFL/VFL × ε alto | 40 |
| Redundância | Centralizado, VFL físico/aleatório e três ablações | 90 |
| **Total** |  | **510** |

Cada configuração grava um JSON atômico e retomável. As execuções foram
distribuídas entre GridUNESP e Pegasus, com validação de completude e integridade
antes da análise final.

## 9. Métricas

- acurácia;
- F1 macro;
- loss;
- ε efetivamente gasto por rodada;
- Attack Success Rate do sensor backdoor;
- pesos do FoolsGold, quando aplicável;
- duração da configuração;
- identificadores dos sujeitos maliciosos.

As comparações principais usam médias sobre cinco seeds e intervalos de
confiança de 95% com distribuição t.

## 10. Principais resultados

### Curva HFL × VFL

| ε | F1 HFL | F1 VFL | Diferença VFL − HFL |
|---:|---:|---:|---:|
| 0 | 0,671 | 0,983 | +0,312 |
| 0,25 | 0,108 | 0,014 | −0,094 |
| 0,5 | 0,224 | 0,014 | −0,210 |
| 0,75 | 0,315 | 0,014 | −0,301 |
| 1 | 0,392 | 0,014 | −0,378 |
| 3 | 0,682 | 0,701 | +0,019; inconclusivo |
| 8 | 0,729 | 0,909 | +0,180 |
| 20 | 0,721 | 0,964 | +0,242 |

O VFL é superior sem DP e a partir de ε=8. Com privacidade forte,
`ε = 0,25–1`, a composição dos mecanismos VFL exige ruído suficiente para
levar o modelo ao desempenho próximo do acaso. ε=3 forma uma região de
transição instável.

### Cauda de orçamento alto

| ε | F1 HFL | F1 VFL |
|---:|---:|---:|
| 50 | 0,711 | 0,966 |
| 100 | 0,698 | 0,958 |
| 150 | 0,689 | 0,962 |
| 200 | 0,686 | 0,969 |

VFL permanece em um patamar alto e supera HFL em todos os quatro budgets.

### Escala e modalidades

- O F1 HFL cresce em média de 2 para 8 sujeitos, mas os intervalos pareados
  ainda incluem zero.
- Retirar o tornozelo esquerdo reduz o F1 VFL em aproximadamente 0,213.
- Retirar o braço direito reduz o F1 em aproximadamente 0,106.
- Retirar o peito produz queda pequena e inconclusiva.
- A diferença entre VFL físico e VFL com partição aleatória não é conclusiva.

### Ataques e agregação

- Model replacement não reduziu consistentemente o F1 do FedAvg.
- O sensor backdoor apresentou ASR quase zero em FedAvg, FoolsGold e VFL.
- FoolsGold ficou essencialmente empatado com FedAvg.
- FLTrust foi comparável sem DP, mas colapsou quando combinado com DP.

Esses resultados indicam que os ataques ficaram fracos no protocolo atual.
Portanto, o braço adversarial **não deve ser apresentado como prova de
robustez** antes de auditar a efetividade dos ataques e a interação FLTrust+DP.

## 11. Comparação com o OPPORTUNITY

O formato qualitativo se repete: DP forte provoca colapso e a utilidade retorna
quando ε aumenta. Entretanto, no MHEALTH a recuperação VFL ocorre muito antes:
ε=3 já é uma transição e ε=8 apresenta F1 alto. No OPPORTUNITY, a recuperação
clara ocorria apenas na cauda de ε=50–200.

O resultado não sustenta uma vantagem universal da topologia. Ele sustenta uma
conclusão condicionada: VFL pode explorar muito bem modalidades complementares,
mas seu custo de composição DP pode dominar o benefício sob privacidade forte.

## 12. Limitações e próximos passos

- cinco seeds por célula e muitos contrastes, sem correção por múltiplas
  comparações;
- somente o sujeito 10 é usado como teste final;
- HFL e VFL foram otimizados separadamente e têm dimensões ocultas diferentes;
- os ataques precisam demonstrar efeito mensurável antes de avaliar defesas;
- a região `1 < ε < 3` deve ser densificada para localizar a transição VFL;
- recomenda-se repetir com múltiplos sujeitos de teste ou leave-one-subject-out;
- a interação entre FLTrust e updates com DP precisa de diagnóstico específico.

## 13. Estado final

- tuning: 72/72;
- matriz: 510/510;
- resultados válidos: 510;
- rodadas por configuração: 25;
- dispositivos: CUDA;
- jobs ativos no Grid/Pegasus: nenhum;
- experimento: **finalizado**.
