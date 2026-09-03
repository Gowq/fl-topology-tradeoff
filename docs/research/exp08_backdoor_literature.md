# Exp. 08 — ataques backdoor publicados para HFL e VFL

Data da pesquisa: 2026-07-21  
Escopo: selecionar um ataque moderno, reproduzível e respaldado pela literatura
para MHEALTH/OPPORTUNITY, sem apresentar uma adaptação própria como se fosse um
algoritmo publicado.

## Resposta curta

**Não foi encontrado um único algoritmo de backdoor clean-label publicado e
validado que possa ser executado literalmente sem adaptação em HFL e VFL.** As
interfaces e a autoridade do atacante são diferentes:

- no HFL, o cliente malicioso normalmente conhece os rótulos dos próprios
  exemplos, treina um modelo completo e transmite uma atualização de modelo;
- no VFL, um silo passivo controla somente suas próprias features/embeddings,
  não possui os rótulos e recebe gradientes dos embeddings.

Os próprios trabalhos de VFL apresentam essa diferença como a motivação para
novos ataques. Portanto, chamar de método conhecido a formulação anterior
“clean-label adaptativo, com os mesmos exemplos e o mesmo ataque nas duas
topologias” seria incorreto.

## Recomendação

Usar dois braços **topology-native**, sem tratá-los como uma comparação causal
direta da robustez HFL–VFL:

1. **VFL: LFBA**, exatamente como definido por Shen et al. (AAAI 2025). É a
   escolha mais forte para este projeto porque é label-free, possui código
   oficial e foi validado explicitamente em UCI-HAR.
2. **HFL: Label-Consistent Backdoor Attack**, de Turner, Tsipras e Mądry, como
   referência clean-label estabelecida e com código oficial. Para uma ameaça
   especificamente federada, DBA ou Model Replacement podem ser incluídos como
   baselines publicados, mas eles usualmente alteram rótulos e/ou atualizações e
   não são equivalentes ao LFBA.

O Exp. 08 pode responder “cada topologia resiste a um ataque moderno e plausível
para sua própria superfície?”. Ele **não** deve usar a diferença de ASR entre
esses dois braços para declarar que uma topologia é superior. A comparação
HFL–VFL controlada continua apoiada pelos ataques pareados do Exp. 07.

## Candidato recomendado para VFL: LFBA

Fonte primária: Wei Shen et al., *Label-Free Backdoor Attacks in Vertical
Federated Learning*, AAAI 2025 ([artigo e
DOI](https://doi.org/10.1609/aaai.v39i19.34246)).  
Código dos autores: [`shentt67/LFBA`](https://github.com/shentt67/LFBA).

### Ameaça e autoridade

- Um cliente VFL passivo é malicioso.
- Ele pode manipular seus dados locais e parâmetros locais.
- Ele não conhece nem altera os rótulos mantidos pelo cliente ativo.
- O objetivo é manter a acurácia limpa e fazer entradas com o trigger serem
  classificadas na classe da amostra âncora.

### Algoritmo publicado

1. O atacante escolhe localmente uma amostra âncora; não precisa conhecer seu
   rótulo. A classe desconhecida da âncora define o alvo do backdoor.
2. **Gradient-Guided Poison-Set Construction (GPC):** ordena/seleciona amostras
   cujos gradientes de embedding são mais consistentes com o gradiente da
   âncora. Isso produz um conjunto majoritariamente pertencente à classe-alvo
   sem revelar os rótulos.
3. O trigger fixa um pequeno conjunto de dimensões locais em um valor
   predefinido.
4. **Selectively Sample Switching (SAW):** entre os exemplos do poison set,
   seleciona os “hard samples” de maior gradiente, troca suas features locais
   pelas de outras amostras e adiciona o trigger. A troca reduz a capacidade de
   aprender as features originais e força associação trigger–alvo.
5. Nenhum rótulo é modificado.

No artigo, `poison_rate = Np/N` varia entre 0,1 e 0,3; o exemplo oficial usa
`poison_rate=0.1`, `select_replace` e um `select_rate` próprio do dataset. Os
modelos são treinados até convergência; UCI-HAR usa Adam, batch 256 e learning
rate 0,003. Esses valores são referências para o gate, não autorização para
tuning no teste.

### Validação publicada em HAR

O artigo avalia UCI-HAR (seis atividades) em VFL com 2 e 4 clientes. As features
são divididas igualmente, e o trigger fixa algumas dimensões do silo atacante.
Na Tabela 1, LFBA registra:

| UCI-HAR | acurácia limpa | ASR |
|---|---:|---:|
| 2 clientes | 91,99% | 99,96% |
| 4 clientes | 90,69% | 90,63% |

O artigo também varia a dimensão do trigger e testa compressão de gradientes e
ruído Gaussiano. Essas defesas reduzem o ASR apenas junto com degradação da
tarefa principal. Isso não equivale aos regimes formais de DP do nosso estudo.

### Métricas e gate de reprodução

Antes de MHEALTH/OPPORTUNITY, reproduzir o caminho UCI-HAR do código oficial.
Depois, portar apenas a entrada/dimensionalidade do dataset, mantendo GPC e SAW.
O gate deve registrar:

- acurácia/F1 macro limpa;
- ASR publicado (“A”) para comparabilidade com o artigo;
- `ASR_non_target`, calculado somente em exemplos cujo rótulo verdadeiro não é
  a classe-alvo, para evitar inflação pela prevalência da classe-alvo;
- taxa de consistência do poison set com a classe da âncora, apenas para
  auditoria offline;
- dimensões, valor e norma do trigger, `poison_rate`, `select_rate`, seed e
  identificação da âncora.

Escolher âncora, classe-alvo e parâmetros somente com treino/validação. O teste
fica reservado à avaliação final. Se a reprodução UCI-HAR falhar, o ataque não
entra na matriz completa.

### Limitações para MHEALTH/OPPORTUNITY

- LFBA foi validado em UCI-HAR com features tabulares pré-computadas, não em
  sinais brutos multissensor temporais como os nossos.
- A seleção das dimensões/valor do trigger precisa ser instanciada para cada
  representação. Isso é uma portabilidade de dataset e deve ser declarada, não
  vendida como novo ataque.
- O ataque depende de gradientes individuais de embeddings; batching, clipping,
  DP e ocultação desses gradientes podem prejudicar o GPC.
- O artigo testa 2/4 clientes. A eficácia pode cair quando mais silos dividem as
  features; essa é uma limitação declarada pelos autores.

## Opção HFL clean-label: Label-Consistent Backdoor

Fonte primária: Turner, Tsipras e Mądry, *Label-Consistent Backdoor Attacks*
([artigo](https://arxiv.org/abs/1912.02771)).  
Código dos autores:
[`MadryLab/label-consistent-backdoor-code`](https://github.com/MadryLab/label-consistent-backdoor-code).

O método conserva os rótulos: seleciona exemplos da classe-alvo, torna suas
features mais difíceis de aprender por perturbação adversarial ou interpolação
via GAN e adiciona um trigger. No teste, o ASR é a fração de exemplos
originalmente fora da classe-alvo que passam a ser classificados como alvo com
o trigger.

É um método canônico e reproduzível, mas foi validado em visão centralizada, não
em HFL nem HAR. Executá-lo em um cliente HFL e adaptar sua perturbação a séries
temporais exige portabilidade adicional. Por isso, ele alinha a família
clean-label, mas não elimina a perda de comparabilidade com LFBA.

## Baselines HFL especificamente federados

- **DBA — Distributed Backdoor Attack**, Xie et al., ICLR 2020
  ([artigo](https://openreview.net/forum?id=rkgyS0VFvr),
  [código dos autores](https://github.com/AI-secure/DBA)): divide um trigger
  global em subtriggers distribuídos entre clientes maliciosos. É reconhecido e
  reproduzível, mas tipicamente usa rótulo-alvo e não é clean-label/label-free.
- **Model Replacement**, Bagdasaryan et al., AISTATS 2020
  ([artigo](https://proceedings.mlr.press/v108/bagdasaryan20a.html),
  [código dos autores](https://github.com/ebagdasa/backdoor_federated_learning)):
  treina um modelo local backdoored e escala sua atualização para substituir o
  modelo global. Também supõe a superfície de atualização exclusiva do HFL.

Eles podem satisfazer a cobertura de ameaças HFL, mas devem aparecer como
ameaças topology-native separadas, não como “o mesmo ataque” usado no VFL.

## Outros candidatos VFL avaliados

| Método | Evidência | Motivo de não ser a primeira escolha |
|---|---|---|
| BadVFL (Naseri et al., IEEE S&P 2024) | [artigo](https://arxiv.org/abs/2304.08847); clean-label com inferência de rótulo e otimização no embedding | Sem código oficial localizado; validado em visão/Criteo, não HAR; seleção do round de troca é sensível. |
| VILLAIN (Bai et al., USENIX Security 2023) | [artigo](https://www.usenix.org/conference/usenixsecurity23/presentation/bai); inferência de rótulo, trigger aditivo e augmentation | Forte e publicado em venue de segurança, mas sem implementação oficial localizada e não validado em HAR. |
| BadVFL (Xuan et al., ECML PKDD 2023) | [artigo](https://arxiv.org/abs/2306.10746), [código dos autores](https://github.com/xuanyx/BadVFL); SDD+SDP, 1% de poisoning e ASR >93% nos quatro datasets publicados | Código disponível, porém sem HAR; requer uma amostra-alvo previamente conhecida e o LFBA remove também essa informação auxiliar. |
| TECB (Chen et al., ICDM 2023) | [DOI](https://doi.org/10.1109/ICDM58522.2023.00013); clean-label e pouca informação de classe-alvo | Validado somente em visão e o repositório de código indicado no trabalho não estava acessível durante esta pesquisa. |

## Consequência para o desenho do Exp. 08

**Atualização em 2026-07-22:** esta recomendação foi superada para o desenho
final do Exp. 08. O experimento usa TimeTrojan-FGSM transferível gerado antes da
partição federada, com as mesmas janelas envenenadas congeladas para HFL e VFL.
Esse modelo upstream remove a assimetria de autoridade entre cliente HFL e silo
VFL que impedia uma comparação causal usando LFBA contra ataques HFL-native. A
recomendação LFBA/Label-Consistent abaixo fica preservada como contexto
bibliográfico, mas não governa a matriz executada.

Uma matriz única pode conter os dois braços, com datasets, splits, rounds, DP,
seeds e critérios de gate comuns. Contudo, os resultados devem ser apresentados
em painéis separados:

- **HFL sob ataque HFL-native**;
- **VFL sob LFBA**;
- sem calcular `ASR_VFL - ASR_HFL` como efeito de topologia;
- sem declarar “VFL/HFL é mais robusto” a partir desses ataques distintos.

A conclusão defensável é sobre **exposição a ameaças práticas próprias de cada
topologia**. A conclusão comparativa sobre topologia exige os ataques pareados
e semanticamente controlados do Exp. 07.
