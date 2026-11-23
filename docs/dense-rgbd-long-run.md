# RGB-D denso para treino longo

## O que é realmente denso

O sensor `minecraft-dense-rgbd-frame/v1` produz, no mesmo estado imutável do
renderer Prismarine:

- RGB JPEG em 640×360;
- 230.400 profundidades independentes em 640×360;
- estado por pixel (`hit`, `max_range` ou `unknown`);
- pose, projeção e identificador do frame compartilhados.

A profundidade é a distância métrica ao longo do eixo óptico da câmera, em
blocos, quantizada em centímetros. Ela não é um raycast esparso ampliado. O
browser renderiza a cena em um framebuffer com `DepthTexture`, lineariza o
depth do WebGL e comprime os valores antes do transporte.

O dataset guarda RGB e profundidade como artefatos separados, ligados por
`camera_frame_id`, tick, hash SHA-256 e `DenseDepthFrameRefV1`. A entrada da
política preserva cinco planos completos em 640×360: RGB, profundidade
normalizada e máscara de validade. Não há redimensionamento no encoder de
dados. A redução espacial é aprendida pelos blocos convolucionais VGG e essa
arquitetura fica registrada no checkpoint.

`VisualHybridActorCriticAgent` aceita o frame vivo por `_dense_rgbd_frame` e
usa o mesmo espaço de ação da Academia. A arquitetura visual é
`vgg-rgbd-shared-flatten-mlp/v1`: cinco blocos convolucionais reduzem o campo
espacial progressivamente, o último mapa é achatado e projetado por um MLP. O
backbone é compartilhado; actor e critic têm heads distintos. Não existe
pooling ou resize no encoder de dados.

Rollouts são atualizados em minibatches para que uma trajetória longa não
materialize centenas de tensores 640×360 simultaneamente no acelerador. O
checkpoint inclui arquitetura, forma nativa de entrada, optimizer e RNG.

## Gate obrigatório

Um treino longo só pode começar depois de um relatório
`minecraft-dense-rgbd-preflight/v1` com `status: passed`. O gate verifica:

1. nenhum erro ou checksum inválido;
2. resolução exatamente 640×360;
3. p95 dentro do orçamento do FPS escolhido;
4. sobrecusto de profundidade dentro do limite declarado;
5. espaço suficiente para toda a duração e a reserva mínima.

Exemplo para o modo conservador de 10 Hz:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_dense_rgbd.py \
  --samples 30 \
  --warmup 3 \
  --target-fps 10 \
  --p95-frame-budget-ms 100 \
  --p95-depth-overhead-ms 50 \
  --planned-hours 8 \
  --output runs/rgbd-preflight/dense-10hz/report.json
```

O FPS do relatório é um contrato do rollout, não apenas uma etiqueta. Em 10 Hz,
uma ação dura dois ticks do Minecraft. A transição continua registrando o tick
inicial/final para que action repeat não seja confundido com perda de frames.

## Limite atual e caminho para 20 Hz

A implementação síncrona usa `readRenderTargetPixels`. Ela é correta, mas o
readback GPU→CPU é o custo dominante. Portanto 20 Hz só deve ser habilitado se
um novo preflight de 50 ms passar. O próximo passo técnico para 20 Hz é readback
assíncrono com WebGL2 Pixel Buffer Objects e uma fila curta com alinhamento por
frame; reduzir a densidade ou interpolar raios não é uma substituição aceitável.

## O que este pipeline não afirma

- Ele não transforma a simulação procedimental da Academia em treino visual
  real do Minecraft.
- Ele não cria ainda um dataset RGB→depth; os pares são observações de RL.
- Ele não usa o mundo familiar para reset ou treino.
- Ele não considera a câmera do demonstrador como a câmera do bot.

O treino visual longo deve usar um clone isolado, o cliente RGB-D acima e o
recorder v2. Se qualquer gate falhar, a execução deve permanecer bloqueada em
vez de degradar silenciosamente para depth esparso ou RGB apenas.

## Rodada de engenharia de 500 episódios — 2026-11-22

A rodada `runs/vgg-rgbd-long/2026-11-22-stable-500ep-v1` validou a arquitetura
e o loop de otimização no proxy procedimental da Academia. Ela **não** foi um
treino no Minecraft real. Para tornar o ensaio local viável, o proxy gerou
cinco planos RGB-D em 160×90; o contrato do sensor real permanece 5×360×640.

O ensaio utilizou MPS, minibatches de oito transições e taxas separadas:
backbone `1e-5`, projeção compartilhada `3e-5`, actor `1e-4` e critic `3e-4`.
As vantagens foram normalizadas por episódio e os gradientes limitados a 1,0.
Foram executados 500 episódios de treino e 100 seeds inéditas de avaliação.

Resultados:

- melhor média móvel de reward em 25 episódios: −18,52 no episódio 452;
- maior sucesso móvel em 25 episódios: 60% no episódio 265;
- sucesso móvel final: 28%;
- checkpoint final em seeds inéditas: 14% de sucesso;
- checkpoint selecionado no episódio 450: 22% de sucesso em seeds inéditas.

A decomposição por dificuldade mostrou interferência catastrófica. O
checkpoint 450 obteve 32,8% nos casos fáceis e 0% nos moderados. O checkpoint
500 obteve 7,5% nos fáceis e 27,3% nos moderados. Logo, o treino produziu
competências, mas alternou especializações e esqueceu a anterior. Nenhum desses
checkpoints deve ser promovido ao BrincaBot.

O próximo gate algorítmico deve comparar a mesma entrada VGG RGB-D com updates
mais conservadores (por exemplo PPO com razão limitada) ou replay estratificado
por dificuldade. A seleção deve usar avaliação congelada balanceada, não apenas
reward recente de treino.

## PPO visual

`VisualHybridPpoAgent` preserva o backbone e os heads híbridos da política
visual, mas substitui a atualização Actor-Critic por PPO clipped. O buffer é
temporário e on-policy: acumula um número fixo de episódios, calcula os valores
e log-probabilidades da política antiga, deriva vantagens com GAE, executa
múltiplas épocas em minibatches e então descarta todas as transições.

Os controles de estabilidade registrados em cada rollout são:

- KL aproximado entre política antiga e nova;
- fração de amostras atingindo o clipping;
- razão média de probabilidades;
- entropia, losses e norma do gradiente antes do clipping;
- número de épocas concluídas e early-stop por KL.

O piloto calibrado usa rollouts balanceados de dez episódios, quatro épocas,
`clip_ratio=0.2`, `GAE lambda=0.95` e `target_kl=0.02`. As taxas são `2e-6`
para o backbone, `5e-6` para a projeção compartilhada, `1e-5` para os heads do
actor e `1e-4` para o value head. Nos dois primeiros rollouts, o KL foi 0,013 e
0,010, com 16% e 13% das amostras clipped e sem early-stop.

Durante essa implementação foi corrigido um erro anterior no head Von Mises:
a concentração angular mantinha uma dimensão extra e transformava batches de
log-probabilidade em matrizes B×B. Cada transição agora produz exatamente um
log-prob conjunto. Por isso, resultados antigos do AC não devem ser usados como
baseline quantitativo definitivo; permanecem úteis apenas como evidência do
problema de estabilidade que motivou o PPO.

### Rodada PPO longa — 500 episódios

A rodada `runs/vgg-rgbd-ppo-long/2026-11-22-500ep-v2` completou 500 episódios
de treino balanceado e 100 episódios de teste em seeds inéditas. Foram feitas
50 atualizações PPO. O KL médio foi 0,011, o máximo 0,016 e nenhuma atualização
precisou do early-stop. A fração média clipped foi 14%.

A média móvel de reward em 25 episódios passou de −22,14 na primeira janela
para −21,27 na última, atingindo −19,38 no episódio 431. O pico móvel de
sucesso foi 52% no episódio 114 e a janela final terminou em 32%, sem colapso.

Na avaliação congelada e balanceada, o checkpoint final obteve 38% de sucesso:
32% em tarefas fáceis e 44% em moderadas. A mediana foi −25,03 de retorno e 232
ticks entre os episódios bem-sucedidos.

O checkpoint 450, escolhido pelo antigo critério de reward recente de treino,
obteve 0% nas mesmas seeds. Isso confirmou que métricas on-policy de treino não
servem para selecionar a política. O runner agora usa validação congelada em
seeds separadas a cada checkpoint, priorizando sucesso e usando retorno mediano
como desempate. O teste final continua separado dessa validação.
