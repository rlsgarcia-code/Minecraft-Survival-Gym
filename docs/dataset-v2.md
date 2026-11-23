# Dataset v2

`minecraft_gym.dataset` é a interface canônica de coleta, curadoria e treino
isolado. Ela contém contratos versionados, gates de admissão, recuperação de
WAL, hashes de artefatos, catálogo, snapshots e exportadores.

## Níveis de dados

- `exact_visual`: input nativo Java e RGB sincronizados. Somente o coletor Java
  dedicado pode produzir este nível.
- `rendered_visual`: ação Gym exata pareada à visão egocêntrica do bot no
  renderer Prismarine. É elegível para política somente com alinhamento de até
  um tick e deve manter o mesmo domínio visual no treino e no rollout.
- `exact_structured`: ação Gym exata e estado estruturado anterior/posterior.
  É o nível produzido por `DatasetRecorderV2`.
- `semantic_weak`: movimento/eventos autoritativos sem input nativo. Nunca é
  elegível para BC corporal, offline RL ou DreamerV3.
- `privileged_tool`: comandos e ferramentas, separados da política corporal.

Estado privilegiado pode calcular recompensa/avaliação, mas fica fora dos
canais da política.

## Ação de política e controle executado

`ActionLabelV2.values` preserva o controle efetivamente aplicado ao Minecraft.
Quando uma política usa uma ação de nível diferente, o mesmo rótulo também
registra `policy_action_contract` e `policy_action`. Para o actor-critic de
navegação v2, isso guarda sem perda `delta_longitude`, `delta_latitude`, os
eixos independentes `strafe`/`move` e os botões `jump`, `sprint`, `sneak` e
`attack` e `use` sob `minecraft-control-camera-keyboard/v3`, ao lado dos controles
realmente executados. O contrato histórico
`minecraft-control-hybrid-camera-body/v1` continua validável somente para
datasets e checkpoints antigos. Assim o dataset permite treinar e auditar a
política fatorada sem a interface do Observer inventar uma representação
paralela.

## Observação multivisão

Uma observação pode carregar vários `ViewpointFrameRefV1`. Cada frame declara
ator, stream, renderer, domínio visual, tick, sequência, checksum e qualidade
de alinhamento. `primary_stream_id` escolhe a câmera que corresponde à ação,
sem descartar as demais:

- ação humana nativa → câmera Fabric do demonstrador (`native_java`);
- ação Gym do bot → câmera Prismarine do bot (`rendered_prismarine`).

Assim, uma demonstração pode preservar simultaneamente o que o jogador viu e
o que o bot viu. A seleção da tela no Observer é apenas uma projeção de UI; o
episódio canônico continua contendo os dois streams. O helper
`assemble_multiview_observation(...)` aplica essa seleção no Gym e o
`EpisodeWriter` rejeita frames cuja origem não esteja declarada no manifesto.

As duas imagens não são tratadas como intercambiáveis. BC corporal usa como
primária a visão do ator da ação; a visão secundária pode apoiar aprendizado
multivisão, reward models e avaliação, sempre preservando seu domínio.

## Mapa espacial derivado

O contrato [`spatial-raycast-v1`](spatial-raycast-v1.md) associa pixels da
câmera Prismarine a blocos estruturados de `bot.world` e produz deltas
reproduzíveis de um lattice X/Z por ator. `ObservationRefV2.spatial_updates`
preserva conjuntamente o frame, pose, raios, contatos, tipos de bloco e
evidência do mapa.

O mapa é `privileged_derived` por padrão e não pode aparecer em
`policy_channels`. A interface do Observer apenas reproduz esses updates; ela
não mantém um mapa paralelo nem altera o conteúdo do dataset.

Uma memória futura construída exclusivamente de RGB-D implantável e odometria
estimada será um contrato separado; ela não poderá herdar evidências já
preenchidas por `bot.world`. A proposta, a proveniência exigida e sua montagem
como observação estão em
[`odometric-belief-state-v1.md`](odometric-belief-state-v1.md).

## Admissão

1. O manifesto cria um diretório `.partial` no volume de destino.
2. Cada passo e evento validado é acrescentado ao WAL com `fsync` periódico.
3. A finalização verifica ordem, ticks, consentimento, privacidade, checksums,
   alinhamento visual e perdas.
4. Os passos são materializados como `steps.parquet`; o JSONL permanece para
   auditoria e reconstrução exata.
5. Os arquivos recebem SHA-256 e o diretório completo é renomeado atomicamente
   para `raw/` ou `quarantine/`.

Captura visual requer volume APFS montado, gravável e com ao menos 15% livre.
Captura estruturada pode usar o spool limitado do sistema. Falta de volume,
Parquet, consentimento, privacidade ou sincronismo bloqueia ou coloca o episódio
em quarentena; nunca há rebaixamento silencioso.

## Gravador estruturado mínimo

```python
from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.dataset import DatasetRecorderV2, ResearchStorage

storage = ResearchStorage("/Volumes/MinecraftRL")
env = DatasetRecorderV2(
    MinecraftSurvivalEnv(backend="socket"),
    storage,
    environment_id="minecraft-local",
    world_id="training-clone",
    server_version="1.21",
)
```

Modos de treino e avaliação também exigem `world_snapshot_id` imutável. Os
helpers de isolamento rejeitam o mundo familiar e verificam vinte resets
determinísticos antes de liberar um clone.

## Coletor autoritativo do servidor 26.3

O módulo `fabric-server-26.3/` é um mod Fabric exclusivo do servidor. Ele
observa o tick autoritativo sem oferecer operações de controle ou reset e
escuta somente em `127.0.0.1:25571`.

O fluxo de leitura é retomável:

1. o cliente negocia `minecraft-gym/v2`, versão do Minecraft e capacidades;
2. cada tick recebe uma sequência monotônica e fica num buffer de 2.400 ticks;
3. `snapshot(after_sequence=...)` devolve somente ticks novos;
4. perda do início do buffer produz um erro de gap, nunca uma trajetória
   aparentemente contínua;
5. requisições e respostas carregam o JSON exato em um envelope assinado com
   SHA-256, sem depender de Python e Java formatarem floats da mesma maneira.

```python
from minecraft_gym.dataset import AuthoritativeCollectorClient

client = AuthoritativeCollectorClient()
try:
    hello = client.connect()
    batch = client.snapshot(limit=200)
    client.ack()
finally:
    client.close()
```

Os atores usam pseudônimos HMAC estáveis no servidor; nomes, chat e UUIDs não
saem do mod. Posição absoluta, rotação e dimensão são contexto privilegiado.
A projeção `semantic_steps_from_ticks(...)` expõe apenas deslocamento e deltas
de câmera como rótulos conhecidos, mascara teclado/mouse/toque e limita o
episódio ao consumidor `semantic_imitation`.

Esse coletor não transforma Android em demonstração corporal exata. Também não
substitui o coletor visual Java: RGB e input nativo sincronizados continuam
obrigatórios para `exact_visual`, BC de baixo nível e DreamerV3.

### RGB-D alinhado

Quando o adaptador de runtime fornece o raycast Prismarine pareado ao frame,
`ObservationRefV2.depth_map` usa `minecraft-depth-map/v1`. A grade preserva
distância métrica em blocos e, para cada amostra, separa contato de superfície,
raio censurado no alcance máximo e chunk ainda desconhecido. O canal `depth`
só pode aparecer em `policy_channels` quando o mapa possui o mesmo `frame_id` e
tick da observação RGB. Isso permite comparar RGB e RGB-D sobre exatamente os
mesmos episódios sem converter espaço desconhecido em profundidade zero.

O teste de integração usa `fabric-server-26.3/run/`, porta de jogo `25575` e
um mundo descartável. O diretório é ignorado pelo Git e nunca deve apontar para
o mundo familiar.
