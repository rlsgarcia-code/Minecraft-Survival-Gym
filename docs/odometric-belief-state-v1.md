# Estado de crença odométrico para navegação

Status: **proposta de contrato; ainda não implementada como observação de
política**.

Esta especificação descreve como RGB-D, odometria, memória espacial, objetivo
e estado corporal devem formar uma observação treinável. Ela não muda o
contrato atual por documentação: a implementação só poderá anunciar a nova
observação depois dos gates de proveniência, replay e ablação descritos aqui.

## Objetivo

A política não deve receber coordenadas verdadeiras do servidor nem um mapa
extraído diretamente de `bot.world`. Ela deve construir uma crença sobre sua
posição e sobre o espaço a partir dos mesmos sensores disponíveis no rollout:

\[
b_t = (V_t, \hat{T}_t, \Sigma_t, M_t, g_t)
\]

onde:

- `V_t` é o RGB-D egocêntrico atual;
- `T_hat_t` é a pose estimada no referencial do episódio;
- `Sigma_t` é a incerteza da pose;
- `M_t` é a memória espacial derivada somente de sensores implantáveis;
- `g_t` é a crença sobre a posição do objetivo no mesmo referencial.

A odometria não deve entrar apenas como coordenadas concatenadas. Sua função
principal é alinhar observações sucessivas e permitir que o agente mantenha
uma memória espacial persistente quando um alvo ou obstáculo sai da imagem.

## Fronteira entre PanorAi, Minecraft Gym e Observer

### PanorAi

PanorAi é dono de geometria de câmera, associação visual, estimação geométrica
robusta e evidência de qualidade. Minecraft Gym deve consumir uma versão
fixada por commit, sem copiar o pacote para dentro do repositório.

Recursos públicos já disponíveis:

| Recurso | Uso no Minecraft | Situação |
|---|---|---|
| `panorai.geometry.GnomonicSpec` | Define resolução e FOV da câmera Prismarine | Disponível |
| `gnomonic_pixels_to_rays(...)` | Converte pixels RGB em bearings unitários | Disponível e usado |
| `rays_to_gnomonic_pixels(...)` | Reprojeção e auditoria geométrica | Disponível |
| `gnomonic_intrinsics(...)` | Intrínsecos equivalentes da câmera | Disponível |
| `camera_rays_to_world_points(...)` e helpers de frames | Transformações câmera/mundo quando o contrato exigir | Experimental; uso futuro controlado |
| `FeatureExtractorConfig` | Configuração declarativa de SIFT/ORB/AKAZE | Disponível |
| `FeatureMatcherConfig` | Ratio test, cross-check e matcher | Disponível |
| `OpenCVFeatureBackend` | Detecção, descrição e matching sem vazar objetos OpenCV | Disponível e usado no auditor vivo |
| `estimate_relative_pose(...)` | Rotação e direção de translação sem escala | Disponível; usado internamente como consenso visual |
| `RelativePoseOptions` e relatórios de qualidade | Configuração e diagnóstico do consenso visual | Disponível |
| `panorai.estimators.metric_rgbd_pose` | Pose SE(3) métrica a partir de bearings e depth radial | Experimental, [PR PanorAi #233](https://github.com/rlsgarcia-code/PanorAi/pull/233) validado pelo CI |
| `MetricRgbdPoseOptions` | Limiar métrico, trials, seed e otimização local | Disponível no módulo dedicado |
| `MetricRgbdPoseAcceptancePolicy` | Gate de inliers, resíduos e incerteza | Disponível no módulo dedicado |
| `MetricRgbdPoseResult` | `R`, `t`, máscaras, resíduos, covariância e proveniência | Disponível no módulo dedicado |
| `panorai.model_fitting.robust_fit(...)` | Infraestrutura MSAC/RANSAC reutilizável | Disponível; o Gym não deve chamá-la para duplicar o estimador RGB-D |

O RGB Prismarine é uma imagem gnomônica/pinhole. Por isso, o frontend atual
usa `OpenCVFeatureBackend` diretamente. `SphericalFeaturePipeline` e os
pipelines ERP não devem ser inseridos apenas porque pertencem ao PanorAi; eles
são úteis quando a entrada for efetivamente equiretangular ou um rig de faces.

`panorai.slam` já oferece trajetória visual, keyframes, mapa esparso e loop
closure, mas seu contrato atual é visual, arbitrário em escala e orientado a
câmeras centrais/fisheye calibradas. Ele não substitui diretamente a odometria
RGB-D métrica do Minecraft. Uma integração futura deve primeiro publicar um
adaptador métrico documentado ou uma fachada incremental RGB-D no PanorAi.

O Workbench do PanorAi pode inspecionar e exercitar as superfícies publicadas,
mas não faz parte do caminho de inferência do bot. O runtime importa a
biblioteca Python diretamente; o Workbench permanece ferramenta de pesquisa e
diagnóstico.

### Minecraft Gym

Minecraft Gym é dono de:

- captura renderer-atomic RGB-D e seus hashes;
- semântica `camera-Z` em blocos e máscara de validade;
- conversão `camera-Z -> radial range`;
- acumulação da pose e propagação de incerteza;
- fallback explícito baseado na ação executada;
- memória espacial, objetivo, estado corporal e espaço de ações;
- contrato da observação treinável e persistência no dataset;
- separação entre canais implantáveis e ground truth privilegiado.

O adaptador existente
`minecraft_gym.odometry.estimate_panorai_rgbd_odometry(...)` converte a
profundidade e mapeia o resultado PanorAi para o contrato do Gym. Ele não deve
ganhar uma segunda implementação de alinhamento rígido.

### Observer

O Observer reproduz a crença mantida pelo Gym e seus diagnósticos. Ele não
mantém uma pose, mapa ou vetor de estado paralelo. A UI pode mostrar RGB-D,
trajetória, covariância, memória e proveniência, mas toda informação treinável
vem dos contratos do Gym.

## Sensor visual canônico

O contrato atual de produção é
`minecraft-visual-policy-input/native-rgbd-v2`:

\[
V_t \in \mathbb{R}^{5\times360\times640}
\]

Os planos são:

1. vermelho em `[0,1]`;
2. verde em `[0,1]`;
3. azul em `[0,1]`;
4. depth `camera-Z / max_distance`;
5. validade do depth em `{0,1}`.

O quinto plano permanece necessário: zero no depth não pode significar ao
mesmo tempo contato próximo e ausência de medição. A redução espacial pertence
à CNN registrada no checkpoint, não ao coletor. Proxies menores continuam
permitidos somente em ensaios declarados.

## Atualização da crença de pose

Para correspondências de pixels `a <-> b`, o Gym usa PanorAi para obter
bearings `b_a`, `b_b`. A profundidade Prismarine mede camera-Z, então:

\[
r = \frac{z_{camera}}{b_z}, \qquad p = r b
\]

O estimador métrico retorna:

\[
p_b = R_{b\leftarrow a}p_a + t_{b\leftarrow a}
\]

Uma atualização aceita compõe a pose:

\[
R_{t+1}=R_{t+1\leftarrow t}R_t
\]

\[
t_{t+1}=R_{t+1\leftarrow t}t_t+t_{t+1\leftarrow t}
\]

A representação instantânea para a rede é o logaritmo de SE(3):

\[
\xi_t=\log(\Delta T_t)\in\mathbb{R}^6
\]

com três componentes de translação e três de rotação. Translações são
normalizadas por uma escala declarada em blocos; rotações, por `pi`. Ambos são
recortados antes da entrada da rede, mas os valores físicos completos continuam
no dataset.

Se a estimação visual for rejeitada:

1. a rejeição e seus motivos são persistidos;
2. o Gym aplica a previsão derivada do controle realmente executado;
3. a covariância aumenta;
4. escritas no mapa recebem peso menor ou são suspensas acima do limite;
5. a transição nunca é rotulada como odometria visual aceita.

## Memória espacial implantável

### Separação obrigatória do mapa privilegiado

`SparseVoxelWorldMemory` atualmente recebe raycasts estruturados de
`bot.world`, incluindo posição de bloco e `block_name`. Esse mapa é
`privileged_derived` e continua proibido nos canais do ator.

A memória treinável proposta é uma instância separada, construída somente por:

- RGB-D do bot;
- intrínsecos da câmera;
- pose odométrica estimada;
- eventos corporais implantáveis;
- detecção visual do alvo.

Ela pode reutilizar DDA, chunks esparsos, recortes e projeções do módulo voxel,
mas não pode reutilizar evidências já preenchidas por `bot.world`. O mapa
privilegiado serve como oracle de avaliação: mede completude, erro de ocupação,
altura e deriva sem entrar na política.

### Estrutura canônica e projeção para a rede

A memória canônica é voxel 3D esparsa, em chunks `16x16x16`. Cada observação
RGB-D é elevada a pontos 3D e transformada para o referencial do episódio pela
pose estimada. Cada update preserva frame de origem, pose, covariância,
validade, recência e peso.

Para permitir correções futuras de loop closure, a implementação deve manter
submapas/keyframes com suas transformações, em vez de fundir toda geometria de
forma irreversível. Uma correção de trajetória altera a pose do submapa e o
recorte local é regenerado.

O ator não recebe o mapa mundial inteiro. Recebe uma projeção local de um
bloco por célula, inicialmente `65x65`, centrada no bot. A memória permanece
alinhada ao referencial do episódio; a projeção para a CNN é rotacionada para
que “frente do bot” permaneça no topo.

Proposta de tensor:

\[
L_t\in\mathbb{R}^{6\times65\times65}
\]

| Plano | Significado | Derivação implantável |
|---|---|---|
| `known_confidence` | Evidência acumulada e confiança | rays RGB-D + incerteza da pose |
| `relative_floor_height` | Piso relativo ao bot | geometria depth fusionada |
| `obstacle_height` | Altura sólida acima do piso | geometria depth fusionada |
| `head_clearance` | Espaço vertical livre | geometria depth fusionada |
| `traversability` | Suporte e dois voxels corporais livres | projeção conservadora |
| `goal_marker` | alvo local ou indicação na borda | crença do alvo |

Não são necessários canais separados `free_probability` e
`occupied_probability`: espaço desconhecido é distinguido por
`known_confidence`, e a geometria é expressa pelos demais planos. O canal
`material_class` do mapa privilegiado fica fora da primeira versão. Ele só
poderá voltar como semântica estimada visualmente, com modelo e confiança
declarados.

## Crença sobre o objetivo

Quando o alvo está visível e possui depth válido, sua posição é reconstruída
em 3D e transformada para o referencial do episódio. Quando sai da imagem, sua
última crença permanece no mapa e a incerteza cresce com a odometria.

O vetor de objetivo contém nove valores:

```text
relative_x, relative_y, relative_z,
log_distance,
sin(bearing), cos(bearing),
sin(elevation), cos(elevation),
confidence
```

As coordenadas relativas são expressas no frame atual do agente e recebem
escala signed-log antes do clipping. Se o alvo estiver fora do recorte local,
`goal_marker` aparece na borda na direção estimada; não é colocado
artificialmente dentro da região conhecida.

## Montagem da observação da política

O contrato proposto é composto, não um único vetor achatado:

```text
MinecraftOdometricBeliefObservationV1
├── visual_rgbd: float32[5, 360, 640]
├── local_memory: float32[6, 65, 65]
├── auxiliary_values: float32[42]
├── auxiliary_valid: bool[42]
├── action_mask: bool[A]
└── provenance
```

### Vetor auxiliar de 42 posições

| Faixa | Dimensão | Conteúdo |
|---|---:|---|
| `delta_pose` | 6 | `log(SE3)` normalizado da última transição |
| `odometry_quality` | 6 | aceita, fallback, inlier ratio, residual normalizado, incerteza e idade da última atualização visual |
| `goal` | 9 | posição relativa, distância, bearing, elevação e confiança |
| `body` | 8 | saúde, fome, ar, no chão, na água, velocidade horizontal, velocidade vertical e colisão recente |
| `previous_executed_action` | 13 | mouse lon/lat, `move` one-hot 3, `strafe` one-hot 3 e cinco botões |

Todos os campos possuem máscara de validade explícita. Campos indisponíveis são
zero depois da normalização, mas o encoder recebe simultaneamente
`auxiliary_valid`; zero desconhecido nunca é confundido com zero medido. A
fonte também fica registrada no dataset.

`odometry_quality` segue a ordem:

```text
visual_accepted,
action_fallback,
inlier_ratio,
median_residual / max_metric_error,
translation_std / max_metric_error,
steps_since_visual_update / 32
```

Razões e idades são recortadas em `[0,1]`. `translation_std` é derivado do
maior termo translacional na diagonal da covariância 6x6. Os valores físicos
completos continuam no dataset.

`body` segue a ordem:

```text
health / health_max,
hunger / hunger_max,
air / air_max,
on_ground,
in_water,
horizontal_speed / speed_reference,
vertical_velocity / vertical_speed_reference,
collision_recent
```

Velocidades são recortadas em `[-1,1]`; flags são binárias.

`previous_executed_action` segue a ação aplicada ao mundo, não apenas a ação
solicitada pela política:

```text
delta_longitude / pi,
delta_latitude / latitude_limit,
one_hot(strafe: left, neutral, right),
one_hot(move: back, neutral, forward),
jump, sprint, sneak, attack, use
```

Os fatores de normalização, limites, máscaras e ordem fazem parte de
`state_assembly_version`; também ficam no manifesto e no checkpoint. Alterar
uma constante exige uma nova versão do contrato.

A pose episódica completa `T_hat_t` não entra por padrão no vetor. Ela indexa
o mapa e transforma o alvo. Expor `x,y,z` diretamente facilitaria memorização
de arenas e seeds. Uma variante com sete valores
`x,y,z,sin(yaw),cos(yaw),sin(pitch),cos(pitch)` será somente um braço explícito
de ablação, nunca uma adição silenciosa.

### Encoders e fusão

```text
z_visual = CNN_RGBD(visual_rgbd)
z_memory = CNN_2D(local_memory)
z_aux    = MLP(concat(auxiliary_values, auxiliary_valid))
h_t      = GRU(concat(z_visual, z_memory, z_aux), h_{t-1})
policy   = actor_heads(h_t, action_mask)
value    = critic_head(h_t)
```

- A CNN RGB-D pode preservar a arquitetura VGG existente inicialmente.
- A CNN de memória é independente; não se empilham mapa e RGB-D como se
  compartilhassem a mesma geometria de pixels.
- A GRU mantém informação dinâmica não representada pela geometria estática,
  como entidades móveis e eventos recentes.
- Actor e critic usam a mesma observação implantável no baseline. Um crítico
  privilegiado só pode existir como ablação assimétrica declarada.

## Sequência de montagem por passo

1. Capturar RGB, depth, validade, calibração e `frame_id` atomicamente.
2. Montar `visual_rgbd` com cinco planos nativos.
3. Associar features entre o frame anterior e o atual pelo frontend PanorAi.
4. Converter pixels para bearings e camera-Z para range radial.
5. Estimar SE(3), qualidade e covariância pelo PanorAi.
6. Atualizar `T_hat` ou aplicar fallback de ação com aumento de incerteza.
7. Elevar o depth atual a 3D e escrever um submapa ponderado pela incerteza.
8. Atualizar ou propagar a crença do alvo.
9. Renderizar o recorte local agente-cêntrico e o `goal_marker`.
10. Normalizar os 42 valores auxiliares e montar sua máscara na mesma ordem
    versionada.
11. Executar os encoders, GRU, actor e critic.
12. Persistir observação, ação solicitada, ação executada e evidência da
    atualização para replay e auditoria.

## Persistência no dataset

O dataset canônico deve guardar informação suficiente para reconstruir uma
representação futura, em vez de persistir somente o tensor final:

```text
odometry:
  interface
  frame_a / frame_b
  delta_rotation / delta_translation
  covariance_6x6
  visual_inlier_mask_ref / metric_inlier_mask_ref
  residual_summary
  accepted / rejection_reasons
  source: panorai_rgbd | action_fallback
  panorai_interface / panorai_commit

belief:
  episode_pose
  pose_covariance
  goal_estimate / goal_covariance
  submap_updates
  local_crop_spec
  state_assembly_version
```

RGB-D, odometria e deltas de submapa são canônicos. Recortes 65x65 e vetores
normalizados podem ser armazenados como cache com hash, mas devem ser
reconstruíveis. A pose verdadeira do servidor permanece em trilha
`privileged_evaluation` e nunca aparece em `policy_channels`.

## Ablação e gates

A integração deve avançar cumulativamente, mantendo arenas, seeds, ações,
reward e orçamento constantes:

1. `RGB-D` — baseline reativo atual;
2. `RGB-D + delta_pose + qualidade` — percepção de movimento recente;
3. `RGB-D + crença persistente do alvo`;
4. `RGB-D + memória local odométrica`;
5. `RGB-D + memória + GRU`;
6. ablação opcional com pose episódica explícita.

Gates mínimos:

- odometria em movimento avaliada contra pose privilegiada sem entregá-la ao
  ator;
- erro translacional, angular e crescimento da covariância calibrados;
- replay reproduz exatamente pose, submapas, recorte e vetor auxiliar;
- nenhuma evidência `bot.world` aparece na memória treinável;
- target permanece consistente durante oclusões curtas;
- fallback não escreve geometria de alta confiança;
- splits por mundo/seed impedem memorização de coordenadas;
- cada condição roda ao menos três seeds e avaliação congelada;
- promoção exige melhora em sucesso e tempo até o alvo, não só reward de
  treino.

## Situação atual e próximas entregas

Já existem e estão validados:

- RGB-D denso renderer-atomic em 640x360;
- encoder visual de cinco planos;
- frontend SIFT/matching pelo PanorAi;
- pose RGB-D métrica e gate de qualidade;
- acumulação SE(3) no adaptador do Gym;
- benchmark isolado multisseed;
- auditor passivo do renderer real;
- memória voxel esparsa e projeção 2D, hoje privilegiadas.

Ainda precisam ser implementados:

1. contrato persistente de crença odométrica e propagação de covariância;
2. captura real com câmera em movimento e ground truth apenas para avaliação;
3. memória RGB-D implantável separada do mapa de raycast;
4. submapas corrigíveis e montagem determinística do vetor auxiliar;
5. encoders de memória/auxiliar e GRU no PPO;
6. replay, ablações e gates de promoção.

Até esses itens passarem, o checkpoint atual continua usando RGB-D sem memória
odométrica e nenhum relatório deve chamá-lo de política espacial persistente.
