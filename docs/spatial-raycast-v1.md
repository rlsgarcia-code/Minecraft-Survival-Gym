# Spatial raycast v1

Este documento fixa o contrato entre Prismarine, Gym, dataset e Observer para
o mapa espacial do BrincaBot. O Observer não possui uma versão particular do
mapa: ele reproduz os mesmos `SpatialMapUpdateV1` persistidos nos episódios.

## Fonte de verdade

Há duas representações sincronizadas do mesmo mundo:

1. `prismarine-viewer` renderiza os chunks recebidos de `bot.world`;
2. `prismarine-world` consulta esses mesmos chunks por coordenada e intersecta
   o raio com as `shapes` do bloco.

O tipo do bloco não é reconhecido a partir da textura RGB. O renderer fornece
a câmera e o pixel fornece uma direção. `bot.world.raycast` devolve o primeiro
bloco atingido, incluindo posição inteira, ponto contínuo de contato, face,
nome e `stateId`.

```text
pixel (u,v)
    │ calibração + base da câmera
    ▼
raio mundial O + tD
    │ travessia dos voxels e shapes de bot.world
    ▼
contato + Block{name, stateId, position}
    │ projeção X/Z
    ▼
evidência aditiva no lattice do ator
```

Uma região fora dos chunks carregados é `unknown`, nunca espaço livre.

## Projeção de câmera

Para uma imagem de largura `W`, altura `H` e campo de visão vertical `fov_y`,
o centro do pixel é convertido para coordenadas normalizadas:

```text
nx = 2 (u + 0.5) / W - 1
ny = 1 - 2 (v + 0.5) / H
```

Usando a base ortonormal do renderer `forward`, `right`, `up`:

```text
sx = nx · (W/H) · tan(fov_y/2)
sy = ny · tan(fov_y/2)
D  = normalize(forward + sx·right + sy·up)
O  = origem da câmera
```

A base vetorial é autoritativa. `yaw` e `pitch` são registrados apenas para
auditoria, evitando diferenças de sinal/eixo entre Minecraft, Mineflayer e
Three.js.

## Contratos persistidos

- `CameraCalibrationV1`: resolução, FOV, planos e convenção de centro do pixel;
- `CameraPoseV1`: origem, base ortonormal, ator, tick e fonte;
- `RaycastSampleV1`: pixel, `O`, `D`, alcance e resultado estruturado;
- `RaycastFrameV1`: amostra esparsa referente a um frame visual imutável;
- `SpatialMapCellDeltaV1`: evidência aditiva numa célula X/Z;
- `SpatialMapUpdateV1`: raycast original mais o delta determinístico do mapa.
- `SpatialMapSummaryV1`: projeção compacta com célula atual, limites, contagens,
  semântica acumulada e fronteiras observadas adjacentes ao desconhecido.

Cada mapa pertence a um ator. A visão do demonstrador e a visão do bot podem
coexistir no episódio, mas nunca compartilham `map_id` nem perdem a referência
ao frame que originou a observação.

## Semântica das células

Uma célula acumula contadores, não um único valor destrutivo:

- `visited_count`: posição autoritativa do corpo ocupou a célula;
- `free_count`: raios atravessaram a célula antes de atingir um bloco;
- `hit_count`: um raio terminou num bloco projetado nessa célula;
- `block_counts`: histograma dos tipos atingidos;
- `solid_y_levels`: níveis Y inteiros dos blocos realmente atingidos;
- `solid_stack_height`: quantidade contígua observada acima do bloco-base
  (um piso isolado vale zero), usada para distinguir paredes/pilhas de uma
  simples mudança de piso;
- `head_clearance`: menor vão vertical observado até um bloco superior; menos
  de dois blocos livres não acomoda o corpo em pé;
- `surface_y_min/max`: faixa vertical observada;
- `last_seen_tick`: tick da evidência mais recente.

Os dois canais topológicos são projeções determinísticas dos contatos exatos
do raycast, não palpites do modelo. Em escalas maiores, a agregação preserva a
maior pilha e a menor folga de suas células finas; ela nunca combina o piso de
uma célula com o teto de outra. Ausência de teto observado é codificada como
folga aberta/indeterminada de quatro blocos, não como prova de espaço infinito.

`visited`, `observed`, `semantic` e `inferred` são conceitos separados. Esta
primeira versão não persiste inferências do modelo como se fossem medições.

O delta é aditivo e suficiente para reconstruir o mapa em replay. O estado
completo em memória é uma projeção descartável que o Observer pode regenerar.
As `frontier_cells` são células com evidência de passagem ou visita que ainda
possuem ao menos um vizinho desconhecido; elas dão ao planejador destinos de
exploração sem enviar os raios brutos ao modelo de linguagem.

## Terminação do raio

- `block_hit`: exige contato e identidade completa do bloco;
- `max_range`: todo o segmento confirmado é observado como livre;
- `unloaded_chunk`: a célula final permanece desconhecida; somente o percurso
  confirmado anterior produz evidência livre.

Blocos não cúbicos usam as `shapes` do Prismarine. Água, vidro, folhas e
plantas continuarão identificados pelo estado real; a decisão posterior de
"visível", "sólido" ou "atravessável" pertence ao modelo semântico, não ao
raycast bruto. Entidades exigem uma camada adicional de interseção com seus
volumes e não são falsamente registradas como blocos.

## Relação com treino

O escopo padrão é `privileged_derived`: o mapa pode apoiar navegação de alto
nível, recompensa, avaliação, rotulagem e depuração, mas seu canal é rejeitado
se aparecer entre os inputs da política visual.

Um experimento pode declarar `deployable_sensor` somente se o mesmo produtor,
amostragem, alcance e contrato estiverem disponíveis no treino, avaliação e
rollout. Essa declaração não transforma uma observação global em uma câmera;
o dataset preserva a origem estruturada.

## Frequência e amostragem

RGB pode permanecer a 20 Hz sem executar um raio por pixel. Uma grade de
40×23 (920 raios) ou 80×45 (3.600 raios) atualizada em frequência menor é
suficiente para a primeira navegação. A configuração exata deve ser
versionada pelo adaptador produtor e medida antes de aumentar a densidade.

O acompanhamento contínuo do Observer usa 32×18 (576 raios) por padrão. Essa
densidade preserva detalhes de corredores, carpetes e obstáculos estreitos sem
bloquear a interface no hardware atual. Coletas de pesquisa podem alterar a densidade por
`BRINCABOT_SPATIAL_COLUMNS` e `BRINCABOT_SPATIAL_ROWS`; esses valores precisam
entrar na proveniência da sessão.

### Limite conhecido do primeiro produtor

O cliente web padrão do `prismarine-viewer` interpola atualizações da câmera
por 50 ms. O produtor atual reconstrói a mesma câmera Three.js a partir da pose
da entidade congelada junto ao screenshot; em cenas paradas a projeção
coincide, mas durante movimento rápido pode haver até um tick de diferença em
relação à matriz efetivamente rasterizada. Por isso os samples atuais declaram
`confidence=0.95` e são `privileged_derived`, não rótulos densos perfeitos por
pixel. O gate para elevar a confiança a 1.0 é exportar a matriz da câmera do
browser no instante do frame ou remover a interpolação no cliente versionado.

## Responsabilidades por componente

### Adaptador Prismarine/BrincaBot

- extrair pose e base da câmera do mesmo frame exibido;
- calcular ou confirmar `O` e `D` por pixel;
- executar `bot.world.raycast` e distinguir chunk descarregado;
- emitir `RaycastFrameV1` alinhado ao tick e ao `camera_frame_id`.

A implementação canônica fica em
`integrations/prismarine/spatial-raycast-v1.js`. O BrincaBot a carrega a partir
do repositório fixado em `gym-source.lock.json`; não deve manter uma cópia.

### Minecraft Survival Gym

- validar contratos e proveniência;
- gerar deltas determinísticos com `SpatialLatticeMap`;
- persistir o update dentro de `ObservationRefV2`;
- impedir vazamento privilegiado para `policy_channels`.

### Observer

- consumir os updates do Gym;
- mostrar visitado, observado, desconhecido e semântica por ator;
- reproduzir um episódio sem recalcular evidência de forma diferente;
- nunca promover inferência visual a medição autoritativa silenciosamente.

### Projeção humana no Observer

O Observer não mantém um segundo mapa. `HttpSpatialStreamClient.snapshot_cells()`
produz uma cópia defensiva do mesmo `SpatialLatticeMap` usado pelo Gym. O
BrincaBot a publica localmente como `brincabot-spatial-map/v1` em
`GET /api/spatial/map`, contendo `summary` e as células acumuladas.

A tela “Agora” usa duas projeções desse contrato:

- o minimapa no Planejador, que apresenta posição atual, área conhecida e
  fronteiras sem expor detalhes técnicos;
- o mapa ampliado, que permite zoom, seleção de células, leitura semântica e
  distinção explícita entre observado e desconhecido.

`POST /api/spatial/chat` entrega ao modelo somente o resumo do mapa, células
semânticas e células visitadas. Essa conversa é explicativa: não executa
`goal_request`, não controla o corpo e deve declarar como desconhecido tudo o
que estiver fora da evidência. Assim, a interface humana, o planejador futuro
e o dataset permanecem consumidores do mesmo contrato, sem “puxadinho” de UI.

### Explorador visual v1

A primeira rotina operacional de exploração é deliberadamente determinística,
sem inferência do Ollama por passo. Ela recebe somente um orçamento de tempo e:

1. constrói ações candidatas a partir do frame/raycast atual: aproximar-se de
   um objeto observado no pixel `(u,v)` ou explorar uma nova rotação;
2. estima valor por novidade adjacente, raridade semântica, distância, custo e
   repetição recente;
3. usa a base `forward/right` da câmera para centralizar o pixel e avança em
   pulsos curtos, sem incorporar salto nesta política;
4. mede ganho de células observadas, visitadas, semânticas e novos tipos de
   recurso;
5. reduz o valor de pixels e rotações que não produziram progresso;
6. para ao terminar o tempo, por solicitação do operador ou por baixa vida.

O espaço de decisão é hierárquico e mantém três projeções sem confundi-las:

- `Q_view(u,v)` avalia somente alvos presentes no frame atual;
- `SphericalValueLatticeV1` é um lattice equiretangular alinhado ao mundo. Ele
  acumula evidência angular dos raycasts, mantém bins nunca vistos explícitos e
  atribui a eles bônus de novidade reproduzível;
- o lattice X/Z fornece fronteiras e custo Manhattan. Uma direção esférica é
  favorecida quando revela ângulos novos e aponta para uma fronteira próxima,
  descontando o número estimado de passos angulares e espaciais.

O campo `novelty_value` não é chamado de Q aprendido: ele pertence ao estimador
de cobertura `coverage-heuristic/v1`. Cada ação candidata registra a decomposição
em `q_components`, permitindo substituir a heurística por uma função de valor
aprendida sem reescrever o sensor ou o dataset. O estado equiretangular completo
é validado pelo Gym, salvo por conteúdo e referenciado no passo do rollout; os
raycasts originais permitem reconstruí-lo.

Cada transição registra os termos `observed_gain`, `semantic_gain`,
`visited_gain`, `new_resource_types` e `time_cost`. A soma atual é apenas um
`exploration_value` heurístico e fica separada dos dados brutos, permitindo
recalcular recompensa ou substituir o controlador por uma política de RL.
Rollouts recebem `controller=visual-explorer/v1`; eles não são confundidos
com demonstrações humanas nem com decisões do VLM.

## Critérios de aceitação do produtor

1. O raio central coincide com o vetor `forward` da câmera.
2. Pixels de borda têm direção compatível com FOV e aspecto registrados.
3. Um bloco conhecido em cenário sintético retorna posição, contato e tipo.
4. O `camera_frame_id`, ator e tick existem na observação correspondente.
5. Chunks ausentes geram `unloaded_chunk`, não `max_range`.
6. Replay dos deltas reproduz os mesmos contadores e histogramas.
7. Mapas de atores diferentes permanecem isolados.
8. O canal privilegiado não é aceito como input da política por omissão.
