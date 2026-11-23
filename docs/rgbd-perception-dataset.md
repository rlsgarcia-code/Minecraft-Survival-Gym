# Minecraft RGB→Depth Dataset v1

## Escopo

`minecraft_gym.rgbd` planeja um dataset de percepção supervisionada composto
por pares RGB e profundidade métrica. Ele é independente do dataset episódico
de RL: trajetórias poderão referenciar pares RGB-D, mas ações, rewards e
terminações não fazem parte deste contrato.

O dataset ainda **não está pronto para coleta**. O depth 640×360 atualmente
disponível no experimento de ablação é uma projeção de 32×18 raycasts. Ele é
útil para depuração, mas é rejeitado pelo contrato como ground truth denso.

## Ground truth aceito

- RGB e depth pertencem ao mesmo `camera_frame_id`;
- resolução inicial: 640×360;
- origem: depth buffer do renderer Prismarine/Three.js;
- profundidade linearizada: `camera_z_blocks`;
- `float16` ou `uint16` linear, acompanhada por máscara de validade;
- calibração, FOV, planos near/far, mundo, seed pseudonimizada, dimensão,
  bioma, situação e tick registrados por par;
- nenhum upsampling de raycast pode ser declarado como depth denso.

## Procedimento de construção

1. Extrair o depth buffer da mesma render pass que produz o RGB.
2. Linearizar o valor OpenGL/WebGL usando os planos near/far e validar em
   planos posicionados a distâncias conhecidas.
3. Vincular RGB, depth, máscara, pose e calibração a um frame imutável.
4. Medir uma hora a 20 Hz: o gate exige pelo menos 19,8 FPS e no máximo 5 ms
   de overhead p95 causado pela captura.
5. Gerar candidatos a 20 Hz, mas admitir normalmente até 5 pares/s por
   novidade de pose, aparência, geometria, bioma ou situação. Isso evita
   sequências quase duplicadas.
6. Coletar um piloto de 5.000 pares e auditar transparência, água, céu,
   partículas, entidades, vegetação, mudanças de bloco e baixa luminosidade.
7. Congelar splits por mundo e seed: 80% treino, 10% validação e 10% teste.
8. Coletar o baseline de 100.000 pares, verificando quotas, hashes, duplicatas,
   distribuição de depth e pixels inválidos.
9. Treinar baselines RGB→depth sem acoplar o formato a uma arquitetura.

## Cobertura inicial

O plano distribui 100.000 pares por famílias de bioma e, separadamente, por
situações de jogo. Entre elas estão florestas, selva, deserto, neve, pântano,
montanhas, água, cavernas, Nether, End, interiores construídos, combate,
lava e mudanças de blocos. Os eixos são marginais; o coletor deverá também
auditar suas combinações para não concentrar casos difíceis em um só split.

São usados por padrão 80 mundos: 64 treino, 8 validação e 8 teste. Nenhum
mundo, seed ou linhagem de chunk pode atravessar splits.

## Estimativa de armazenamento

As estimativas incluem RGB comprimido, depth de 16 bits comprimido, máscara e
aproximadamente 2 KiB de manifesto/índice por par. A capacidade de trabalho
adiciona 25% para WAL, arquivos temporários e compactação. Checkpoints e
snapshots de mundos não estão incluídos.

| Pares | Otimista, dados | Esperado, dados | Conservador, dados | Capacidade segura conservadora |
|---:|---:|---:|---:|---:|
| 5.000 | ~1,0 GiB | ~1,7 GiB | ~2,8 GiB | ~3,5 GiB |
| 100.000 | ~20,6 GiB | ~34,5 GiB | ~56,0 GiB | ~70,0 GiB |
| 500.000 | ~102,9 GiB | ~172,6 GiB | ~279,9 GiB | ~349,9 GiB |
| 1.000.000 | ~205,8 GiB | ~345,2 GiB | ~559,8 GiB | ~699,8 GiB |

Um SSD de 1 TB comporta o baseline de 100 mil com ampla margem. Mantendo 15%
livre e usando a hipótese conservadora com overhead, o limite operacional é da
ordem de 1,2 milhão de pares, antes de reservar espaço para mundos e modelos.

Gere números exatos para outra resolução ou quantidade:

```bash
minecraft-gym rgbd estimate --pairs 100000 --width 640 --height 360
```

## Plano versionado

O comando abaixo grava somente o plano JSON; ele não inicia Minecraft nem cria
payloads de dataset:

```bash
minecraft-gym rgbd plan \
  --output datasets/rgbd/plans/minecraft-rgb-depth-v1.json \
  --pairs 100000 --width 640 --height 360 --worlds 80
```

Enquanto os seis gates não passarem, `capture_ready` permanece `false`.
