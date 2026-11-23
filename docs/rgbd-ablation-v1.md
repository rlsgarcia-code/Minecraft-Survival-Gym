# Ablação visual RGB × RGB-D v1

## Hipótese

Em navegação local até um alvo visual, profundidade métrica alinhada ao RGB
reduz colisões e tempo até o alvo em comparação com RGB sozinho. Memória
recorrente, mapa persistente e canais privilegiados ficam fora desta ablação.

## Controle experimental

- mesmas arenas, seeds, ações, reward, número de episódios e ordem de amostras;
- mesma arquitetura e número de parâmetros: a rede sempre recebe cinco planos
  (`RGB`, profundidade normalizada e validade); no braço RGB os dois últimos
  planos são zero;
- inicializações pareadas por seed e três réplicas independentes;
- treino e avaliação agrupados por mundo/seed, sem cruzamento;
- alvo visualmente identificável no frame inicial; episódios onde a tarefa não
  define o alvo são inelegíveis para esta comparação;
- nenhuma coordenada absoluta, rota ótima, block id ou mapa global entra na
  política.

## Observação RGB-D

O ensaio histórico descrito abaixo usou o raycast esparso
`minecraft-depth-map/v1`, 32×18, pareado ao RGB 640×360. Ele permanece válido
apenas como proxy de engenharia e não deve ser chamado de depth denso.

O caminho de treino longo usa agora `minecraft-dense-rgbd-frame/v1`: RGB e
230.400 valores de depth são renderizados no mesmo framebuffer em 640×360 e
compartilham `camera_frame_id`. Os cinco canais entram na política em resolução
nativa: RGB, profundidade normalizada e validade. A redução espacial pertence
aos blocos convolucionais VGG, seguida por flatten e MLP. Distâncias
são expressas em blocos e normalizadas por `max_distance`. Contato geométrico e
alcance máximo são observações válidas distintas; somente `unknown` é mascarado.
O gate de latência, armazenamento e integridade está documentado em
[`dense-rgbd-long-run.md`](dense-rgbd-long-run.md).

## Métricas primárias

1. sucesso em seeds inéditas;
2. ticks até o alvo, condicionado a sucesso;
3. retorno episódico mediano;
4. colisões e tempo parado por episódio.

Métricas secundárias: variância entre seeds, entropia da política, dano,
quebras/colocações e calibração do crítico. A decisão usa intervalo de confiança
por bootstrap pareado sobre seeds, não apenas a melhor curva de treino.

## Gate

O RGB-D avança para o estudo de memória somente se melhorar a mediana de ticks
até o alvo sem reduzir sucesso inédito, em pelo menos duas de três réplicas, e
se a vantagem persistir quando o depth é recalculado sem interpolação. Depois
disso, a terceira condição adicionará memória espacial/recorrente mantendo RGB-D
e todo o restante congelado.

## Execução local sem o volume de pesquisa

O script `scripts/run_rgbd_ablation.py` permite somente um ensaio de engenharia
deliberadamente limitado no disco interno. Ele exige um diretório novo, reserva
no mínimo 20 GiB livres, interrompe ao atingir 512 MiB por padrão e atualiza
`storage-monitor.json` após cada episódio. O comportamento seguro normal de
`ResearchStorage` não é alterado: captura visual de dataset continua bloqueada
sem o SSD externo.

O script separa duas evidências:

1. `live-sensor-smoke/` contém um único par RGB/raycast real e sincronizado,
   capturado passivamente sem controlar o mundo familiar;
2. os braços `rgb-*` e `rgbd-*` usam o ambiente procedimental da Academia para
   validar pesos pareados, encoder, ação, reward, métricas e checkpoints.

O segundo item é marcado `engineering_proxy_complete` e nunca satisfaz o gate
de desempenho no Minecraft real. Resultado com zero sucessos em avaliação é
inconclusivo mesmo que um braço apresente retorno menos negativo.
