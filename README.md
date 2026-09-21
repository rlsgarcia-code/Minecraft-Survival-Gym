# Minecraft Survival Gym

> Um ambiente Gymnasium para treinar agentes de reinforcement learning a sobreviver no Minecraft Java — com observações visuais, controle completo do jogador e gravação de demonstrações por teclado e mouse.

`Minecraft Survival Gym` conecta políticas Python ao Minecraft Java 1.21 por meio de um mod Fabric e expõe o jogo como o ambiente `MinecraftSurvival-v0`. O objetivo é servir como base experimental para reinforcement learning, imitation learning, DAgger e, futuramente, geração dinâmica de objetivos e recompensas com VLMs e modelos de linguagem.

> [!IMPORTANT]
> Este é um projeto experimental e independente. Não é um produto oficial nem possui afiliação com Mojang ou Microsoft.

## Visão geral

O ambiente oferece:

- interface compatível com Gymnasium: `reset()`, `step()`, `render()` e `close()`;
- observação visual RGB combinada com estado estruturado do jogador;
- ações de movimento, câmera, combate, uso de itens, hotbar e interfaces gráficas;
- execução temporal controlada, com um número exato de ticks por ação;
- modos de controle por agente, humano e DAgger;
- gravação atômica de trajetórias para behavior cloning e imitation learning;
- backend simulado para desenvolver e testar sem abrir o Minecraft;
- função de recompensa substituível, sem acoplamento ao loop do jogo.

## Arquitetura

```text
Política RL / Operador humano
             │
             ▼
  MinecraftSurvivalEnv
       (Gymnasium)
             │  TCP local / JSON
             ▼
      Mod Fabric Bridge
             │
             ▼
 Minecraft Java — Survival
             │
             └── RGB + vitais + pose + inventário + eventos
```

O bridge escuta exclusivamente em `127.0.0.1:25570`. As requisições são processadas sequencialmente para preservar a relação entre ação, ticks executados e observação retornada.

## Estado do projeto

O ambiente foi validado ponta a ponta com uma instância real do Minecraft:

- handshake entre Python e o mod Fabric;
- captura RGB em `128 × 128 × 3`;
- avanço exato de quatro ticks por ação;
- movimento e rotação da câmera;
- execução no modo DAgger;
- gravação de episódios humanos;
- conformidade do ambiente com o verificador do Gymnasium;
- suíte automatizada com 8 testes.

## Requisitos

- Python 3.10 ou superior;
- [`uv`](https://docs.astral.sh/uv/);
- Java 21;
- Minecraft Java Edition 1.21;
- Fabric Loader;
- Fabric API compatível com Minecraft 1.21.

## Instalação

Clone o repositório e instale o pacote Python:

```bash
git clone https://github.com/rlsgarcia-code/Minecraft-Survival-Gym.git
cd Minecraft-Survival-Gym
uv sync --extra dev
```

Compile o mod Fabric:

```bash
cd fabric
./gradlew build
cd ..
```

O artefato será criado em:

```text
fabric/build/libs/minecraft-gym-bridge-0.1.0.jar
```

Copie esse arquivo para a pasta `mods` de uma instalação Fabric do Minecraft 1.21 que também contenha a Fabric API. Depois, abra um mundo single-player em modo Survival.

## Teste rápido

Com o Minecraft aberto dentro de um mundo:

```bash
uv run python scripts/smoke_env.py --steps 20
```

Para verificar apenas a interface Gymnasium, sem iniciar o jogo:

```bash
uv run python scripts/smoke_env.py --backend mock --steps 20
```

## Uso básico

```python
import gymnasium as gym
import minecraft_gym  # registra MinecraftSurvival-v0

env = gym.make(
    "MinecraftSurvival-v0",
    backend="socket",
    width=128,
    height=128,
    frame_skip=4,
    max_episode_steps=9_000,
    control_mode="agent",
    render_mode="rgb_array",
)

observation, info = env.reset(seed=42)

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Use `backend="mock"` para testes rápidos e determinísticos sem uma instância do Minecraft.

## Espaço de ações

O espaço é um `gymnasium.spaces.MultiDiscrete` com 15 componentes:

```text
MultiDiscrete([3, 3, 2, 2, 2, 2, 2, 2, 2, 5, 5, 10, 5, 5, 3])
```

O valor `0` representa `NOOP` em todos os componentes.

| Índice | Ação | Valores |
|---:|---|---|
| 0 | Strafe | parado, esquerda, direita |
| 1 | Movimento | parado, frente, trás |
| 2 | Pular | não, sim |
| 3 | Correr | não, sim |
| 4 | Agachar | não, sim |
| 5 | Atacar | não, sim |
| 6 | Usar | não, sim |
| 7 | Inventário | manter, alternar |
| 8 | Soltar item | não, sim |
| 9 | Câmera horizontal | neutro, ±6°, ±18° |
| 10 | Câmera vertical | neutro, ±6°, ±18° |
| 11 | Hotbar | manter ou selecionar slots 1–9 |
| 12 | Cursor horizontal | neutro, lento ou rápido nos dois sentidos |
| 13 | Cursor vertical | neutro, lento ou rápido nos dois sentidos |
| 14 | Clique em GUI | nenhum, primário, secundário |

O objeto canônico `ControlState` preserva deltas contínuos de mouse. Assim, demonstrações humanas mantêm mais informação do que a projeção discreta usada pela primeira versão da política.

## Espaço de observações

O ambiente retorna um `gymnasium.spaces.Dict`:

| Campo | Tipo e forma | Conteúdo |
|---|---|---|
| `rgb` | `uint8[H, W, 3]` | imagem renderizada em RGB |
| `vitals` | `float32[8]` | vida, fome, armadura, ar, XP e flags corporais |
| `pose` | `float32[10]` | posição, velocidade, câmera, contato com o chão e queda |
| `inventory_ids` | `int32[36]` | identificadores dos itens |
| `inventory_counts` | `int32[36]` | quantidade em cada slot |
| `equipped_slot` | `Discrete(9)` | slot ativo da hotbar |
| `ui_mode` | `Discrete(8)` | jogo, inventário, container, chat, morte ou outra tela |

O dicionário `info` inclui:

- tick atual e seed real do mundo;
- eventos, como dano, morte e aumento de inventário;
- decomposição dos termos da recompensa;
- `policy_action`, `human_action` e `executed_action`;
- `control_source`, indicando se o controle veio do agente ou do humano.

## Modos de controle

| Modo | Comportamento |
|---|---|
| `agent` | executa somente a ação enviada pela política |
| `human` | ignora a ação da política e registra teclado/mouse |
| `dagger` | usa a política normalmente e aceita intervenção humana |

### Gravação de demonstrações

Mantenha a janela do Minecraft em foco e execute:

```bash
uv run python scripts/record_human.py \
  --mode human \
  --steps 9000 \
  --output datasets/demonstrations
```

Para coletar correções humanas durante a execução de uma política, use:

```bash
uv run python scripts/record_human.py \
  --mode dagger \
  --steps 9000 \
  --output datasets/dagger
```

Cada episódio produz um arquivo `.npz` e metadados `.json`. A gravação é atômica, reduzindo o risco de datasets parcialmente escritos.

Principais arrays:

| Campo | Uso |
|---|---|
| `actions` | ação originalmente enviada ao ambiente |
| `policy_actions` | ação da política projetada no espaço discreto |
| `human_actions` | ação humana projetada no espaço discreto |
| `executed_actions` | ação efetivamente executada |
| `policy_controls` | controle canônico contínuo da política |
| `human_controls` | teclado e deltas contínuos do mouse |
| `executed_controls` | controle canônico efetivamente aplicado |
| `human_action_present` | máscara de disponibilidade/intervenção humana |

> [!NOTE]
> Bloquear a tela não impede testes e treinamento puramente Python, mas o Minecraft pode interromper a renderização. Captura RGB e entrada física de teclado/mouse requerem uma sessão gráfica desbloqueada.

## Recompensas

A recompensa padrão é deliberadamente simples:

- pequeno bônus por permanecer vivo;
- penalidade proporcional à perda de vida;
- penalidade maior por morte.

Uma estratégia diferente pode ser injetada sem alterar o ambiente:

```python
from minecraft_gym import MinecraftSurvivalEnv
from minecraft_gym.rewards import RewardResult


class MyReward:
    def __call__(self, previous, transition):
        collected = sum(
            event.get("count", 0)
            for event in transition.events
            if event.get("type") == "inventory_increased"
        )
        return RewardResult(
            total=float(collected),
            terms={"collected_items": float(collected)},
        )


env = MinecraftSurvivalEnv(reward_function=MyReward())
```

Essa separação permite evoluir para recompensas semânticas produzidas por um VLM/LLM, objetivos intermediários e conhecimento recuperado de guias de sobrevivência, sem acoplar esses experimentos ao bridge do Minecraft.

## Semântica temporal e episódios

Durante uma sessão Gym, o bridge congela o servidor integrado e avança exatamente `frame_skip` ticks para cada chamada de `step()`.

- `terminated=True`: o jogador morreu;
- `truncated=True`: o episódio atingiu um limite operacional, como `max_episode_steps`.

### Reset

A versão atual implementa um **soft reset**. Ela:

- força o jogador para Survival;
- retorna ao ponto inicial da sessão;
- limpa inventário e efeitos;
- restaura vida, fome e saturação;
- restaura horário e clima.

O reset ainda não reconstrói blocos modificados nem troca a seed do mundo aberto. O campo `requested_seed_matches_world` informa se a seed solicitada coincide com a seed real. Para experimentos que exigem terreno idêntico, inicie cada lote a partir de uma cópia limpa do save.

## Configuração do ambiente

| Parâmetro | Padrão | Descrição |
|---|---:|---|
| `backend` | `"socket"` | backend real ou `"mock"` |
| `host` | `127.0.0.1` | endereço do bridge |
| `port` | `25570` | porta TCP local |
| `bridge_timeout` | `30.0` | timeout da conexão em segundos |
| `width` | `128` | largura da observação RGB |
| `height` | `128` | altura da observação RGB |
| `frame_skip` | `4` | ticks executados por ação |
| `max_episode_steps` | `9000` | limite interno do episódio |
| `control_mode` | `"agent"` | `agent`, `human` ou `dagger` |
| `render_mode` | `None` | use `"rgb_array"` para `render()` |
| `reward_function` | sobrevivência esparsa | estratégia substituível de recompensa |

## Desenvolvimento

Execute os testes Python:

```bash
uv run pytest
```

Compile e valide o mod:

```bash
cd fabric
./gradlew build
```

Estrutura principal:

```text
src/minecraft_gym/       ambiente, ações, reward, recorder e transporte
fabric/                  mod Fabric e bridge local
scripts/smoke_env.py     teste rápido real ou simulado
scripts/record_human.py  coleta de demonstrações humanas
tests/                   testes do contrato Gym e do protocolo
docs/protocol.md         especificação do protocolo TCP
```

## Roadmap

- reset rígido e restauração automática de mundos;
- ambientes vetorizados e múltiplas instâncias do Minecraft;
- wrappers de ações hierárquicas e temporais;
- baselines de behavior cloning e reinforcement learning;
- avaliação e geração dinâmica de recompensas com VLMs;
- planejamento de submetas com modelos de linguagem e retrieval de guias;
- ferramentas de inspeção, replay e curadoria de demonstrações.

## Protocolo

A especificação do bridge está em [`docs/protocol.md`](docs/protocol.md).

## Contribuições

Issues e pull requests são bem-vindos. Ao relatar um problema, inclua a versão do Minecraft, Fabric Loader, Fabric API, Java, sistema operacional e os passos mínimos para reprodução.
