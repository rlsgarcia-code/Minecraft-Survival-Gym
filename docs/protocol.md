# Protocolo Minecraft Gym Bridge v2

O cliente Python abre uma conexão TCP persistente com o mod Fabric em
`127.0.0.1:25570`. Cada mensagem contém um tamanho unsigned big-endian de
quatro bytes seguido por um objeto JSON UTF-8. O limite é 64 MiB.

Requisições e respostas são sequenciais. Cada conexão recebe um `session_id`;
transições usam `step_id` contíguo e o servidor confirma sessão, episódio e
passo. Toda requisição após o handshake contém SHA-256 do JSON canônico.

## Handshake

```json
{
  "type": "hello",
  "protocol_version": 2,
  "action_contract": "minecraft-native-control/v2",
  "observation_contract": "minecraft-observation/v2"
}
```

```json
{
  "ok": true,
  "protocol_version": 2,
  "bridge_version": "0.2.0",
  "minecraft_version": "fixed-by-mod-build",
  "session_id": "session:uuid",
  "action_contract": "minecraft-native-control/v2",
  "observation_contract": "minecraft-observation/v2",
  "tick_rate_hz": 20,
  "capabilities": [
    "rgb",
    "human_input",
    "dagger",
    "survival_soft_reset",
    "frozen_tick_step",
    "semantic_events",
    "privileged_context",
    "checksummed_requests",
    "contiguous_step_ack"
  ]
}
```

## Reset

```json
{
  "type": "reset",
  "protocol_version": 2,
  "session_id": "session:uuid",
  "step_id": 0,
  "seed": 42,
  "options": {"difficulty": "normal", "start_time": 0},
  "image": {"width": 128, "height": 128, "format": "raw_rgb8"},
  "payload_sha256": "sha256-do-json-sem-este-campo"
}
```

O reset só responde depois que mundo, jogador e renderizador estiverem prontos.
Ele restaura o jogador e condições globais no mundo aberto; a seed solicitada é
informativa e a resposta indica se coincide com a seed real.

## Step

```json
{
  "type": "step",
  "protocol_version": 2,
  "session_id": "session:uuid",
  "step_id": 1,
  "frame_skip": 4,
  "control_mode": "agent",
  "action": {
    "move_x": 0.0,
    "move_z": 1.0,
    "jump": false,
    "sprint": true,
    "sneak": false,
    "attack": false,
    "use": false,
    "inventory": false,
    "drop": false,
    "yaw_delta": 6.0,
    "pitch_delta": 0.0,
    "hotbar_slot": -1,
    "cursor_x_delta": 0.0,
    "cursor_y_delta": 0.0,
    "primary_click": false,
    "secondary_click": false
  },
  "payload_sha256": "sha256-do-json-sem-este-campo"
}
```

Em `human`, a ação enviada é ignorada e o mod usa teclado e mouse. Em `dagger`,
o mod usa input humano quando houver intervenção e informa ambas as ações.

## Resposta de transição

```json
{
  "ok": true,
  "session_id": "session:uuid",
  "episode_id": "episode:uuid",
  "step_id": 1,
  "observation": {
    "rgb": {
      "encoding": "raw_rgb8",
      "width": 128,
      "height": 128,
      "data": "base64..."
    },
    "vitals": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "pose": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "inventory_ids": [0],
    "inventory_counts": [0],
    "equipped_slot": 0,
    "ui_mode": 0,
    "tick": 4,
    "world_seed": 42
  },
  "terminated": false,
  "truncated": false,
  "events": [{"type": "inventory_delta", "item": "minecraft:oak_log", "delta": 1}],
  "info": {
    "control_source": "agent",
    "policy_action": {},
    "human_action": null,
    "executed_action": {},
    "privileged_context": {
      "day_time": 4000,
      "raining": false,
      "light_level": 15,
      "can_see_sky": true,
      "submerged": false,
      "hostile_count": 0,
      "nearest_hostile_distance": -1.0
    }
  }
}
```

`inventory_ids` e `inventory_counts` possuem 36 elementos. Imagens são RGB,
row-major e sem padding. Eventos incluem `inventory_delta`, `recipe_unlocked`,
`vital_delta`, `damage`, `death` e `ui_changed`. `privileged_context` existe
para recompensa e avaliação; não deve entrar na política visual.

## Close

```json
{
  "type": "close",
  "protocol_version": 2,
  "session_id": "session:uuid",
  "payload_sha256": "sha256-do-json-sem-este-campo"
}
```

O `close` encerra a sessão de controle, mas não fecha o Minecraft.

## JSON canônico e falhas

O hash é calculado removendo `payload_sha256` e serializando o restante com
chaves ordenadas, Unicode UTF-8 sem escape HTML e sem espaços entre separadores. Sessão incorreta,
checksum inválido ou `step_id` fora de ordem são rejeitados antes da ação chegar
ao runtime.

O cliente Python negocia v1 apenas para migração de mods antigos. Uma coleta de
pesquisa deve registrar a versão negociada e nunca tratar v1 como episódio v2.
