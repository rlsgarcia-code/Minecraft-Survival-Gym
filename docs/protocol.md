# Protocolo Minecraft Gym Bridge v1

O cliente Python abre uma conexão TCP persistente com o mod Fabric em
`127.0.0.1:25570`. Cada mensagem contém:

1. tamanho JSON em quatro bytes unsigned, big-endian;
2. objeto JSON UTF-8 com esse tamanho.

O limite é 64 MiB por mensagem. Requisições e respostas são estritamente
sequenciais para preservar a relação ação–tick–observação.

## Handshake

```json
{"type":"hello","protocol_version":1}
```

```json
{
  "ok": true,
  "protocol_version": 1,
  "minecraft_version": "fixed-by-mod-build",
  "capabilities": [
    "rgb",
    "human_input",
    "dagger",
    "survival_soft_reset",
    "frozen_tick_step",
    "semantic_events",
    "privileged_context"
  ]
}
```

## Reset

```json
{
  "type": "reset",
  "protocol_version": 1,
  "seed": 42,
  "options": {"difficulty": "normal", "start_time": 0},
  "image": {"width": 128, "height": 128, "format": "raw_rgb8"}
}
```

O reset só responde depois que mundo, jogador e renderizador estiverem prontos.
Na versão 1 ele restaura o jogador e condições globais no mundo aberto; a seed
solicitada é informativa e a resposta indica se coincide com a seed real.

## Step

```json
{
  "type": "step",
  "protocol_version": 1,
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
  }
}
```

Em `human`, a ação enviada é ignorada e o mod usa teclado e mouse. Em `dagger`,
o mod usa input humano quando houver intervenção e informa ambas as ações.

## Resposta de transição

```json
{
  "ok": true,
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
  "events": [
    {
      "type": "inventory_delta",
      "item": "minecraft:oak_log",
      "before": 0,
      "after": 1,
      "delta": 1
    }
  ],
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

`inventory_ids` e `inventory_counts` sempre possuem 36 elementos. Imagens são
RGB, row-major, sem padding. `terminated` representa morte; desconexão, timeout
ou limite operacional devem produzir `truncated`.

Eventos suportados são `inventory_delta`, `recipe_unlocked`, `vital_delta`,
`damage`, `death` e `ui_changed`. Durante a transição, o bridge também mantém
os eventos legados `inventory_increased` e `health_lost`. Inventário é agregado
por item, incluindo o stack carregado pelo cursor da GUI; reorganizar slots não
produz progresso falso.

`privileged_context` existe para rotulagem e avaliação. Ele não faz parte do
espaço de observação e deve ser filtrado por políticas visuais.

## Close

```json
{"type":"close","protocol_version":1}
```

O `close` encerra a sessão de controle, mas não precisa fechar o Minecraft.
