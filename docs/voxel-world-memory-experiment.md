# Experimento de memória voxel esparsa

## Hipótese

O mundo Minecraft já é composto por blocos inteiros em `(x, y, z)`. A memória
espacial canônica pode, portanto, preservar a geometria como voxels, enquanto
o minimapa 2D passa a ser somente uma projeção para navegação e interface.

O experimento não entrega ao agente todos os chunks carregados. Ele registra
apenas voxels atravessados ou atingidos pelos mesmos raios disponíveis no
rollout. Isso evita que a política receba uma visão privilegiada do mundo.

## Representação

- armazenamento esparso em chunks `16×16×16`;
- `unknown`, raios que atravessaram, bloco confirmado e material por voxel;
- confiança derivada da quantidade de observações;
- `last_seen_tick`, fonte e versão por chunk;
- eventos autoritativos de quebrar/colocar substituem evidência obsoleta;
- janela local densa `C×Y×Z×X` somente no momento de alimentar a política;
- minimapa derivado conservador: desconhecido nunca vira livre por omissão.

Os canais do volume local são `known`, `solid_probability`, `material_class`,
`confidence` e `visited`. O primeiro teste usa uma janela `33×17×33`: ela
representa 33 blocos nos eixos horizontais e oito blocos acima/abaixo do centro
sem criar um tensor denso do mundo inteiro.

`free_count` e `solid_count` não são complementares. Um bloco pode possuir
shape parcial — muro, cerca, escada, porta ou vegetação — e diferentes raios
podem atravessar o volume unitário e atingir sua geometria. Portanto, qualquer
hit confirma `block_present`; raios livres permanecem como evidência de
passagem, e a coexistência dos dois é marcada como `partial_geometry`. O nome
`solid_probability` foi mantido temporariamente na projeção compatível, mas
vale `0/1`; não é a razão artificial `hits/(hits+free)`.

## Gate do experimento

1. O DDA 3D deve preservar raios horizontais, verticais e diagonais.
2. Um hit deve marcar o percurso anterior como livre e o voxel exato como sólido.
3. Coordenadas negativas devem manter chunk/local corretos.
4. Uma ponte sobre um túnel deve permanecer como duas superfícies distintas no
   mesmo `(x,z)` no volume 3D.
5. A projeção 2D deve marcar como não atravessável uma coluna com teto baixo e
   manter regiões não observadas como desconhecidas.

## Limites antes de uso no treino

- A semântica de materiais ainda é uma classe grosseira, não um embedding.
- Líquidos e blocos não cúbicos precisam de uma classe de colisão explícita.
- O update voxel ainda não é um contrato persistido no dataset v2.
- A política atual continua consumindo o mapa 2D; trocar para convolução 3D é
  uma etapa separada, condicionada ao resultado deste teste.
