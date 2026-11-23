'use strict'

// Canonical Prismarine producer for minecraft-spatial-map-update/v1.  The
// BrincaBot camera injects its installed THREE/Vec3/world so this file remains
// versioned with the Gym contracts instead of becoming Observer-only logic.

function finitePositive(name, value) {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`)
  return value
}

function vectorArray(vector) {
  return [vector.x, vector.y, vector.z]
}

function samplePixels(width, height, columns, rows) {
  if (columns > width || rows > height) throw new Error('ray grid exceeds frame resolution')
  const pixels = []
  for (let row = 0; row < rows; row += 1) {
    const y = Math.min(height - 1, Math.floor((row + 0.5) * height / rows))
    for (let column = 0; column < columns; column += 1) {
      const x = Math.min(width - 1, Math.floor((column + 0.5) * width / columns))
      pixels.push([x, y])
    }
  }
  return pixels
}

function confirmedLoadedDistance(world, Vec3, origin, direction, maxDistance, step) {
  let lastConfirmed = 0
  for (let distance = 0; distance <= maxDistance; distance += step) {
    const point = new Vec3(
      origin.x + direction.x * distance,
      origin.y + direction.y * distance,
      origin.z + direction.z * distance,
    )
    if (!world.getColumnAt(point)) return { distance: lastConfirmed, unloaded: true }
    lastConfirmed = distance
  }
  return { distance: maxDistance, unloaded: false }
}

function createPrismarineSpatialSensor({
  THREE,
  Vec3,
  world,
  width,
  height,
  verticalFovDegrees = 75,
  columns = 40,
  rows = 23,
  maxDistance = 64,
  playerHeight = 1.6,
  loadedProbeStep = 0.5,
  actorId = 'actor:bot',
  sensorId = 'sensor:bot:prismarine-raycast',
}) {
  if (!THREE?.PerspectiveCamera || !THREE?.Raycaster) throw new Error('THREE camera API is required')
  if (typeof Vec3 !== 'function') throw new Error('Vec3 constructor is required')
  if (!world?.raycast || !world?.getColumnAt) throw new Error('Prismarine world API is required')
  finitePositive('width', width)
  finitePositive('height', height)
  finitePositive('verticalFovDegrees', verticalFovDegrees)
  finitePositive('columns', columns)
  finitePositive('rows', rows)
  finitePositive('maxDistance', maxDistance)
  finitePositive('loadedProbeStep', loadedProbeStep)
  const pixels = samplePixels(width, height, columns, rows)

  function capture({
    entity,
    serverTick,
    monotonicNs,
    cameraFrameId,
    cameraStreamId = 'view:bot:egocentric',
  }) {
    if (!entity?.position) throw new Error('camera entity is unavailable')
    if (!Number.isInteger(serverTick) || serverTick < 0) throw new Error('serverTick is invalid')
    if (!cameraFrameId) throw new Error('cameraFrameId is required')
    const camera = new THREE.PerspectiveCamera(
      verticalFovDegrees,
      width / height,
      0.1,
      1000,
    )
    camera.position.set(entity.position.x, entity.position.y + playerHeight, entity.position.z)
    camera.rotation.set(entity.pitch || 0, entity.yaw || 0, 0, 'ZYX')
    camera.updateProjectionMatrix()
    camera.updateMatrixWorld(true)

    const origin = camera.position.clone()
    const forward = camera.getWorldDirection(new THREE.Vector3()).normalize()
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize()
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize()
    const raycaster = new THREE.Raycaster()
    const samples = pixels.map(([pixelX, pixelY]) => {
      const ndc = new THREE.Vector2(
        2 * (pixelX + 0.5) / width - 1,
        1 - 2 * (pixelY + 0.5) / height,
      )
      raycaster.setFromCamera(ndc, camera)
      const direction = raycaster.ray.direction.clone().normalize()
      const loaded = confirmedLoadedDistance(
        world,
        Vec3,
        origin,
        direction,
        maxDistance,
        loadedProbeStep,
      )
      const effectiveRange = loaded.distance
      const hit = effectiveRange > 0
        ? world.raycast(
          new Vec3(origin.x, origin.y, origin.z),
          new Vec3(direction.x, direction.y, direction.z),
          effectiveRange,
        )
        : null
      const base = {
        interface: 'minecraft-raycast-sample/v1',
        pixel_x: pixelX,
        pixel_y: pixelY,
        origin: vectorArray(origin),
        direction: vectorArray(direction),
        max_distance: maxDistance,
        intersection_source: 'prismarine_world_shapes',
        // The stock browser viewer eases pose updates over 50 ms. Until its
        // exact rendered camera matrix is exported, moving-frame projection is
        // bounded to one tick but must not claim perfect pixel calibration.
        confidence: 0.95,
      }
      if (hit?.intersect && hit?.position && hit?.name && Number.isInteger(hit.stateId)) {
        const contact = hit.intersect
        return {
          ...base,
          traversed_distance: origin.distanceTo(contact),
          termination: 'block_hit',
          contact_point: vectorArray(contact),
          block_position: [hit.position.x, hit.position.y, hit.position.z],
          block_name: hit.name,
          block_state_id: hit.stateId,
          face: hit.face ?? null,
        }
      }
      return {
        ...base,
        traversed_distance: effectiveRange,
        termination: loaded.unloaded ? 'unloaded_chunk' : 'max_range',
        contact_point: null,
        block_position: null,
        block_name: null,
        block_state_id: null,
        face: null,
      }
    })

    return {
      interface: 'minecraft-raycast-frame/v1',
      sensor_id: sensorId,
      actor_id: actorId,
      server_tick: serverTick,
      monotonic_ns: Number(monotonicNs),
      camera_stream_id: cameraStreamId,
      camera_frame_id: cameraFrameId,
      calibration: {
        interface: 'minecraft-camera-calibration/v1',
        width,
        height,
        vertical_fov_degrees: verticalFovDegrees,
        near_plane: 0.1,
        far_plane: 1000,
        projection: 'perspective',
        renderer: 'prismarine_viewer',
        pixel_center_convention: 'half_pixel',
      },
      sampling_pattern: 'uniform_grid',
      sampling_columns: columns,
      sampling_rows: rows,
      pose: {
        interface: 'minecraft-camera-pose/v1',
        actor_id: actorId,
        server_tick: serverTick,
        monotonic_ns: Number(monotonicNs),
        origin: vectorArray(origin),
        forward: vectorArray(forward),
        right: vectorArray(right),
        up: vectorArray(up),
        yaw_degrees: (entity.yaw || 0) * 180 / Math.PI,
        pitch_degrees: (entity.pitch || 0) * 180 / Math.PI,
        source: 'prismarine_camera',
      },
      samples,
    }
  }

  return { capture, sampleCount: pixels.length }
}

module.exports = {
  createPrismarineSpatialSensor,
  samplePixels,
}
