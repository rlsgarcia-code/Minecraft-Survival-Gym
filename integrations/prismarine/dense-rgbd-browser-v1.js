'use strict'

// Injected into the Prismarine browser before its bundle is loaded.  The
// integration patches THREE.WebGLRenderer only after Prismarine publishes
// THREE, and captures RGB plus metric camera-Z depth from one immutable scene
// and camera state.  Nothing is captured unless requestCapture() is called.

;(function installMinecraftGymDenseRgbd () {
  const API_NAME = '__minecraftGymDenseRgbd'
  const state = {
    installed: false,
    rendererReady: false,
    pending: null,
    captureSequence: 0,
    lastError: null,
    kickCapture: null,
  }

  function bytesToBase64 (bytes) {
    const chunk = 0x8000
    let binary = ''
    for (let offset = 0; offset < bytes.length; offset += chunk) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk))
    }
    return btoa(binary)
  }

  function flipRgbaRows (source, width, height) {
    const stride = width * 4
    const result = new Uint8Array(source.length)
    for (let row = 0; row < height; row += 1) {
      const sourceOffset = (height - 1 - row) * stride
      result.set(source.subarray(sourceOffset, sourceOffset + stride), row * stride)
    }
    return result
  }

  function rgbaToJpegBase64 (rgba, width, height, quality) {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d', { alpha: false })
    const image = new ImageData(new Uint8ClampedArray(rgba.buffer), width, height)
    context.putImageData(image, 0, 0)
    return canvas.toDataURL('image/jpeg', quality).split(',', 2)[1]
  }

  async function deflateBytes (bytes) {
    if (typeof CompressionStream !== 'function') {
      throw new Error('browser CompressionStream(deflate) is unavailable')
    }
    const compressed = new Blob([bytes])
      .stream()
      .pipeThrough(new CompressionStream('deflate'))
    return new Uint8Array(await new Response(compressed).arrayBuffer())
  }

  function buildResources (THREE, width, height, renderer) {
    const colorTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
    })
    colorTarget.depthTexture = new THREE.DepthTexture(width, height)
    colorTarget.depthTexture.format = THREE.DepthFormat
    colorTarget.depthTexture.type = THREE.UnsignedIntType
    colorTarget.texture.encoding = renderer.outputEncoding

    const encodedDepthTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: false,
      stencilBuffer: false,
    })
    encodedDepthTarget.texture.encoding = THREE.LinearEncoding

    const material = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: colorTarget.depthTexture },
        cameraNear: { value: 0.1 },
        cameraFar: { value: 1000.0 },
        maxDistance: { value: 64.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;
        uniform float maxDistance;

        float perspectiveDepthToViewZ(float depth, float near, float far) {
          return (near * far) / ((far - near) * depth - far);
        }

        void main() {
          float projected = texture2D(tDepth, vUv).x;
          float cameraZ = -perspectiveDepthToViewZ(projected, cameraNear, cameraFar);
          bool measured = projected < 0.999999 && cameraZ <= maxDistance;
          float blocks = measured ? cameraZ : maxDistance;
          float quantized = floor(clamp(blocks * 100.0 + 0.5, 0.0, 65535.0));
          float highByte = floor(quantized / 256.0);
          float lowByte = quantized - highByte * 256.0;
          float status = measured ? 1.0 : 2.0;
          gl_FragColor = vec4(lowByte / 255.0, highByte / 255.0, status / 255.0, 1.0);
        }
      `,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
      blending: THREE.NoBlending,
    })
    const postScene = new THREE.Scene()
    const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material))
    return { width, height, colorTarget, encodedDepthTarget, material, postScene, postCamera }
  }

  function disposeResources (resources) {
    if (!resources) return
    resources.colorTarget.dispose()
    resources.encodedDepthTarget.dispose()
    resources.material.dispose()
    for (const child of resources.postScene.children) child.geometry?.dispose()
  }

  function patchRenderer (renderer, THREE) {
    if (renderer.__minecraftGymDenseRgbdPatched) return
    renderer.__minecraftGymDenseRgbdPatched = true
    state.rendererReady = true
    const originalRender = renderer.render.bind(renderer)
    let resources = null
    let lastScene = null
    let lastCamera = null

    renderer.render = function minecraftGymRender (scene, camera) {
      const request = state.pending
      if (!request || request.capturing) {
        lastScene = scene
        lastCamera = camera
        return originalRender(scene, camera)
      }
      request.capturing = true
      const started = performance.now()
      let previousTarget = null
      let previousAutoClear = true
      let previousXr = false
      let rendererStateChanged = false
      try {
        const drawingSize = this.getDrawingBufferSize(new THREE.Vector2())
        const width = Math.floor(drawingSize.x)
        const height = Math.floor(drawingSize.y)
        if (!resources || resources.width !== width || resources.height !== height) {
          disposeResources(resources)
          resources = buildResources(THREE, width, height, this)
        }
        previousTarget = this.getRenderTarget()
        previousAutoClear = this.autoClear
        previousXr = this.xr.enabled
        rendererStateChanged = true
        this.xr.enabled = false
        this.autoClear = true

        const renderStarted = performance.now()
        this.setRenderTarget(resources.colorTarget)
        this.clear(true, true, true)
        originalRender(scene, camera)
        const colorBytes = new Uint8Array(width * height * 4)
        this.readRenderTargetPixels(resources.colorTarget, 0, 0, width, height, colorBytes)
        const renderReadMs = performance.now() - renderStarted

        let depthPacked = null
        let depthReadMs = 0
        if (request.options.includeDepth) {
          const depthStarted = performance.now()
          resources.material.uniforms.cameraNear.value = camera.near
          resources.material.uniforms.cameraFar.value = camera.far
          resources.material.uniforms.maxDistance.value = request.options.maxDistance
          this.setRenderTarget(resources.encodedDepthTarget)
          this.clear(true, true, true)
          originalRender(resources.postScene, resources.postCamera)
          const encoded = new Uint8Array(width * height * 4)
          this.readRenderTargetPixels(
            resources.encodedDepthTarget, 0, 0, width, height, encoded,
          )
          const flipped = flipRgbaRows(encoded, width, height)
          const packed = new Uint8Array(width * height * 3)
          for (let pixel = 0; pixel < width * height; pixel += 1) {
            packed[pixel * 3] = flipped[pixel * 4]
            packed[pixel * 3 + 1] = flipped[pixel * 4 + 1]
            packed[pixel * 3 + 2] = flipped[pixel * 4 + 2]
          }
          depthPacked = packed
          depthReadMs = performance.now() - depthStarted
        }

        this.setRenderTarget(previousTarget)
        this.autoClear = previousAutoClear
        this.xr.enabled = previousXr
        const rgbStarted = performance.now()
        const rgbBase64 = rgbaToJpegBase64(
          flipRgbaRows(colorBytes, width, height), width, height, request.options.jpegQuality,
        )
        const rgbEncodeMs = performance.now() - rgbStarted
        void (async () => {
          const compressionStarted = performance.now()
          const compressedDepth = depthPacked === null
            ? null
            : await deflateBytes(depthPacked)
          const depthCompressionMs = performance.now() - compressionStarted
          state.captureSequence += 1
          state.pending = null
          request.resolve({
            interface: 'minecraft-dense-rgbd-browser-capture/v1',
            capture_sequence: state.captureSequence,
            width,
            height,
            rgb_jpeg_base64: rgbBase64,
            depth_zlib_base64: compressedDepth === null
              ? null
              : bytesToBase64(compressedDepth),
            depth_encoding: request.options.includeDepth
              ? 'uint16_le_centiblocks+uint8_status_interleaved'
              : null,
            max_distance: request.options.maxDistance,
            camera_near: camera.near,
            camera_far: camera.far,
            vertical_fov_degrees: camera.fov,
            timings_ms: {
              render_and_rgb_read: renderReadMs,
              depth_encode_and_read: depthReadMs,
              depth_compression: depthCompressionMs,
              jpeg_encode: rgbEncodeMs,
              total: performance.now() - started,
            },
          })
        })().catch(error => {
          state.lastError = String(error?.stack || error)
          state.pending = null
          request.reject(new Error(state.lastError))
        })
      } catch (error) {
        if (rendererStateChanged) {
          this.setRenderTarget(previousTarget)
          this.autoClear = previousAutoClear
          this.xr.enabled = previousXr
        }
        state.lastError = String(error?.stack || error)
        state.pending = null
        request.reject(new Error(state.lastError))
      }
      return undefined
    }
    state.kickCapture = () => {
      if (!lastScene || !lastCamera) throw new Error('renderer has not produced its first frame')
      renderer.render(lastScene, lastCamera)
    }

  }

  function install (renderer, THREE) {
    if (!renderer?.render || !THREE?.WebGLRenderTarget) return false
    patchRenderer(renderer, THREE)
    state.installed = true
    return true
  }

  globalThis[API_NAME] = {
    interface: 'minecraft-dense-rgbd-browser/v1',
    attachRenderer (renderer, THREE) {
      if (!install(renderer, THREE)) {
        throw new Error('compatible THREE renderer is required')
      }
      return this.status()
    },
    requestCapture (options = {}) {
      if (state.pending) return Promise.reject(new Error('RGB-D capture is already pending'))
      const pending = new Promise((resolve, reject) => {
        state.pending = {
          resolve,
          reject,
          capturing: false,
          options: {
            includeDepth: options.includeDepth !== false,
            maxDistance: Number(options.maxDistance || 64),
            jpegQuality: Number(options.jpegQuality || 0.82),
          },
        }
      })
      try {
        state.kickCapture()
      } catch (error) {
        const request = state.pending
        state.pending = null
        request.reject(error)
      }
      return pending
    },
    status () {
      return {
        interface: 'minecraft-dense-rgbd-browser-status/v1',
        installed: state.installed,
        renderer_ready: state.rendererReady,
        pending: Boolean(state.pending),
        capture_sequence: state.captureSequence,
        last_error: state.lastError,
      }
    },
  }

})()
