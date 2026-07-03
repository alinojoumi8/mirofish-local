import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { resolveApiBaseUrl } from './baseUrl.js'

describe('resolveApiBaseUrl', () => {
  it('uses the same origin by default so Vite can proxy API calls', () => {
    assert.equal(resolveApiBaseUrl({}), '')
  })

  it('ignores VITE_API_BASE_URL in development so preview tunnels can proxy API calls', () => {
    assert.equal(
      resolveApiBaseUrl({ DEV: true, VITE_API_BASE_URL: 'http://127.0.0.1:5101' }),
      ''
    )
  })

  it('keeps an explicit VITE_API_BASE_URL override in production', () => {
    assert.equal(
      resolveApiBaseUrl({ DEV: false, VITE_API_BASE_URL: 'http://backend.example.test:5001' }),
      'http://backend.example.test:5001'
    )
  })

  it('can opt into an absolute API base URL in development', () => {
    assert.equal(
      resolveApiBaseUrl({
        DEV: true,
        VITE_USE_ABSOLUTE_API_BASE_URL: 'true',
        VITE_API_BASE_URL: 'http://backend.example.test:5001',
      }),
      'http://backend.example.test:5001'
    )
  })
})
