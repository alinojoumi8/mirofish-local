import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { routes } from './routes.js'

describe('application routes', () => {
  it('lazy-loads every page component', () => {
    assert.ok(routes.length > 0)
    for (const route of routes) {
      assert.equal(typeof route.component, 'function', `${route.name} should be lazy-loaded`)
    }
  })
})
