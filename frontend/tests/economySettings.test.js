import assert from 'node:assert/strict'
import test from 'node:test'

import {
  DEFAULT_ECONOMY_SETTINGS,
  economySettingsFromQuery,
  economySettingsToQuery,
  normalizeEconomySettings
} from '../src/utils/economySettings.js'


test('economic twin defaults to disabled and round-trips through route query', () => {
  assert.deepEqual(
    economySettingsFromQuery(economySettingsToQuery(DEFAULT_ECONOMY_SETTINGS)),
    DEFAULT_ECONOMY_SETTINGS
  )
  assert.equal(DEFAULT_ECONOMY_SETTINGS.enabled, false)
})


test('economic twin settings normalize currency and validate numeric limits', () => {
  assert.deepEqual(normalizeEconomySettings({
    enabled: true,
    initial_balance_cents: '2500',
    currency: 'cad',
    max_decisions_per_tick: '500'
  }), {
    enabled: true,
    initial_balance_cents: 2500,
    currency: 'CAD',
    max_decisions_per_tick: 500
  })

  assert.throws(
    () => normalizeEconomySettings({ max_decisions_per_tick: 501 }),
    /between 1 and 500/
  )
  assert.throws(
    () => normalizeEconomySettings({ initial_balance_cents: -1 }),
    /non-negative integer/
  )
})
