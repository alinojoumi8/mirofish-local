export const DEFAULT_ECONOMY_SETTINGS = Object.freeze({
  enabled: false,
  initial_balance_cents: 1000000,
  currency: 'USD',
  max_decisions_per_tick: 50
})

export const normalizeEconomySettings = (settings = {}) => {
  const normalized = {
    ...DEFAULT_ECONOMY_SETTINGS,
    ...settings,
    enabled: settings.enabled === true || settings.enabled === 'true'
  }
  normalized.initial_balance_cents = Number(normalized.initial_balance_cents)
  normalized.max_decisions_per_tick = Number(normalized.max_decisions_per_tick)
  normalized.currency = String(normalized.currency || '').trim().toUpperCase()

  if (!Number.isInteger(normalized.initial_balance_cents) || normalized.initial_balance_cents < 0) {
    throw new Error('Initial balance must be a non-negative integer number of cents.')
  }
  if (!Number.isInteger(normalized.max_decisions_per_tick)
      || normalized.max_decisions_per_tick < 1
      || normalized.max_decisions_per_tick > 500) {
    throw new Error('Decisions per tick must be an integer between 1 and 500.')
  }
  if (!/^[A-Z0-9]{3,8}$/.test(normalized.currency)) {
    throw new Error('Currency must be a 3-8 character alphanumeric code.')
  }
  return normalized
}

export const economySettingsToQuery = (settings) => {
  const normalized = normalizeEconomySettings(settings)
  return {
    economyEnabled: String(normalized.enabled),
    economyInitialBalanceCents: String(normalized.initial_balance_cents),
    economyCurrency: normalized.currency,
    economyMaxDecisionsPerTick: String(normalized.max_decisions_per_tick)
  }
}

export const economySettingsFromQuery = (query = {}) => {
  try {
    return normalizeEconomySettings({
      enabled: query.economyEnabled ?? DEFAULT_ECONOMY_SETTINGS.enabled,
      initial_balance_cents: query.economyInitialBalanceCents
        ?? DEFAULT_ECONOMY_SETTINGS.initial_balance_cents,
      currency: query.economyCurrency ?? DEFAULT_ECONOMY_SETTINGS.currency,
      max_decisions_per_tick: query.economyMaxDecisionsPerTick
        ?? DEFAULT_ECONOMY_SETTINGS.max_decisions_per_tick
    })
  } catch {
    return { ...DEFAULT_ECONOMY_SETTINGS }
  }
}
