/** Create an idempotent, non-overlapping interval owner. */
export function createPoller(callback, intervalMs, options = {}) {
  const setIntervalFn = options.setIntervalFn || globalThis.setInterval
  const clearIntervalFn = options.clearIntervalFn || globalThis.clearInterval
  let timer = null
  let inFlight = false

  const tick = async () => {
    if (inFlight) return
    inFlight = true
    try {
      await callback()
    } finally {
      inFlight = false
    }
  }

  return {
    start({ immediate = false } = {}) {
      if (timer !== null) return
      if (immediate) void tick()
      timer = setIntervalFn(tick, intervalMs)
    },
    stop() {
      if (timer === null) return
      clearIntervalFn(timer)
      timer = null
    },
    isRunning() {
      return timer !== null
    },
    tick,
  }
}
