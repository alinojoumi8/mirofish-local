import service from './index'

/**
 * Resolve a market symbol's current value + suggested numeric outcome bands.
 * @param {String} symbol - e.g. "^GSPC" (price) or "CPIAUCSL" (FRED indicator)
 * @param {String} kind - "price" | "indicator"
 * @param {String} horizon - e.g. "1m", "3m"
 */
export function resolveMarketTarget(symbol, kind = 'price', horizon = '1m') {
  return service({
    url: '/api/market/resolve',
    method: 'get',
    params: { symbol, kind, horizon }
  })
}

export function getMarketStatus() {
  return service({ url: '/api/market/status', method: 'get' })
}
