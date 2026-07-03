export function resolveApiBaseUrl(env = {}) {
  if (env.DEV && env.VITE_USE_ABSOLUTE_API_BASE_URL !== 'true') {
    return ''
  }

  return env.VITE_API_BASE_URL || ''
}
