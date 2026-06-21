import service, { requestWithRetry } from './index'

/**
 * Start report generation
 * @param {Object} data - { simulation_id, force_regenerate? }
 */
export const generateReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/generate', data), 3, 1000)
}

/**
 * Get report generation status
 * @param {Object} data - { task_id?, simulation_id? }
 */
export const getReportStatus = (data) => {
  return service.post('/api/report/generate/status', data)
}

/**
 * Get persisted report progress by report id
 * @param {string} reportId
 */
export const getReportProgress = (reportId) => {
  return service.get(`/api/report/${reportId}/progress`)
}

/**
 * Get Agent log (incremental)
 * @param {string} reportId
 * @param {number} fromLine - Start from which line
 */
export const getAgentLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/agent-log`, { params: { from_line: fromLine } })
}

/**
 * Get console log (incremental)
 * @param {string} reportId
 * @param {number} fromLine - Start from which line
 */
export const getConsoleLog = (reportId, fromLine = 0) => {
  return service.get(`/api/report/${reportId}/console-log`, { params: { from_line: fromLine } })
}

/**
 * Get report details
 * @param {string} reportId
 */
export const getReport = (reportId) => {
  return service.get(`/api/report/${reportId}`)
}

/**
 * Regenerate one report section
 * @param {string} reportId
 * @param {number} sectionIndex
 * @param {Object} data - { disable_interviews?, strict_antirepetition? }
 */
export const regenerateReportSection = (reportId, sectionIndex, data = {}) => {
  return requestWithRetry(
    () => service.post(`/api/report/${reportId}/section/${sectionIndex}/regenerate`, data),
    1,
    1000
  )
}

/**
 * Resume a failed or partial report from the first missing section
 * @param {string} reportId
 * @param {Object} data - { disable_interviews?, strict_antirepetition? }
 */
export const resumeReport = (reportId, data = {}) => {
  return requestWithRetry(
    () => service.post(`/api/report/${reportId}/resume`, data),
    2,
    1000
  )
}

/**
 * Chat with Report Agent
 * @param {Object} data - { simulation_id, message, chat_history? }
 */
export const chatWithReport = (data) => {
  return requestWithRetry(() => service.post('/api/report/chat', data), 3, 1000)
}
