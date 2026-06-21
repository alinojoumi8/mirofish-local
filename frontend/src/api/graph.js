import service, { requestWithRetry } from './index'

/**
 * Generate ontology (upload documents and simulation requirements)
 * @param {Object} data - Contains files, simulation_requirement, project_name, etc.
 * @returns {Promise}
 */
export function generateOntology(formData) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/ontology/generate',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  )
}

/**
 * Build graph
 * @param {Object} data - Contains project_id, graph_name, etc.
 * @returns {Promise}
 */
export function buildGraph(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/build',
      method: 'post',
      data
    })
  )
}

/**
 * Query task status
 * @param {String} taskId - Task ID
 * @returns {Promise}
 */
export function getTaskStatus(taskId) {
  return service({
    url: `/api/graph/task/${taskId}`,
    method: 'get'
  })
}

/**
 * Get graph data
 * @param {String} graphId - Graph ID
 * @returns {Promise}
 */
export function getGraphData(graphId) {
  return service({
    url: `/api/graph/data/${graphId}`,
    method: 'get'
  })
}

/**
 * Get graph embedding readiness and coverage
 * @param {String} graphId - Graph ID
 * @returns {Promise}
 */
export function getGraphEmbeddingStatus(graphId) {
  return service({
    url: `/api/graph/${graphId}/embedding-status`,
    method: 'get'
  })
}

/**
 * Get structural graph quality and synthesis readiness
 * @param {String} graphId - Graph ID
 * @returns {Promise}
 */
export function getGraphQuality(graphId) {
  return service({
    url: `/api/graph/${graphId}/quality`,
    method: 'get'
  })
}

/**
 * Benchmark embedding providers against this graph
 * @param {String} graphId - Graph ID
 * @param {Object} data - { providers?, queries?, limit? }
 * @returns {Promise}
 */
export function benchmarkGraphEmbeddings(graphId, data = {}) {
  return service({
    url: `/api/graph/${graphId}/benchmark-embeddings`,
    method: 'post',
    data
  })
}

/**
 * Repair/backfill graph embeddings
 * @param {String} graphId - Graph ID
 * @param {Object} data - { batch_size? }
 * @returns {Promise}
 */
export function reembedGraph(graphId, data = {}) {
  return service({
    url: `/api/graph/${graphId}/reembed`,
    method: 'post',
    data
  })
}

/**
 * Get project information
 * @param {String} projectId - Project ID
 * @returns {Promise}
 */
export function getProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}`,
    method: 'get'
  })
}

export function createCase(data) {
  return service({
    url: '/api/graph/case/create',
    method: 'post',
    data
  })
}

export function listCases(limit = 50) {
  return service({
    url: '/api/graph/case/list',
    method: 'get',
    params: { limit }
  })
}

export function getCase(caseId) {
  return service({
    url: `/api/graph/case/${caseId}`,
    method: 'get'
  })
}

export function listCaseVersions(caseId) {
  return service({
    url: `/api/graph/case/${caseId}/versions`,
    method: 'get'
  })
}

export function compareCaseVersions(caseId, data) {
  return service({
    url: `/api/graph/case/${caseId}/compare`,
    method: 'post',
    data
  })
}
