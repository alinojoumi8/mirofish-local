/**
 * Temporarily store files and requirements to be uploaded
 * Used to immediately navigate after clicking Start Engine on home page, API call is made on Process page
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  urls: [],
  simulationRequirement: '',
  // When set, the upload is attached as a new version of an existing case instead of
  // spawning a brand-new (single-version) case.
  caseId: null,
  isPending: false
})

export function setPendingUpload(files, requirement, urls = [], caseId = null) {
  state.files = files
  state.urls = urls
  state.simulationRequirement = requirement
  state.caseId = caseId
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    urls: state.urls,
    simulationRequirement: state.simulationRequirement,
    caseId: state.caseId,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.urls = []
  state.simulationRequirement = ''
  state.caseId = null
  state.isPending = false
}

export default state
