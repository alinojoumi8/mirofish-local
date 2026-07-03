<template>
  <div class="case-detail">
    <header class="app-header">
      <div class="brand" @click="router.push('/')">MIROFISH OFFLINE</div>
      <button class="link-btn" @click="router.push('/cases')">← Cases</button>
      <div class="spacer"></div>
      <button v-if="canCompare" class="btn-ghost" @click="goCompare">Compare versions</button>
      <button class="btn-primary" @click="showAdd = true">+ Add version</button>
    </header>

    <main class="content">
      <div v-if="loading" class="state-msg">Loading case…</div>
      <div v-else-if="error" class="state-msg error">{{ error }}</div>

      <template v-else-if="caseData">
        <div class="case-head">
          <h1>{{ caseData.name }}</h1>
          <p class="case-req">{{ caseData.simulation_requirement || 'No prediction goal set.' }}</p>
        </div>

        <div v-if="!versions.length" class="state-msg">
          No versions yet. Add a document batch to build the first version.
        </div>

        <div v-else class="version-timeline">
          <div v-for="v in versions" :key="v.version_id" class="version-row">
            <div class="version-badge mono">v{{ v.version_number }}</div>
            <div class="version-body">
              <div class="version-top">
                <span class="version-status" :class="statusClass(v.status)">{{ v.status }}</span>
                <span class="version-date mono">{{ formatDate(v.created_at) }}</span>
                <span v-if="v.documents?.length" class="version-docs">{{ v.documents.length }} doc(s)</span>
              </div>

              <div v-if="prediction(v)" class="version-prediction">
                <div class="pred-head">
                  <span class="pred-confidence">{{ prediction(v).confidence || '—' }} confidence</span>
                  <span v-if="prediction(v).pointEstimate !== null" class="pred-point mono">
                    est {{ prediction(v).pointEstimate }}
                  </span>
                </div>
                <div class="pred-bars">
                  <div v-for="p in prediction(v).top" :key="p.outcome" class="pred-bar-row">
                    <span class="pred-outcome">{{ p.outcome }}</span>
                    <span class="pred-pct mono">{{ p.pct }}%</span>
                  </div>
                </div>
              </div>
              <div v-else class="version-nopred">No prediction recorded for this version yet.</div>

              <div class="version-actions">
                <button v-if="v.project_id" class="mini-btn" @click="openProject(v.project_id)">Open project</button>
                <button v-if="v.report_ids?.length" class="mini-btn" @click="openReport(v.report_ids[v.report_ids.length - 1])">
                  View report
                </button>
              </div>
            </div>
          </div>
        </div>
      </template>
    </main>

    <!-- Add version modal -->
    <div v-if="showAdd" class="modal-overlay" @click.self="showAdd = false">
      <div class="modal">
        <h3>Add version to case</h3>
        <p class="modal-note">Upload a new document batch. It runs the full analyze &amp; build pipeline as version {{ versions.length + 1 }}.</p>
        <label class="modal-field">
          <span>Documents</span>
          <input type="file" multiple @change="onFiles" />
        </label>
        <label class="modal-field">
          <span>Prediction goal (optional override)</span>
          <textarea v-model="addRequirement" rows="2" :placeholder="caseData?.simulation_requirement || 'Prediction goal'"></textarea>
        </label>
        <div class="modal-actions">
          <button class="btn-ghost" @click="showAdd = false">Cancel</button>
          <button class="btn-primary" :disabled="!addFiles.length" @click="startAddVersion">Build version</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { getCase } from '../api/graph'
import { setPendingUpload } from '../store/pendingUpload'

const props = defineProps({ caseId: String })
const router = useRouter()

const caseData = ref(null)
const loading = ref(true)
const error = ref('')

const showAdd = ref(false)
const addFiles = ref([])
const addRequirement = ref('')

const versions = computed(() => caseData.value?.versions || [])
const canCompare = computed(() => versions.value.filter(v => v.prediction_snapshots?.length).length >= 2)

const loadCase = async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getCase(props.caseId)
    if (res.success) {
      caseData.value = res.data
    } else {
      error.value = res.error || 'Failed to load case'
    }
  } catch (err) {
    error.value = err.message || 'Failed to load case'
  } finally {
    loading.value = false
  }
}

// Latest prediction snapshot summarized for display, preferring the ordered
// `_meta.probabilities` list, falling back to the flat outcome→probability keys.
const prediction = (version) => {
  const snap = version.prediction_snapshots?.[version.prediction_snapshots.length - 1]
  if (!snap) return null
  const meta = snap._meta || {}
  let rows = []
  if (Array.isArray(meta.probabilities) && meta.probabilities.length) {
    rows = meta.probabilities.map(p => ({ outcome: p.outcome, prob: Number(p.probability || 0) }))
  } else {
    rows = Object.entries(snap)
      .filter(([k, val]) => !k.startsWith('_') && k !== 'confidence' && k !== 'point_estimate' && typeof val === 'number')
      .map(([outcome, prob]) => ({ outcome, prob: Number(prob) }))
  }
  rows.sort((a, b) => b.prob - a.prob)
  return {
    confidence: snap.confidence,
    pointEstimate: snap.point_estimate ?? null,
    top: rows.slice(0, 3).map(r => ({ outcome: r.outcome, pct: Math.round(r.prob * 100) }))
  }
}

const goCompare = () => router.push({ name: 'CaseCompare', params: { caseId: props.caseId } })
const openProject = (projectId) => router.push({ name: 'Process', params: { projectId } })
const openReport = (reportId) => router.push({ name: 'Report', params: { reportId } })

const onFiles = (e) => { addFiles.value = Array.from(e.target.files || []) }

const startAddVersion = () => {
  if (!addFiles.value.length) return
  const requirement = addRequirement.value.trim() || caseData.value?.simulation_requirement || ''
  setPendingUpload(addFiles.value, requirement, [], props.caseId)
  showAdd.value = false
  router.push({ name: 'Process', params: { projectId: 'new' } })
}

const statusClass = (status) => {
  if (status === 'reported') return 'reported'
  if (status === 'built' || status === 'ready') return 'ready'
  return 'pending'
}
const formatDate = (iso) => {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return '—'
  }
}

onMounted(loadCase)
</script>

<style scoped>
.case-detail { min-height: 100vh; background: #FFF; font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif; color: #111; }
.app-header { height: 60px; border-bottom: 1px solid #EAEAEA; display: flex; align-items: center; gap: 14px; padding: 0 24px; }
.brand { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 18px; letter-spacing: 1px; cursor: pointer; }
.link-btn { border: none; background: transparent; color: #6B7280; cursor: pointer; font-size: 13px; }
.spacer { flex: 1; }

.content { max-width: 900px; margin: 0 auto; padding: 32px 24px; }
.state-msg { color: #6B7280; font-size: 14px; padding: 24px 0; text-align: center; }
.state-msg.error { color: #B91C1C; }

.case-head h1 { margin: 0 0 8px; font-size: 24px; }
.case-req { margin: 0 0 28px; color: #6B7280; font-size: 14px; line-height: 1.55; }

.version-timeline { display: flex; flex-direction: column; gap: 14px; }
.version-row { display: flex; gap: 14px; }
.version-badge {
  flex: 0 0 auto; width: 40px; height: 40px; border-radius: 999px;
  display: flex; align-items: center; justify-content: center;
  background: #111; color: #FFF; font-size: 12px; font-weight: 800;
}
.version-body { flex: 1; border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px 16px; }
.version-top { display: flex; align-items: center; gap: 12px; }
.version-status { font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; }
.version-status.reported { color: #065F46; background: #D1FAE5; }
.version-status.ready { color: #1E40AF; background: #DBEAFE; }
.version-status.pending { color: #92400E; background: #FEF3C7; }
.version-date { font-size: 12px; color: #9CA3AF; }
.version-docs { font-size: 12px; color: #6B7280; }

.version-prediction { margin-top: 12px; padding: 10px 12px; border: 1px solid #ECFDF5; background: #F0FDF4; border-radius: 8px; }
.pred-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.pred-confidence { font-size: 11px; font-weight: 700; text-transform: uppercase; color: #047857; }
.pred-point { font-size: 13px; font-weight: 700; color: #065F46; }
.pred-bars { display: flex; flex-direction: column; gap: 4px; }
.pred-bar-row { display: flex; justify-content: space-between; font-size: 12px; color: #374151; }
.pred-pct { font-weight: 700; color: #111; }

.version-nopred { margin-top: 12px; font-size: 12px; color: #9CA3AF; font-style: italic; }
.version-actions { margin-top: 12px; display: flex; gap: 8px; }
.mini-btn { padding: 5px 12px; border: 1px solid #D1D5DB; background: #FFF; border-radius: 6px; font-size: 12px; cursor: pointer; }
.mini-btn:hover { background: #F3F4F6; }

.btn-ghost { padding: 8px 16px; border: 1px solid #D1D5DB; background: #FFF; border-radius: 6px; cursor: pointer; font-size: 13px; }
.btn-primary { padding: 8px 16px; border: none; background: #111; color: #FFF; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal { background: #FFF; border-radius: 12px; padding: 24px; width: 460px; max-width: 90vw; }
.modal h3 { margin: 0 0 8px; font-size: 16px; }
.modal-note { margin: 0 0 16px; font-size: 12px; color: #6B7280; line-height: 1.5; }
.modal-field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; font-size: 12px; font-weight: 600; color: #374151; }
.modal-field input, .modal-field textarea { padding: 9px 11px; border: 1px solid #D1D5DB; border-radius: 6px; font-size: 13px; font-family: inherit; }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px; }
</style>
