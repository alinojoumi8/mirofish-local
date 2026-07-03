<template>
  <div class="case-compare">
    <header class="app-header">
      <div class="brand" @click="router.push('/')">MIROFISH OFFLINE</div>
      <button class="link-btn" @click="back">← Case</button>
      <div class="header-title">Compare predictions</div>
    </header>

    <main class="content">
      <div v-if="loading" class="state-msg">Loading…</div>
      <div v-else-if="error" class="state-msg error">{{ error }}</div>

      <template v-else>
        <div class="picker-row">
          <label class="picker">
            <span>From version</span>
            <select v-model="fromId">
              <option v-for="v in versions" :key="v.version_id" :value="v.version_id" :disabled="!v.prediction_snapshots?.length">
                v{{ v.version_number }}{{ v.prediction_snapshots?.length ? '' : ' (no prediction)' }}
              </option>
            </select>
          </label>
          <span class="arrow">→</span>
          <label class="picker">
            <span>To version</span>
            <select v-model="toId">
              <option v-for="v in versions" :key="v.version_id" :value="v.version_id" :disabled="!v.prediction_snapshots?.length">
                v{{ v.version_number }}{{ v.prediction_snapshots?.length ? '' : ' (no prediction)' }}
              </option>
            </select>
          </label>
          <button class="btn-primary" :disabled="!fromId || !toId || fromId === toId || comparing" @click="runCompare">
            {{ comparing ? 'Comparing…' : 'Compare' }}
          </button>
        </div>

        <div v-if="compareError" class="state-msg error">{{ compareError }}</div>

        <div v-if="result" class="compare-result">
          <div v-if="!changedRows.length" class="state-msg">
            No differences between these versions' predictions (or one version has no recorded prediction yet).
          </div>
          <table v-else class="diff-table">
            <thead>
              <tr>
                <th>Prediction</th>
                <th class="num">From</th>
                <th class="num">To</th>
                <th class="num">Δ</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in changedRows" :key="row.key">
                <td>{{ row.label }}</td>
                <td class="num mono">{{ row.from }}</td>
                <td class="num mono">{{ row.to }}</td>
                <td class="num mono" :class="deltaClass(row.delta)">{{ row.deltaText }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getCase, compareCaseVersions } from '../api/graph'

const props = defineProps({ caseId: String })
const route = useRoute()
const router = useRouter()

const versions = ref([])
const loading = ref(true)
const error = ref('')

const fromId = ref('')
const toId = ref('')
const comparing = ref(false)
const compareError = ref('')
const result = ref(null)

const withPrediction = computed(() => versions.value.filter(v => v.prediction_snapshots?.length))

const loadCase = async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await getCase(props.caseId)
    if (res.success) {
      versions.value = res.data?.versions || []
      // Seed from query params, else the two most recent versions that have predictions.
      const predicted = withPrediction.value
      fromId.value = route.query.from || predicted[predicted.length - 2]?.version_id || ''
      toId.value = route.query.to || predicted[predicted.length - 1]?.version_id || ''
      if (fromId.value && toId.value && fromId.value !== toId.value) {
        await runCompare()
      }
    } else {
      error.value = res.error || 'Failed to load case'
    }
  } catch (err) {
    error.value = err.message || 'Failed to load case'
  } finally {
    loading.value = false
  }
}

const runCompare = async () => {
  if (!fromId.value || !toId.value || fromId.value === toId.value) return
  comparing.value = true
  compareError.value = ''
  try {
    const res = await compareCaseVersions(props.caseId, {
      from_version_id: fromId.value,
      to_version_id: toId.value
    })
    if (res.success) {
      result.value = res.data
    } else {
      compareError.value = res.error || 'Comparison failed'
    }
  } catch (err) {
    compareError.value = err.message || 'Comparison failed'
  } finally {
    comparing.value = false
  }
}

const fmt = (v) => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(v < 1 && v > -1 ? 3 : 2)
  return String(v)
}

const changedRows = computed(() => {
  const changed = result.value?.changed_predictions || {}
  return Object.entries(changed).map(([key, row]) => {
    const isPct = typeof row.from === 'number' && typeof row.to === 'number' && key !== 'point_estimate'
    const from = isPct ? `${Math.round(row.from * 100)}%` : fmt(row.from)
    const to = isPct ? `${Math.round(row.to * 100)}%` : fmt(row.to)
    let deltaText = '—'
    if (typeof row.delta === 'number') {
      deltaText = isPct
        ? `${row.delta >= 0 ? '+' : ''}${Math.round(row.delta * 100)}%`
        : `${row.delta >= 0 ? '+' : ''}${fmt(row.delta)}`
    }
    return { key, label: key, from, to, delta: row.delta, deltaText }
  })
})

const deltaClass = (delta) => {
  if (typeof delta !== 'number' || delta === 0) return ''
  return delta > 0 ? 'up' : 'down'
}

const back = () => router.push({ name: 'CaseDetail', params: { caseId: props.caseId } })

onMounted(loadCase)
</script>

<style scoped>
.case-compare { min-height: 100vh; background: #FFF; font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif; color: #111; }
.app-header { height: 60px; border-bottom: 1px solid #EAEAEA; display: flex; align-items: center; gap: 14px; padding: 0 24px; }
.brand { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 18px; letter-spacing: 1px; cursor: pointer; }
.link-btn { border: none; background: transparent; color: #6B7280; cursor: pointer; font-size: 13px; }
.header-title { font-weight: 700; color: #333; }

.content { max-width: 760px; margin: 0 auto; padding: 32px 24px; }
.state-msg { color: #6B7280; font-size: 14px; padding: 20px 0; text-align: center; }
.state-msg.error { color: #B91C1C; }

.picker-row { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
.picker { display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 600; color: #374151; }
.picker select { padding: 8px 10px; border: 1px solid #D1D5DB; border-radius: 6px; font-size: 13px; font-family: inherit; min-width: 160px; }
.arrow { padding-bottom: 8px; color: #9CA3AF; font-weight: 700; }
.btn-primary { padding: 9px 18px; border: none; background: #111; color: #FFF; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.diff-table { width: 100%; border-collapse: collapse; }
.diff-table th, .diff-table td { padding: 10px 12px; border-bottom: 1px solid #EEE; font-size: 13px; text-align: left; }
.diff-table th { font-size: 10px; text-transform: uppercase; color: #9CA3AF; font-weight: 700; }
.diff-table .num { text-align: right; }
.diff-table td.up { color: #047857; font-weight: 700; }
.diff-table td.down { color: #B91C1C; font-weight: 700; }
</style>
