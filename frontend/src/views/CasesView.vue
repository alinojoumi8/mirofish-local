<template>
  <div class="cases-view">
    <header class="app-header">
      <div class="brand" @click="router.push('/')">MIROFISH OFFLINE</div>
      <div class="header-title">Cases</div>
      <button class="new-case-btn" @click="showCreate = true">+ New Case</button>
    </header>

    <main class="content">
      <div v-if="loading" class="state-msg">Loading cases…</div>
      <div v-else-if="error" class="state-msg error">{{ error }}</div>
      <div v-else-if="!cases.length" class="state-msg">
        No cases yet. Create one to track predictions across document versions.
      </div>

      <div v-else class="case-grid">
        <div
          v-for="c in cases"
          :key="c.case_id"
          class="case-card"
          @click="openCase(c.case_id)"
        >
          <div class="case-card-head">
            <span class="case-name">{{ c.name }}</span>
            <span class="case-versions mono">v{{ c.versions?.length || 0 }}</span>
          </div>
          <p class="case-req">{{ c.simulation_requirement || 'No requirement set.' }}</p>
          <div class="case-card-foot">
            <span class="case-status" :class="statusClass(c.latest_version?.status)">
              {{ c.latest_version?.status || 'empty' }}
            </span>
            <span class="case-date mono">{{ formatDate(c.updated_at) }}</span>
          </div>
        </div>
      </div>
    </main>

    <!-- Create case modal -->
    <div v-if="showCreate" class="modal-overlay" @click.self="showCreate = false">
      <div class="modal">
        <h3>New Case</h3>
        <label class="modal-field">
          <span>Case name</span>
          <input v-model="newName" placeholder="e.g. Smith v Jones" />
        </label>
        <label class="modal-field">
          <span>Prediction goal</span>
          <textarea v-model="newRequirement" rows="3" placeholder="What outcome should the case predict?"></textarea>
        </label>
        <div v-if="createError" class="state-msg error">{{ createError }}</div>
        <div class="modal-actions">
          <button class="btn-ghost" @click="showCreate = false">Cancel</button>
          <button class="btn-primary" :disabled="creating || !newName.trim()" @click="createNewCase">
            {{ creating ? 'Creating…' : 'Create' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { listCases, createCase } from '../api/graph'

const router = useRouter()

const cases = ref([])
const loading = ref(true)
const error = ref('')

const showCreate = ref(false)
const newName = ref('')
const newRequirement = ref('')
const creating = ref(false)
const createError = ref('')

const loadCases = async () => {
  loading.value = true
  error.value = ''
  try {
    const res = await listCases(100)
    if (res.success) {
      cases.value = res.data?.cases || res.data || []
    } else {
      error.value = res.error || 'Failed to load cases'
    }
  } catch (err) {
    error.value = err.message || 'Failed to load cases'
  } finally {
    loading.value = false
  }
}

const openCase = (caseId) => {
  router.push({ name: 'CaseDetail', params: { caseId } })
}

const createNewCase = async () => {
  creating.value = true
  createError.value = ''
  try {
    const res = await createCase({
      name: newName.value.trim(),
      simulation_requirement: newRequirement.value.trim()
    })
    if (res.success && res.data?.case_id) {
      showCreate.value = false
      router.push({ name: 'CaseDetail', params: { caseId: res.data.case_id } })
    } else {
      createError.value = res.error || 'Failed to create case'
    }
  } catch (err) {
    createError.value = err.message || 'Failed to create case'
  } finally {
    creating.value = false
  }
}

const statusClass = (status) => {
  if (status === 'reported') return 'reported'
  if (status === 'built' || status === 'ready') return 'ready'
  return 'pending'
}

const formatDate = (iso) => {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  } catch {
    return '—'
  }
}

onMounted(loadCases)
</script>

<style scoped>
.cases-view {
  min-height: 100vh;
  background: #FFF;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
  color: #111;
}
.app-header {
  height: 60px;
  border-bottom: 1px solid #EAEAEA;
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 0 24px;
}
.brand { font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 18px; letter-spacing: 1px; cursor: pointer; }
.header-title { font-weight: 700; color: #333; flex: 1; }
.new-case-btn {
  padding: 8px 16px; border: none; background: #111; color: #FFF;
  border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;
}
.new-case-btn:hover { background: #333; }

.content { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
.state-msg { color: #6B7280; font-size: 14px; padding: 24px 0; text-align: center; }
.state-msg.error { color: #B91C1C; }

.case-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.case-card {
  border: 1px solid #E5E7EB; border-radius: 10px; padding: 16px;
  cursor: pointer; transition: box-shadow 0.15s ease, border-color 0.15s ease; background: #FFF;
}
.case-card:hover { border-color: #111; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.case-card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.case-name { font-weight: 700; font-size: 15px; color: #111; }
.case-versions { font-size: 12px; font-weight: 700; color: #6B7280; }
.case-req {
  margin: 10px 0 14px; font-size: 12px; color: #6B7280; line-height: 1.5;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.case-card-foot { display: flex; justify-content: space-between; align-items: center; }
.case-status { font-size: 10px; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; }
.case-status.reported { color: #065F46; background: #D1FAE5; }
.case-status.ready { color: #1E40AF; background: #DBEAFE; }
.case-status.pending { color: #92400E; background: #FEF3C7; }
.case-date { font-size: 11px; color: #9CA3AF; }

.modal-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.4);
  display: flex; align-items: center; justify-content: center; z-index: 100;
}
.modal { background: #FFF; border-radius: 12px; padding: 24px; width: 420px; max-width: 90vw; }
.modal h3 { margin: 0 0 16px; font-size: 16px; }
.modal-field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; font-size: 12px; font-weight: 600; color: #374151; }
.modal-field input, .modal-field textarea {
  padding: 9px 11px; border: 1px solid #D1D5DB; border-radius: 6px; font-size: 13px; font-family: inherit;
}
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 8px; }
.btn-ghost { padding: 9px 16px; border: 1px solid #D1D5DB; background: #FFF; border-radius: 6px; cursor: pointer; font-size: 13px; }
.btn-primary { padding: 9px 16px; border: none; background: #111; color: #FFF; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
