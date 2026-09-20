import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, expect, it, vi } from 'vitest'
import Step3Simulation from './Step3Simulation.vue'
import { runEnsemble } from '../api/simulation'

vi.mock('vue-router', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('../api/report', () => ({ generateReport: vi.fn() }))
vi.mock('../api/simulation', () => ({
  startSimulation: vi.fn().mockResolvedValue({ success: true, data: {} }),
  stopSimulation: vi.fn().mockResolvedValue({ success: true }),
  getRunStatus: vi.fn().mockResolvedValue({ success: false }),
  getRunStatusDetail: vi.fn().mockResolvedValue({ success: false }),
  getRunDiagnostics: vi.fn().mockResolvedValue({ success: false }),
  getEconomySummary: vi.fn().mockResolvedValue({ success: false }),
  getEconomyEvents: vi.fn().mockResolvedValue({ success: false }),
  runEnsemble: vi.fn().mockResolvedValue({ success: true, data: { task_id: 'task_ensemble' } }),
  getEnsembleStatus: vi.fn(),
  getDiagnosticsExportUrl: vi.fn()
}))

let wrapper
afterEach(() => {
  wrapper?.unmount()
  vi.clearAllMocks()
})

it('passes the selected round cap and economic settings when running an ensemble', async () => {
  wrapper = mount(Step3Simulation, {
    props: {
      simulationId: 'sim_test',
      maxRounds: 17,
      ensembleRuns: 3,
      runtimeMode: 'realistic',
      economySettings: { enabled: true, initial_balance_cents: 2500, currency: 'CAD', max_decisions_per_tick: 8 }
    }
  })
  await flushPromises()
  await wrapper.get('button.ensemble').trigger('click')
  await flushPromises()
  expect(runEnsemble).toHaveBeenCalledWith({
    simulation_id: 'sim_test',
    runs: 3,
    max_rounds: 17,
    economy: { enabled: true, initial_balance_cents: 2500, currency: 'CAD', max_decisions_per_tick: 8 },
    enable_graph_memory_update: true
  })
})
