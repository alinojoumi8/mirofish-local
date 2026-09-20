export const routes = [
  { path: '/', name: 'Home', component: () => import('../views/Home.vue') },
  { path: '/cases', name: 'Cases', component: () => import('../views/CasesView.vue') },
  {
    path: '/case/:caseId',
    name: 'CaseDetail',
    component: () => import('../views/CaseDetailView.vue'),
    props: true,
  },
  {
    path: '/case/:caseId/compare',
    name: 'CaseCompare',
    component: () => import('../views/CaseCompareView.vue'),
    props: true,
  },
  {
    path: '/process/:projectId',
    name: 'Process',
    component: () => import('../views/MainView.vue'),
    props: true,
  },
  {
    path: '/simulation/:simulationId',
    name: 'Simulation',
    component: () => import('../views/SimulationView.vue'),
    props: true,
  },
  {
    path: '/simulation/:simulationId/start',
    name: 'SimulationRun',
    component: () => import('../views/SimulationRunView.vue'),
    props: true,
  },
  {
    path: '/report/:reportId',
    name: 'Report',
    component: () => import('../views/ReportView.vue'),
    props: true,
  },
  {
    path: '/interaction/:reportId',
    name: 'Interaction',
    component: () => import('../views/InteractionView.vue'),
    props: true,
  },
]
