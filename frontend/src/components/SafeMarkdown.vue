<template>
  <div class="safe-markdown" v-html="rendered"></div>
</template>

<script setup>
import { computed } from 'vue'

import { renderSafeMarkdown } from '../utils/markdown'

const props = defineProps({
  content: {
    type: [String, Object],
    default: '',
  },
})

const rendered = computed(() => renderSafeMarkdown(props.content))
</script>
