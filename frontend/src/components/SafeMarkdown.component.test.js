import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import SafeMarkdown from './SafeMarkdown.vue'


describe('SafeMarkdown', () => {
  it('mounts attacker-controlled HTML as inert text while retaining Markdown', () => {
    const wrapper = mount(SafeMarkdown, {
      props: {
        content: '<img src=x onerror="globalThis.pwned=true"> **trusted formatting**',
      },
    })

    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('<img src=x onerror="globalThis.pwned=true">')
    expect(wrapper.find('strong').text()).toBe('trusted formatting')
    expect(globalThis.pwned).toBeUndefined()
  })
})
