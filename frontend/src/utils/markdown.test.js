import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { renderSafeMarkdown } from './markdown.js'

describe('renderSafeMarkdown', () => {
  it('renders raw HTML and event handlers as inert text', () => {
    const rendered = renderSafeMarkdown('<img src=x onerror="globalThis.pwned=true"> **safe**')

    assert.doesNotMatch(rendered, /<img\b/i)
    assert.match(rendered, /&lt;img src=x onerror=&quot;globalThis\.pwned=true&quot;&gt;/)
    assert.match(rendered, /<strong>safe<\/strong>/)
  })

  it('escapes HTML inside code blocks while preserving Markdown formatting', () => {
    const rendered = renderSafeMarkdown('```html\n<script>alert(1)</script>\n```\n\n# Heading')

    assert.doesNotMatch(rendered, /<script\b/i)
    assert.match(rendered, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/)
    assert.match(rendered, /<h2 class="md-h2">Heading<\/h2>/)
  })
})
