import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { createPoller } from './poller.js'

describe('createPoller', () => {
  it('owns one interval and releases it on stop', () => {
    const scheduled = []
    const cleared = []
    const poller = createPoller(() => {}, 2500, {
      setIntervalFn: (callback, delay) => {
        scheduled.push({ callback, delay })
        return 17
      },
      clearIntervalFn: timer => cleared.push(timer),
    })

    poller.start()
    poller.start()

    assert.equal(scheduled.length, 1)
    assert.equal(scheduled[0].delay, 2500)
    assert.equal(poller.isRunning(), true)

    poller.stop()
    assert.deepEqual(cleared, [17])
    assert.equal(poller.isRunning(), false)
  })

  it('does not overlap asynchronous polling calls', async () => {
    let scheduledCallback
    let release
    let calls = 0
    const blocked = new Promise(resolve => { release = resolve })
    const poller = createPoller(async () => {
      calls += 1
      await blocked
    }, 1000, {
      setIntervalFn: callback => {
        scheduledCallback = callback
        return 1
      },
      clearIntervalFn: () => {},
    })
    poller.start()

    const first = scheduledCallback()
    const overlapping = scheduledCallback()
    assert.equal(calls, 1)

    release()
    await Promise.all([first, overlapping])
    await scheduledCallback()
    assert.equal(calls, 2)
  })
})
