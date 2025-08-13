# parent

### parent : Timeline

The parent [Timeline](/docs/v3/GSAP/Timeline.md) to which the animation is attached. Anything that's not in a Timeline that you create is placed on the [gsap.globalTimeline](/docs/v3/GSAP/gsap.globalTimeline.md) by default.

### Details[​](#details "Direct link to Details")

The parent [Timeline](/docs/v3/GSAP/Timeline.md) to which the Timeline is attached. Anything that's not inside a Timeline that you create is placed on the [gsap.globalTimeline](/docs/v3/GSAP/gsap.globalTimeline\(\)) by default.

Each animation ([Tweens](/docs/v3/GSAP/Tween.md) and [Timelines](/docs/v3/GSAP/Timeline.md)) can only exist in one parent. Think of it like a DOM element that can't have multiple parents. If you [add()](/docs/v3/GSAP/Timeline/add\(\).md) an animation to a different Timeline, its `parent` will change to that Timeline.

## How do timelines work?[​](#how-do-timelines-work "Direct link to How do timelines work?")

See the [Timeline docs for details](/docs/v3/GSAP/Timeline.md#mechanics). It's very helpful to understand how the mechanics work conceptually.
