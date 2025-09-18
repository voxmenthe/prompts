# smoothChildTiming

### smoothChildTiming : Boolean

Controls whether or not child tweens and timelines are repositioned automatically (changing their `startTime`) in order to maintain smooth playback when properties are changed on-the-fly.

### Details[​](#details "Direct link to Details")

Controls whether or not child tweens and timelines are repositioned automatically (changing their `startTime`) in order to maintain smooth playback when properties are changed on-the-fly.

For example, imagine that the timeline's playhead is on a child tween that is 75% complete, moving `obj.x` from 0 to 100 and then that tween's `reverse()` method is called. If `smoothChildTiming` is `false` (the default except for the root timelines), the tween would flip in place, keeping its `startTime` consistent. Therefore the playhead of the timeline would now be at the tween's 25% completion point instead of 75%. Remember, the timeline's playhead position and direction are unaffected by child tween/timeline changes. `obj.x` would jump from 75 to 25, but the tween's position in the timeline would remain consistent.

However, if `smoothChildTiming` is `true`, that child tween's `startTime` would be adjusted so that the timeline's playhead intersects with the same spot on the tween (75% complete) as it had immediately before `reverse()` was called, thus playback appears perfectly smooth. `obj.x` would still be 75 and it would continue from there as the playhead moves on, but since the tween is reversed now `obj.x` will travel back towards 0 instead of 100. Ultimately it's a decision between prioritizing smooth on-the-fly playback (`true`) or consistent position(s) of child tweens and timelines (`false`).

Some examples of properties/methods that on-the-fly changes could affect `startTime` (when `smoothChildTiming` is `true`) : `reversed`, `timeScale`, `progress`, `totalProgress`, `time`, `totalTime`, `delay`, `pause`, `resume`, `duration`, and `totalDuration`.

The `gsap.globalTimeline` has `smoothChildTiming` set to `true`.
