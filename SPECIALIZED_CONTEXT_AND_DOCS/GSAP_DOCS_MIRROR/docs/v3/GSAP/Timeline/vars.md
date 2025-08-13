# vars

### vars : Object

The configuration object passed into the original timeline via the constructor, like `gsap.timeline({onComplete: func});`

### Details[​](#details "Direct link to Details")

The `vars` object passed into the constructor which contains all the properties/values you want a timeline to have.

* ### Property

  ### Description

  #### autoRemoveChildren[](#autoRemoveChildren)

  Boolean If `autoRemoveChildren` is set to `true`, as soon as child tweens/timelines complete, they will automatically get killed/removed. This is normally undesireable because it prevents going backwards in time (like if you want to `reverse()` or set the progress lower, etc.). It can, however, improve speed and memory management. The root timelines use `autoRemoveChildren: true`.

* #### callbackScope[](#callbackScope)

  Object The scope to be used for all of the callbacks (`onStart`, `onUpdate`, `onComplete`, etc.). The scope is what `this` refers to inside any of the callbacks.

* #### defaults[](#defaults)

  Object A simple way to set defaults that get inherited by the child animations. See the "[defaults](https://gsap.com/docs/v3/GSAP/Timeline#setting-defaults)" section for details.

* #### delay[](#delay)

  Number Amount of delay in seconds before the animation should begin.

* #### onComplete[](#onComplete)

  Function A function that should be called when the animation has completed.

* #### onCompleteParams[](#onCompleteParams)

  Array An array of parameters to pass the `onComplete` function. For example, `gsap.timeline({onComplete: myFunction, onCompleteParams: ["param1", "param2"]});`.

* #### onInterrupt[](#onInterrupt)

  A function to call when the animation is interrupted min animation. Note that this does not fire if the animation completes normally.

* #### onInterruptParams[](#onInterruptParams)

  An Array of parameters to pass the onInterrupt function. For example, `gsap.to(".class", {x:100, onInterrupt:myFunction, onInterruptParams:["param1", "param2"]});`.

* #### onRepeat[](#onRepeat)

  Function A function that should be called each time the animation repeats.

* #### onRepeatParams[](#onRepeatParams)

  Array An Array of parameters to pass the `onRepeat` function. For example, `gsap.timeline({onRepeat: myFunction, onRepeatParams: ["param1", "param2"]});`.

* #### onReverseComplete[](#onReverseComplete)

  Function A function that should be called when the animation has reached its beginning again from the reverse direction. For example, if `reverse()` is called the tween will move back towards its beginning and when its `time` reaches `0`, `onReverseComplete` will be called. This can also happen if the animation is placed in a timeline instance that gets reversed and plays the animation backwards to (or past) the beginning.

* #### onReverseCompleteParams[](#onReverseCompleteParams)

  Array An array of parameters to pass the `onReverseComplete` function. For example, `gsap.timeline({onReverseComplete: myFunction, onReverseCompleteParams: ["param1", "param2"]});`.

* #### onStart[](#onStart)

  Function A function that should be called when the animation begins (when its `time` changes from `0` to some other value which can happen more than once if the tween is restarted multiple times).

* #### onStartParams[](#onStartParams)

  Array An array of parameters to pass the `onStart` function. For example, `gsap.timeline({onStart: myFunction, onStartParams: ["param1", "param2"]});`.

* #### onUpdate[](#onUpdate)

  Function A function that should be called every time the animation updates (on every frame while the animation is active).

* #### onUpdateParams[](#onUpdateParams)

  Array An array of parameters to pass the `onUpdate` function. For example, `gsap.timeline({onUpdate: myFunction, onUpdateParams: ["param1", "param2"]});`.

* #### paused[](#paused)

  Boolean If `true`, the animation will pause itself immediately upon creation.

* #### repeat[](#repeat)

  Number Number of times that the animation should repeat after its first iteration. For example, if `repeat` is `1`, the animation will play a total of twice (the initial play plus 1 repeat). To repeat indefinitely, use `-1`. `repeat` should always be an integer.

* #### repeatDelay[](#repeatDelay)

  Number Amount of time in seconds between repeats. For example, if `repeat` is `2` and `repeatDelay` is `1`, the animation will play initially, then wait for 1 second before it repeats, then play again, then wait 1 second again before doing its final repeat.

* #### repeatRefresh[](#repeatRefresh)

  Setting `repeatRefresh: true` causes a repeating timeline to `invalidate()` all of its child tweens and re-record their starting/ending values internally on each full iteration (not including yoyo's). This is useful when you use dynamic values (relative, random, or function-based). For example, `x: "random(-100, 100)"` would get a new random x value on each repeat. `duration`, `delay`, and `stagger` do **NOT** refresh.

* #### smoothChildTiming[](#smoothChildTiming)

  Boolean Controls whether or not child animations are repositioned automatically (changing their `startTime`) in order to maintain smooth playback when timing-related properties are changed on-the-fly. For example, imagine that the timeline’s playhead is on a child tween that is 75% complete, moving element’s left from 0 to 100 and then that tween’s `reverse()` method is called. If `smoothChildTiming` is `false` (the default except for the globalTimeline), the tween would flip in place, keeping its `startTime` consistent. Therefore the playhead of the timeline would now be at the tween’s 25% completion point instead of 75%. See the "[How to timelines work?](https://gsap.com/docs/v3/GSAP/Timeline#mechanics)" section for details.

* #### yoyo[](#yoyo)

  Boolean If `true`, every other repeat cycle will run in the opposite direction so that the tween appears to go back and forth (forward then backward). This has no affect on the `reversed` property though. So if `repeat` is `2` and `yoyo` is `false`, it will look like: start - 1 - 2 - 3 - 1 - 2 - 3 - 1 - 2 - 3 - end. But if `yoyo` is `true`, it will look like: start - 1 - 2 - 3 - 3 - 2 - 1 - 1 - 2 - 3 - end.
