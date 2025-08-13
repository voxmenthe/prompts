# CustomWiggle

Quick Start

#### CDN Link

Copy

```
gsap.registerPlugin(CustomEase, CustomWiggle) 
```

#### Minimal usage

```
//Create a wiggle with 6 oscillations (default type:"easeOut")
CustomWiggle.create("myWiggle", {wiggles: 6});

//now use it in an ease. "rotation" will wiggle to 30 and back just as much in the opposite direction, ending where it began.
gsap.to(".class", {duration: 2, rotation: 30, ease: "myWiggle"});
```

### Description[​](#description "Direct link to Description")

CustomWiggle extends [CustomEase](/docs/v3/Eases/CustomEase.md) (which you must include in your project as well), and it lets you set a wiggle amount and type.

Ease walkthrough

[YouTube video player](https://www.youtube.com/embed/5lk5sLTd6N4)

### Demo[​](#demo "Direct link to Demo")

CustomWiggle Types

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/ebdbdadcd678a211681b4aa66cb58c4f?default-tab=result\&theme-id=41164)

### Config Object[​](#config-object "Direct link to Config Object")

* ### Property

  ### Description

  #### wiggles[](#wiggles)

  Integer - The number of oscillations back and forth. Default: 10.

* #### type[](#type)

  String (“easeOut” | “easeInOut” | “anticipate” | “uniform” | “random”) - The type (or style) of wiggle (see demo above). Default: “easeOut”.

* #### amplitudeEase[](#amplitudeEase)

  Ease Provides advanced control over the shape of the amplitude (y-axis in the [ease visualizer](/ease-visualizer/)). You define an ease that controls the amplitude’s progress from 1 toward 0 over the course of the tween. Defining an amplitudeEase (or timingEase) will override the “type” (think of the 5 “types” as convenient presets for amplitudeEase and timingEase combinations). See the [example CodePen](//codepen.io/GreenSock/pen/a8a7bc33cf80a74165dd966244a6cc00?editors=0010) to play around and visualize how it works.

* #### timingEase[](#timingEase)

  Ease Provides advanced control over how the waves are plotted over time (x-axis in the [ease visualizer](/ease-visualizer/)). Defining an timingEase (or amplitudeEase) will override the “type” (think of the 5 “types” as convenient presets for amplitudeEase and timingEase combinations). See the [example CodePen](//codepen.io/GreenSock/pen/a8a7bc33cf80a74165dd966244a6cc00?editors=0010) to play around and visualize how it works.

How do you control the strength of the wiggle (or how far it goes)? Simply by setting the tween property values themselves. For example, a wiggle to `rotation:30` would be stronger than `rotation:10`. Remember that an ease just controls the ratio of movement toward whatever value you supply for each property in your tween.

### Sample code[​](#sample-code "Direct link to Sample code")

```
gsap.registerPlugin(CustomEase, CustomWiggle); // register

//Create a wiggle with 6 oscillations (default type:"easeOut")
CustomWiggle.create("myWiggle", {wiggles: 6});

//now use it in an ease. "rotation" will wiggle to 30 and back just as much in the opposite direction, ending where it began.
gsap.to(".class", {duration: 2, rotation: 30, ease: "myWiggle"});

//Create a 10-wiggle anticipation ease:
CustomWiggle.create("funWiggle", {wiggles: 10, type: "anticipate"});
gsap.to(".class", {duration: 2, rotation: 30, ease: "funWiggle"});

//Alternatively, make sure CustomWiggle is loaded and use GSAP's string ease format
ease: "wiggle(15)" //<-- easy!
ease: "wiggle({type:anticipate, wiggles:8})" //advanced
```

Wiggling isn't just for "rotation"; you can use it for any property. For example, you could create a swarm effect by using just 2 randomized wiggle tweens on "x" and "y", as [demonstrated here](https://codepen.io/GreenSock/pen/wzkBYZ).

### Demo collection[​](#demo-collection "Direct link to Demo collection")

* [CustomWiggle demos](https://codepen.io/collection/AxZmqK)
