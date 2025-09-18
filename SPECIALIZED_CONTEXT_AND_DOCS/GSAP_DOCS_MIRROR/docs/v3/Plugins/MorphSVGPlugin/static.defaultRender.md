# MorphSVGPlugin.defaultRender

### MorphSVGPlugin.defaultRender : Function

Sets the default function that should be called whenever a morphSVG tween updates. This is useful if you're rendering to `<canvas>`.

### Details[​](#details "Direct link to Details")

Sets the default function that should be called whenever a morphSVG tween updates. This is useful if you're rendering to `<canvas>`.

## Video explanation[​](#video-explanation "Direct link to Video explanation")

## Demo: MorphSVG canvas rendering[​](#demo-morphsvg-canvas-rendering "Direct link to Demo: MorphSVG canvas rendering")

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/WYWZab?default-tab=result\&theme-id=41164)

Here's an example of a tween and a render function that'd draw the morphing shape to canvas:

```
var canvas = document.querySelector("canvas"),
  ctx = canvas.getContext("2d"),
  vw = (canvas.width = window.innerWidth),
  vh = (canvas.height = window.innerHeight);
ctx.fillStyle = "#ccc";
MorphSVGPlugin.defaultRender = draw;
gsap.to("#hippo", { duration: 2, morphSVG: "#circle" });
function draw(rawPath, target) {
  var l, segment, j, i;
  ctx.clearRect(0, 0, vw, vh);
  ctx.beginPath();
  for (j = 0; j < rawPath.length; j++) {
    segment = rawPath[j];
    l = segment.length;
    ctx.moveTo(segment[0], segment[1]);
    for (i = 2; i < l; i += 6) {
      ctx.bezierCurveTo(
        segment[i],
        segment[i + 1],
        segment[i + 2],
        segment[i + 3],
        segment[i + 4],
        segment[i + 5]
      );
    }
    if (segment.closed) {
      ctx.closePath();
    }
  }
  ctx.fill("evenodd");
}
```
