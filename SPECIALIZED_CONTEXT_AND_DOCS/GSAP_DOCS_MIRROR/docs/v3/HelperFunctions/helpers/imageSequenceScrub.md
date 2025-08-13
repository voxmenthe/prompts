# Scrub through a `<canvas>` image sequence

Create an Array of image URLs, feed it to this helper function along with a reference to your `<canvas>` object and a ScrollTrigger config object to have it scrub through those, drawing the appropriate one to the canvas.

**DEMO**

#### loading...

[GSAP Basic Tween](https://codepen.io/GreenSock/embed/VwgevYW?default-tab=result\&theme-id=41164)

```
/*
Helper function that handles scrubbing through a sequence of images, drawing the appropriate one to the provided canvas.
Config object properties:
  - urls [Array]: an Array of image URLs
  - canvas [Canvas]: the <canvas> object to draw to
  - scrollTrigger [Object]: an optional ScrollTrigger configuration object like {trigger: "#trigger", start: "top top", end: "+=1000", scrub: true, pin: true}
  - onUpdate [Function]: optional callback for when the Tween updates (probably not used very often)

 Returns a Tween instance
*/
function imageSequence(config) {
  let playhead = {frame: 0},
      ctx = gsap.utils.toArray(config.canvas)[0].getContext("2d"),
      onUpdate = config.onUpdate,
      images,
      updateImage = function() {
        ctx.drawImage(images[Math.round(playhead.frame)], 0, 0);
        onUpdate && onUpdate.call(this);
      };
    images = config.urls.map((url, i) => {
      let img = new Image();
      img.src = url;
      i || (img.onload = updateImage);
      return img;
    });
    return gsap.to(playhead, {
      frame: images.length - 1,
      ease: "none",
      onUpdate: updateImage,
      scrollTrigger: config.scrollTrigger
    });
}
```

Usage:

```
imageSequence({
  urls, // Array of image URLs
  canvas: "#image-sequence", // <canvas> object to draw images to
  scrollTrigger: {
    start: 0,   // start at the very top
    end: "max", // entire page
    scrub: true // important!
  }
});
```
