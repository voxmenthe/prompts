# Weighted Random

Have more control over the numbers you pick by providing this function an ease curve of your choice!

```
// reusable function. Feed in an array and an ease and it'll return
// a function that pulls a random element from that array, weighted
// according to the ease you provide.
function weightedRandom(collection, ease) {
return gsap.utils.pipe(
	Math.random,            //random number between 0 and 1
	gsap.parseEase(ease),   //apply the ease
	gsap.utils.mapRange(0, 1, -0.5, collection.length-0.5), //map to the index range of the array, stretched by 0.5 each direction because we'll round and want to keep distribution (otherwise linear distribution would be center-weighted slightly)
	gsap.utils.snap(1),     //snap to the closest integer
	i => collection[i]      //return that element from the array
);
}

// usage:
var myArray = [0, 1, 2, 3],
getRandom = weightedRandom(myArray, "power4");

// now you can call it anytime and it'll pull a random element from myArray, weighted toward the end.
getRandom();
getRandom();
...
```

info

For a deeper look at how to use the weightedRandom function, check out this video from the ["GSAP 3: Beyond the Basics" course](https://courses.snorkl.tv/courses/gsap3-beyond-the-basics?ref=44f484) by Snorkl.tv - one of the best ways to learn more about GSAP.
