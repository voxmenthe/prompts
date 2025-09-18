# MotionPathPlugin.pointsToSegment

### MotionPathPlugin.pointsToSegment( points:Array, curviness:Number ) : Array

Plots a curved cubic bezier path through the provided x,y point coordinates, returning a segment Array that's typically dropped into a RawPath Array

#### Parameters

* #### **points**: Array

  An Array of alternating x, y, x, y point coordinates like \[x, y, x, y, x, y...]

* #### **curviness**: Number

  \[optional] This determines how "curvy" the resulting path is. So 0 would make straight lines (hard corners), 1 (the default) creates a nicely curved path, and 2 would make it much more curvy. Think of it like pulling the control points further and further outward from the anchors as the number goes higher.

### Returns : Array[​](#returns--array "Direct link to Returns : Array")

Cubic Bezier data in alternating x, y, x, y format (ordered like: anchor, two control points, anchor, two control points, anchor, etc.)

### Details[​](#details "Direct link to Details")

Plots a curved cubic bezier path through the provided x,y point coordinates, returning a segment Array that's typically dropped into a RawPath Array.
