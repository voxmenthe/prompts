LiveKit docs › Recording & export › Overview

---

# Recording and export

> Use LiveKit's egress service to record or livestream a room.

## Introduction

LiveKit Egress gives you a powerful and consistent set of APIs to export any room or individual tracks from a LiveKit session. It supports recording to an MP4 file or HLS segments, as well as exporting to livestreaming services like YouTube Live, Twitch, and Facebook via RTMP.

For LiveKit Cloud customers, Egress is available for your project without any additional configuration. If you're self-hosting LiveKit, Egress must be [deployed](https://docs.livekit.io/home/self-hosting/egress.md) separately.

## Egress types

The egress service supports multiple types of exports for different use cases.

### Room composite egress

Export an entire room's video and/or audio using a web layout rendered by Chrome. Room composites are tied to a room's lifecycle, and stop automatically when the room ends. Composition templates are customizable web pages that can be hosted anywhere.

Example use case: recording a meeting for team members to watch later.

### Web egress

Record and export any web page. Web egress is similar to room composite egress, but _isn't_ tied to a LiveKit room and can record non-LiveKit content.

Example use case: restreaming content from a third-party source to YouTube and Twitch.

### Participant egress

Export a participant's video and audio together. This is a newer API and is designed to be easier to use than Track Composite Egress.

Example use case: record the teacher's video in an online class.

### Track composite egress

Sync and export one audio and one video track together. Transcoding and multiplexing happen automatically.

Example use case: exporting audio and video from many cameras at once during a production, for use in additional post-production.

### Track egress

Export individual tracks directly. Video tracks aren't transcoded.

Example use case: streaming an audio track to a captioning service via websocket.

## Service architecture

Depending on your request type, the egress service either launches a web template in Chrome and connects to the room (for example, for room composite requests), or it uses the SDK directly (for track and track composite requests). It uses GStreamer to encode, and can output to a file or to one or more streams.

![Egress instance](/images/diagrams/egress-instance.svg)

## Additional resources

The following topics provide more in-depth information about the various egress types.

- **[Room composite and web egress](https://docs.livekit.io/home/egress/composite-recording.md)**: Composite recording using a web-based recorder. Export an entire room or any web page.

- **[Participant and track composite egress](https://docs.livekit.io/home/egress/participant.md)**: Record a participant's audio and video tracks. Use TrackComposite egress for fine-grained control over tracks.

- **[Track egress](https://docs.livekit.io/home/egress/track.md)**: Export a single track without transcoding.

- **[Output and stream types](https://docs.livekit.io/home/egress/outputs.md)**: Sync and export one audio and one video track together. Transcoding and multiplexing happen automatically.

---


For the latest version of this document, see [https://docs.livekit.io/home/egress/overview.md](https://docs.livekit.io/home/egress/overview.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).