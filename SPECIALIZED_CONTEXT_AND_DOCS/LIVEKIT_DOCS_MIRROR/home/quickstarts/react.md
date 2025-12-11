LiveKit docs › LiveKit SDKs › Platform-specific quickstarts › React

---

# React quickstart

> Build a voice AI frontend with React in less than 10 minutes.

## Overview

This guide walks you through building a voice AI frontend using React and the LiveKit React components library. In less than 10 minutes, you'll have a working frontend that connects to your agent and allows users to have voice conversations through their browser.

## Starter project

The fastest way to get started with a full fledged agent experience is the React starter project. Click "Use this template" in the top right to create a new repo on GitHub, then follow the instructions in the project's README.

- **[Next.js Voice Agent](https://github.com/livekit-examples/agent-starter-react)**: A web voice AI assistant built with React and Next.js.

## Requirements

The following sections describe the minimum requirements to build a React frontend for your voice AI agent.

### LiveKit Cloud account

This guide assumes you have signed up for a free [LiveKit Cloud](https://cloud.livekit.io/) account. Create a free project to get started with your voice AI application.

### Agent backend

You need a LiveKit agent running on the backend that is configured for your LiveKit Cloud project. Follow the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to create and deploy your agent.

### Token server

You need a token server to generate authentication tokens for your users. For development and testing purposes, this guide uses a sandbox token server for ease of use. You can create one for your cloud project [here](https://cloud.livekit.io/projects/p_/sandbox/templates/token-server)

For production usage, you should set up a dedicated token server implementation. See the [generating tokens guide](https://docs.livekit.io/home/server/generating-tokens.md) for more details.

## Setup

Use the instructions in the following sections to set up your new React frontend project.

### Create React project

Create a new React project using your preferred method:

**pnpm**:

```shell
pnpm create vite@latest my-agent-app --template react-ts
cd my-agent-app

```

---

**npm**:

```shell
npm create vite@latest my-agent-app -- --template react-ts
cd my-agent-app

```

### Install packages

Install the LiveKit SDK and React components:

**pnpm**:

```shell
pnpm add @livekit/components-react @livekit/components-styles livekit-client

```

---

**npm**:

```shell
npm install @livekit/components-react @livekit/components-styles livekit-client --save

```

### Add agent frontend code

Replace the contents of your `src/App.tsx` file with the following code:

> ℹ️ **Note**
> 
> Update the `sandboxId` with your own sandbox token server ID, and set the `agentName` to match your deployed agent's name.

** Filename: `src/App.tsx`**

```tsx
'use client';
import { useEffect, useRef } from 'react';
import {
  ControlBar,
  RoomAudioRenderer,
  useSession,
  SessionProvider,
  useAgent,
  BarVisualizer,
} from '@livekit/components-react';
import { TokenSource, TokenSourceConfigurable, TokenSourceFetchOptions } from 'livekit-client';
import '@livekit/components-styles';

export default function App() {
  const tokenSource: TokenSourceConfigurable = useRef(
    TokenSource.sandboxTokenServer('my-token-server-id'),
  ).current;
  const tokenOptions: TokenSourceFetchOptions = { agentName: 'my-agent-name' };

  const session = useSession(tokenSource, tokenOptions);

  // Connect to session
  useEffect(() => {
    session.start();
    return () => {
      session.end();
    };
  }, []);

  return (
    <SessionProvider session={session}>
      <div data-lk-theme="default" style={{ height: '100vh' }}>
        {/* Your custom component with basic video agent functionality. */}
        <MyAgentView />
        {/* Controls for the user to start/stop audio and disconnect from the session */}
        <ControlBar controls={{ microphone: true, camera: false, screenShare: false }} />
        {/* The RoomAudioRenderer takes care of room-wide audio for you. */}
        <RoomAudioRenderer />
      </div>
    </SessionProvider>
  );
}

function MyAgentView() {
  const agent = useAgent();
  return (
    <div style={{ height: '350px' }}>
      <p>Agent state: {agent.state}</p>
      {/* Renders a visualizer for the agent's audio track */}
      {agent.canListen && (
        <BarVisualizer track={agent.microphoneTrack} state={agent.state} barCount={5} />
      )}
    </div>
  );
}

```

## Run your application

Start the development server:

**pnpm**:

```shell
pnpm dev

```

---

**npm**:

```shell
npm run dev

```

Open your browser to the URL shown in the terminal (typically `http://localhost:5173`). You should see your agent frontend with controls to enable your microphone and speak with your agent.

## Next steps

The following resources are useful for getting started with LiveKit on React.

- **[Generating tokens](https://docs.livekit.io/home/server/generating-tokens.md)**: Guide to generating authentication tokens for your users.

- **[Realtime media](https://docs.livekit.io/home/client/tracks.md)**: Complete documentation for live video and audio tracks.

- **[Realtime data](https://docs.livekit.io/home/client/data.md)**: Send and receive realtime data between clients.

- **[JavaScript SDK](https://github.com/livekit/client-sdk-js)**: LiveKit JavaScript SDK on GitHub.

- **[React components](https://github.com/livekit/components-js)**: LiveKit React components on GitHub.

- **[JavaScript SDK reference](https://docs.livekit.io/reference/client-sdk-js.md)**: LiveKit JavaScript SDK reference docs.

- **[React components reference](https://docs.livekit.io/reference/components/react.md)**: LiveKit React components reference docs.

---


For the latest version of this document, see [https://docs.livekit.io/home/quickstarts/react.md](https://docs.livekit.io/home/quickstarts/react.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).