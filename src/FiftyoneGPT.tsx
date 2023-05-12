import { Selector } from "@fiftyone/components";
import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import { usePanelStatePartial, usePanelTitle } from "@fiftyone/spaces";
import React, { useEffect } from "react";
import styled from "styled-components";
import { scrollbarStyles } from "@fiftyone/utilities";
import { OperatorPlacements, registerOperator, useOperatorExecutor } from "@fiftyone/operators";
import Chat from "./Chat";
import { Grid, Paper, Typography } from '@mui/material';
import InputBar from "./InputBar";
import { ShowMessage } from "./ShowMessage";
import { SendMessageToGPT } from "./SendMessageToGPT";
const examples = [
  { type: 'incoming', content: 'Hello, **Dave**. It is nice to **meet** you.' },
  { type: 'incoming', content: `
\`\`\`js
console.log('hello')
\`\`\`
` },
  { type: 'incoming', content: null, button: {label: 'Click Here to let me out of Jail!'} },
];

const PLUGIN_NAME = "@voxel51/fiftyone-gpt"

const ChatPanel = () => {
  const executor = useOperatorExecutor(`${PLUGIN_NAME}/send_message_to_gpt`);
  const [messages, setMessages] = React.useState([]);
  const incomingAvatar = 'path-to-incoming-avatar';
  const outgoingAvatar = 'path-to-outgoing-avatar';
  const handleMessageSend = (message) => {
    executor.execute({ message })
  };

  return (
    <Grid container direction="column" spacing={2}>
      <Grid item>
        <Paper elevation={3} style={{ padding: '20px', marginBottom: '16px' }}>
          <Typography variant="h6" gutterBottom>
            Examples
          </Typography>
          <Typography variant="body1" gutterBottom>
            Here are some example questions you can ask:
          </Typography>
          <Typography variant="body2">
            - How does the GPT-3 model work?
            <br />
            - What's the weather like?
            <br />
            - How can I integrate GPT-3 into my application?
          </Typography>
        </Paper>
      </Grid>
      <Grid item style={{ flexGrow: 1 }}>
        <Chat incomingAvatar={incomingAvatar} outgoingAvatar={outgoingAvatar} />
        <InputBar onMessageSend={handleMessageSend} />
      </Grid>
    </Grid>
  );
};

registerComponent({
  name: "gpt_search",
  label: "Search with GPT",
  component: ChatPanel,
  type: PluginComponentType.Panel,
  activator: () => true
});

registerOperator(ShowMessage, PLUGIN_NAME)
registerOperator(SendMessageToGPT, PLUGIN_NAME)

