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
import {useRecoilValue} from "recoil";
import * as state from "./state";
import { Actions } from "./Actions";

const PLUGIN_NAME = "@voxel51/fiftyone-gpt"

const ChatPanel = () => {
  const executor = useOperatorExecutor(`${PLUGIN_NAME}/send_message_to_gpt`);
  const messages = useRecoilValue(state.atoms.messages);
  const incomingAvatar = 'path-to-incoming-avatar';
  const outgoingAvatar = 'path-to-outgoing-avatar';
  const handleMessageSend = (message) => {
    executor.execute({ message })
  };
  const receiving = useRecoilValue(state.atoms.receiving);

  return (
    <Grid container direction="column" spacing={2} 
    justifyContent="flex-start"
    alignItems="center">
      {messages.length == 0 && <Grid item>
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
      </Grid>}
      <Grid item>
        <Chat incomingAvatar={incomingAvatar} outgoingAvatar={outgoingAvatar} />
      </Grid>
      <Grid item style={{marginTop: 'auto'}}>
        <Actions />
        <InputBar disabled={receiving} onMessageSend={handleMessageSend} />
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

