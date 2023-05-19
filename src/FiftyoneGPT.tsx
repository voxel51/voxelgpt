import { Selector } from "@fiftyone/components";
import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import { usePanelStatePartial, usePanelTitle } from "@fiftyone/spaces";
import React, { useEffect } from "react";
import styled from "styled-components";
import { scrollbarStyles } from "@fiftyone/utilities";
import {
  OperatorPlacements,
  registerOperator,
  useOperatorExecutor,
} from "@fiftyone/operators";
import Chat from "./Chat";
import { Grid, Typography, Link } from "@mui/material";
import InputBar from "./InputBar";
import { ShowMessage } from "./ShowMessage";
import { SendMessageToGPT } from "./SendMessageToGPT";
import { useRecoilValue } from "recoil";
import * as state from "./state";
import { Actions } from "./Actions";
import { Intro } from "./Intro";
import { ChatGPTAvatar } from "./avatars";

const PLUGIN_NAME = "@voxel51/fiftyone-gpt";

const ChatPanel = () => {
  const executor = useOperatorExecutor(`${PLUGIN_NAME}/send_message_to_gpt`);
  const messages = useRecoilValue(state.atoms.messages);
  const handleMessageSend = (message) => {
    executor.execute({ message });
  };
  const receiving = useRecoilValue(state.atoms.receiving);
  const hasMessages = messages.length > 0;

  return (
    <Grid
      container
      direction="row"
      spacing={2}
      sx={{ height: "100%" }}
      justifyContent="center"
    >
      {!hasMessages && <Intro />}
      {hasMessages && (
        <Grid item lg={12}>
          <Chat />
        </Grid>
      )}
      <Grid
        item
        container
        sx={{ marginTop: hasMessages ? "auto" : undefined }}
        justifyContent="center"
      >
        <Grid item sm={12} md={6} lg={8}>
          <Actions />
          <InputBar
            hasMessages={hasMessages}
            disabled={receiving}
            onMessageSend={handleMessageSend}
          />
          <Typography
            variant="caption"
            sx={{ marginTop: "8px", display: "block", textAlign: "center" }}
          >
            VoxelGPT is in beta and may not understand certain queries.{" "}
            <Link>Learn more</Link>
          </Typography>
        </Grid>
      </Grid>
    </Grid>
  );
};

registerComponent({
  name: "gpt_search",
  label: "VoxelGPT",
  component: ChatPanel,
  type: PluginComponentType.Panel,
  activator: () => true,
  Icon: () => <ChatGPTAvatar size={"1rem"} style={{ marginRight: "0.5rem" }} />,
});

registerOperator(ShowMessage, PLUGIN_NAME);
registerOperator(SendMessageToGPT, PLUGIN_NAME);
