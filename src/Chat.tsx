import { Grid } from "@mui/material";
import { throttle } from "lodash";
import React, { useCallback, useEffect, useRef } from "react";
import { useRecoilValue } from "recoil";
import { MessageWrapper } from "./Message";
import { ChatGPTAvatar } from "./avatars";
import * as state from "./state";
import { SCROLL_TO_BOTTOM_THROTTLE } from "./constants";
import LoadingIndicator from "./LoadingIndicator";

const Chat = () => {
  const ref = useRef(null);
  const bottomRef = useRef(null);
  const messages = useRecoilValue(state.atoms.messages);
  const receiving = useRecoilValue(state.atoms.receiving);
  const waiting = useRecoilValue(state.atoms.waiting);

  const scrollToBottom = useCallback(
    throttle(() => {
      if (bottomRef.current && messages.length > 0) {
        bottomRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }, SCROLL_TO_BOTTOM_THROTTLE),
    [throttle]
  );

  useEffect(() => {
    const refElem = ref.current;
    if (refElem) {
      const refResizeObserver = new ResizeObserver(scrollToBottom)
      refResizeObserver.observe(
        refElem
      );

      return () => {
        refResizeObserver?.disconnect?.();
      };
    }
  }, []);

  const avatars = {
    incoming: <ChatGPTAvatar />,
  };

  const groupedMessages = groupConsecutiveMessages(messages, receiving, waiting);

  return (
    <div style={{ overflow: "auto" }} ref={ref}>
      <Grid container direction="row">
        {groupedMessages.map((group) => (
          <MessageWrapper {...group} />
        ))}
      </Grid>
      <div ref={bottomRef} />
    </div>
  );
};

// a function that groups consecutive messages of the same type
function groupConsecutiveMessages(messages, receiving, waiting) {
  const groups = [];
  let currentGroup = [];
  for (const message of messages) {
    // group messages by type
    if (currentGroup.length > 0 && currentGroup[0].type !== message.type) {
      groups.push({type: currentGroup[0].type, messages: currentGroup});
      currentGroup = [message];
    } else {
      currentGroup.push(message);
    }
  }
  if (currentGroup.length > 0) {
    groups.push({type: currentGroup[0].type, messages: currentGroup});
  }
  if (groups.length > 0) {
    const lastGroup = groups[groups.length - 1];
    lastGroup.last = true;
    if (lastGroup.type === "incoming") {
      lastGroup.receiving = receiving;
      lastGroup.waiting = waiting;
    } else {
      groups.push({type: "incoming", messages: [], receiving: true});
    }
  }
  return groups;
}

export default Chat;
