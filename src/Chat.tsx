import React, { useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import Message, { MessageWrapper } from './Message';
import * as state from './state';

const Chat = ({ incomingAvatar, outgoingAvatar }) => {
  const bottomRef = useRef(null);
  const messages = useRecoilValue(state.atoms.messages);

  useEffect(() => {
    if (bottomRef.current && messages.length > 0) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div style={{ overflow: 'auto' }}>
      {groupMessages(messages)}
      <div ref={bottomRef} />
    </div>
  );
};

function groupMessages(messages) {
  const els = [];
  let currentGroup = [];
  let idx = 0;
  for (const message of messages) {
    idx += 1;
    // group messages by type
    if (currentGroup.length > 0 && currentGroup[0].type !== message.type) {
      const groupMessage = currentGroup[0]
      els.push(
        <MessageWrapper avatar={groupMessage.avatar} type={groupMessage.type} index={idx} messages={currentGroup} />
      );
      currentGroup = [message];
    } else {
      currentGroup.push(message);
    }
  }
  if (currentGroup.length > 0) {
    const groupMessage = currentGroup[0]
    els.push(
      <MessageWrapper avatar={groupMessage.avatar} type={groupMessage.type} index={idx} messages={currentGroup} />
    );
  }
  return els
}


export default Chat;