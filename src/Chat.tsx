import React, { useEffect, useRef } from 'react';
import { useRecoilValue } from 'recoil';
import Message from './Message';
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
    <div style={{ overflow: 'auto', height: 'calc(100vh - 358px)' }}>
      {messages.map((message, index) => (
        <Message
          key={index}
          type={message.type}
          avatar={message.type === 'incoming' ? incomingAvatar : outgoingAvatar}
          content={message.content}
          button={message.button}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
};

export default Chat;