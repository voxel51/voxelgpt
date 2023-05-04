import React, { useEffect, useRef } from 'react';
import Message from './Message';

const Chat = ({ messages, incomingAvatar, outgoingAvatar }) => {
  const bottomRef = useRef(null);

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